use std::collections::{HashMap, HashSet};
use std::iter::zip;
use std::ops::Range;
use std::panic;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::geometry::{Ray, Vec3};

use crate::scene::Scene;

use byteorder::{LittleEndian, WriteBytesExt};
use futures_util::future::join_all;
use futures_util::stream::SplitSink;
use futures_util::{sink::Buffer, SinkExt, StreamExt};
use rand::random;
use rand::seq::IteratorRandom;
use serde::Deserialize;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio_tungstenite::{accept_async, WebSocketStream};
use tungstenite::{Error, Message};

pub struct Server {
    scenes: Arc<HashMap<String, Scene>>,
    connections: Arc<RwLock<HashSet<String>>>,
}

type Outgoing = Buffer<SplitSink<WebSocketStream<TcpStream>, Message>, Message>;

impl Server {
    const SEND_BUFFER_SIZE: usize = 4_000;
    // Range: 0-255
    const PIXELS_PER_MSG: usize = 30;
    const WIDTH: usize = 600;
    const HEIGHT: usize = 450;

    pub fn new(scenes: HashMap<String, Scene>) -> Self {
        Self {
            scenes: Arc::new(scenes),
            connections: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    pub async fn listen(self, port: &str) {
        let listener = TcpListener::bind(format!("127.0.0.1:{port}"))
            .await
            .unwrap();

        println!("Listening on port {port}.");
        while let Ok((connection, _)) = listener.accept().await {
            let websocket = accept_async(connection).await.unwrap();

            let id = self.unique_connection_id();
            self.connections.write().unwrap().insert(id.clone());

            let connections = Arc::clone(&self.connections);
            let scenes = Arc::clone(&self.scenes);
            tokio::spawn(async move {
                Server::handle_connection(&id, &scenes, websocket).await;
                connections.write().unwrap().remove(&id);
            });
        }
    }

    fn unique_connection_id(&self) -> String {
        let connections = self.connections.read().unwrap();
        const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";
        loop {
            let id = ALPHABET
                .chars()
                .choose_multiple(&mut rand::thread_rng(), 6)
                .into_iter()
                .collect::<String>();
            if !connections.contains(&id) {
                return id;
            }
        }
    }

    async fn handle_connection(
        id: &String,
        scenes: &Arc<HashMap<String, Scene>>,
        websocket: WebSocketStream<TcpStream>,
    ) {
        println!("[{id}] Accepted connection.");

        let (outgoing, mut incoming) = websocket.split();
        let outgoing = Arc::new(Mutex::new(outgoing.buffer(Self::SEND_BUFFER_SIZE)));

        // Toggling to false cancels the current render.
        let render_in_progress = Arc::new(AtomicBool::new(false));

        while let Some(Ok(msg)) = incoming.next().await {
            println!("[{id}] New message: '{msg}'");

            if let Message::Text(text) = msg {
                let msg: ClientMessage =
                    serde_json::from_str(text.as_str()).expect("failed to parse message");
                match (render_in_progress.load(Ordering::SeqCst), msg) {
                    (false, ClientMessage::Render { scene }) => {
                        render_in_progress.store(true, Ordering::SeqCst);
                        let render_in_progress = Arc::clone(&render_in_progress);
                        let id = id.clone();
                        let outgoing = Arc::clone(&outgoing);
                        let scenes = Arc::clone(&scenes);
                        tokio::spawn(async move {
                            println!("[{id}] Rendering...");
                            let scene = scenes.get(&scene).unwrap();
                            Self::render_to_websocket(
                                &scene,
                                Self::WIDTH,
                                Self::HEIGHT,
                                4,
                                &outgoing,
                                &render_in_progress,
                            )
                            .await;
                            outgoing.lock().await.flush().await.unwrap();
                            render_in_progress.store(false, Ordering::SeqCst);
                            println!("[{id}] Done rendering.");
                        });
                    }
                    (true, ClientMessage::StopRendering) => {
                        render_in_progress.store(false, Ordering::SeqCst);
                        println!("[{id}] Render cancelled.");
                    }
                    _ => {}
                }
            }
        }

        println!("[{id}] Disconnected.");
    }

    async fn render_to_websocket(
        scene: &Scene,
        width: usize,
        height: usize,
        samples_per_pixel: usize,
        outgoing: &Arc<Mutex<Outgoing>>,
        render_in_progress: &Arc<AtomicBool>,
    ) {
        join_all((0..10).map(|t| {
            let outgoing = Arc::clone(outgoing);
            let render_in_progress = Arc::clone(render_in_progress);
            async move {
                for y in (t * height / 10)..((t + 1) * height / 10) {
                    for x in (0..width).step_by(Self::PIXELS_PER_MSG) {
                        if !render_in_progress.load(Ordering::SeqCst) {
                            return;
                        }
                        let num_pixels = Self::PIXELS_PER_MSG.min(width - x);
                        let mut msg = Vec::<u8>::with_capacity(2 + 3 * num_pixels);
                        msg.push(0);
                        msg.push(num_pixels as u8);
                        msg.write_u16::<LittleEndian>(x as u16).unwrap();
                        msg.write_u16::<LittleEndian>(y as u16).unwrap();
                        for x_ in x..x + num_pixels {
                            let pixel = sample_pixel(
                                x_,
                                height - y - 1,
                                width,
                                height,
                                samples_per_pixel,
                                scene,
                            );
                            Self::serialize_rendered_pixel(&mut msg, pixel);
                        }
                        Self::send(&outgoing, msg).await;
                    }
                }
            }
        }))
        .await;
    }

    async fn send(outgoing: &Arc<Mutex<Outgoing>>, msg: Vec<u8>) {
        let msg = Message::Binary(msg);
        let mut outgoing = outgoing.lock().await;
        if let Err(err) = outgoing.feed(msg).await {
            match err {
                Error::AlreadyClosed => {}
                Error::ConnectionClosed => {}
                _ => {
                    eprintln!("{err}");
                }
            }
        }
    }

    // Serialization format:
    //   MsgLength = 2 + 8 * NumPixels
    //
    //   HEADER (2 bytes)
    //        0: (u8, always 0) Message type
    //        1: (u8) Number of records
    //
    //   PIXEL i (8 bytes)
    //    3*i+6: r (u8)
    //    3*i+7: g (u8)
    //    3*i+8: b (u8)
    fn serialize_rendered_pixel(data: &mut Vec<u8>, Vec3 { x: r, y: g, z: b }: Vec3) {
        // data.write_u16::<LittleEndian>(x as u16).unwrap();
        // data.write_u16::<LittleEndian>(y as u16).unwrap();
        data.push(r as u8);
        data.push(g as u8);
        data.push(b as u8);
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
enum ClientMessage {
    Render { scene: String },
    StopRendering,
}

struct Range2 {
    x: usize,
    y: usize,
    max_x: usize,
    max_y: usize,
}

impl Range2 {
    pub fn new(max_x: usize, max_y: usize) -> Self {
        Self {
            x: 0,
            y: 0,
            max_x,
            max_y,
        }
    }
}

impl Iterator for Range2 {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.y < self.max_y {
            let coord = (self.y, self.x);
            self.x += 1;
            if self.x == self.max_x {
                self.y += 1;
                self.x = 0;
            }
            Some(coord)
        } else {
            None
        }
    }
}

fn sample_pixel(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    samples_per_pixel: usize,
    scene: &Scene,
) -> Vec3 {
    let w = width as f64;
    let h = height as f64;
    let cx = Vec3::new(w * 0.5135 / h, 0., 0.);
    let cy = cx.cross(&scene.camera.dir).norm() * 0.5135;
    let num_samples = samples_per_pixel / 4;

    let mut pixel = Vec3::zero();
    for sy in 0..2 {
        for sx in 0..2 {
            let mut rad = Vec3::zero();
            for _ in 0..num_samples {
                let r1 = 2. * random::<f64>();
                let dx = if r1 < 1. {
                    r1.sqrt() - 1.
                } else {
                    1. - (2. - r1).sqrt()
                };

                let r2 = 2. * random::<f64>();
                let dy = if r2 < 1. {
                    r2.sqrt() - 1.
                } else {
                    1. - (2. - r2).sqrt()
                };

                let d = cx * (((sx as f64 + 0.5 + dx) / 2. + x as f64) / w - 0.5)
                    + cy * (((sy as f64 + 0.5 + dy) / 2. + y as f64) / h - 0.5)
                    + scene.camera.dir;

                rad += scene.received_radiance(&Ray::new(scene.camera.pos, d.norm()))
                    * (1. / num_samples as f64);
            }
            pixel += rad.clamp(0., 1.) * 0.25;
        }
    }
    gamma_correct(&pixel)
}

fn gamma_correct(v: &Vec3) -> Vec3 {
    v.clamp(0., 1.).powf(1.0 / 2.2) * 255.0 + Vec3::repeat(0.5)
}
