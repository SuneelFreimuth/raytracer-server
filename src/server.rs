use std::collections::{HashMap, HashSet};
use std::iter::zip;
use std::ops::Range;
use std::{panic, thread};
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
use tokio_tungstenite::tungstenite::{Bytes, Error, Message};


pub struct Server {
    scenes: Arc<HashMap<String, Scene>>,
    connections: Arc<Mutex<HashSet<String>>>,
}

impl Server {
    const WIDTH: i32 = 600;
    const HEIGHT: i32 = 450;

    pub fn new(scenes: HashMap<String, Scene>) -> Self {
        Self {
            scenes: Arc::new(scenes),
            connections: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    pub async fn listen(self, port: &str) {
        let listener = TcpListener::bind(format!("0.0.0.0:{port}"))
            .await
            .unwrap();
        println!("Listening on port {port}.");
        while let Ok((connection, _)) = listener.accept().await {
            let id = self.generate_connection_id().await;
            let scenes = Arc::clone(&self.scenes);
            let websocket = accept_async(connection).await.unwrap();
            let connections = Arc::clone(&self.connections);
            tokio::spawn(async move {
                Server::handle_connection(&id, scenes, websocket).await;
                connections.lock().await.remove(&id);
            });
        }
    }

    async fn generate_connection_id(&self) -> String {
        let mut connections = self.connections.lock().await;
        const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";
        let id = loop {
            let random_id = ALPHABET
                .chars()
                .choose_multiple(&mut rand::thread_rng(), 6)
                .into_iter()
                .collect::<String>();
            if !connections.contains(&random_id) {
                break random_id;
            }
        };
        connections.insert(id.clone());
        id
    }

    async fn handle_connection(
        id: &String,
        scenes: Arc<HashMap<String, Scene>>,
        websocket: WebSocketStream<TcpStream>,
    ) {
        println!("[{id}] Accepted connection.");
        let (outgoing, mut incoming) = websocket.split();
        let render_job = Arc::new(RenderJob::new(outgoing));
        while let Some(Ok(msg)) = incoming.next().await {
            println!("[{id}] New message: '{msg}'");
            if let Message::Text(text) = msg {
                let msg: ClientMessage =
                    serde_json::from_str(text.as_str()).expect("failed to parse message");
                match (render_job.running(), msg) {
                    (false, ClientMessage::Render { scene, spp }) => {
                        let id = id.clone();
                        let scenes = Arc::clone(&scenes);
                        let render_job = Arc::clone(&render_job);
                        tokio::spawn(async move {
                            println!("[{id}] Rendering...");
                            let scene = scenes.get(&scene).unwrap();
                            let cancelled_early =
                                render_job.run(&scene, Self::WIDTH, Self::HEIGHT, spp).await;
                            if !cancelled_early {
                                println!("[{id}] Done rendering.");
                            }
                        });
                    }
                    (true, ClientMessage::StopRendering) => {
                        render_job.stop();
                        println!("[{id}] Render cancelled.");
                    }
                    _ => {}
                }
            }
        }
        println!("[{id}] Disconnected.");
    }
}


#[derive(Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
enum ClientMessage {
    Render { scene: String, spp: i32 },
    StopRendering,
}


type Outgoing = SplitSink<WebSocketStream<TcpStream>, Message>;

// Serialization format for outgoing WebSocket messages:
//   Message Length: 2 + 3 * NumPixels
//
//        [0]  Message Type (u8)
//        [1]  Num Records (u8)
//     [3i+6]  r (u8)
//     [3i+7]  g (u8)
//     [3i+8]  b (u8)
struct RenderJob {
    outgoing: Arc<Mutex<Outgoing>>,
    cancel_token: Arc<CancellationToken>,
}

impl RenderJob {
    const PIXELS_PER_MSG: i32 = 60;

    pub fn new(outgoing: Outgoing) -> Self {
        let cancelled_token = CancellationToken::new();
        cancelled_token.cancel();
        Self {
            outgoing: Arc::new(Mutex::new(outgoing)),
            cancel_token: Arc::new(cancelled_token),
        }
    }

    // Returns whether the run was stopped before its completion
    pub async fn run(
        &self,
        scene: &Scene,
        width: i32,
        height: i32,
        samples_per_pixel: i32,
    ) -> bool {
        self.cancel_token.reset();
        let num_tasks = thread::available_parallelism().unwrap().get() as i32;
        join_all((0..num_tasks).map(|t| async move {
            // Assumes height is evenly divisible by NUM_TASKS.
            for y in (t * height / num_tasks)..((t + 1) * height / num_tasks) {
                for (x, num_pixels) in windows(0, width, Self::PIXELS_PER_MSG) {
                    if self.cancel_token.is_cancelled() {
                        return;
                    }
                    let mut msg = Vec::<u8>::with_capacity(6 + 3 * num_pixels as usize);
                    msg.push(0);
                    msg.push(num_pixels as u8);
                    msg.write_u16::<LittleEndian>(x as u16).unwrap();
                    msg.write_u16::<LittleEndian>(y as u16).unwrap();
                    for x_ in x..x + num_pixels {
                        let Vec3 { x: r, y: g, z: b } = sample_pixel(
                            x_,
                            height - y - 1,
                            width,
                            height,
                            samples_per_pixel,
                            scene,
                        );
                        msg.write_u8(r as u8).unwrap();
                        msg.write_u8(g as u8).unwrap();
                        msg.write_u8(b as u8).unwrap();
                    }
                    self.send(msg).await;
                }
            }
        }))
        .await;
        self.outgoing.lock().await.flush().await.unwrap();
        self.cancel_token.cancel()
    }

    pub fn stop(&self) {
        _ = self.cancel_token.cancel();
    }

    pub fn running(&self) -> bool {
        !self.cancel_token.is_cancelled()
    }

    async fn send(&self, msg: Vec<u8>) {
        let msg = Message::Binary(Bytes::from(msg));
        let mut outgoing = self.outgoing.lock().await;
        if let Err(err) = outgoing.feed(msg).await {
            match err {
                Error::AlreadyClosed | Error::ConnectionClosed => {
                    self.cancel_token.cancel();
                }
                _ => {
                    eprintln!("{err}");
                }
            }
        }
    }
}


struct CancellationToken {
    cancelled: AtomicBool,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    // Returns whether the token was already cancelled.
    pub fn cancel(&self) -> bool {
        self.cancelled
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
    }

    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
    }
}


fn windows(start: i32, end: i32, window_size: i32) -> Windows {
    Windows {
        i: start,
        end,
        window_size,
    }
}

struct Windows {
    i: i32,
    end: i32,
    window_size: i32,
}

impl Iterator for Windows {
    type Item = (i32, i32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.end {
            let result = (self.i, self.window_size.min(self.end - self.i));
            self.i += self.window_size;
            Some(result)
        } else {
            None
        }
    }
}


struct Range2 {
    x: i32,
    y: i32,
    max_x: i32,
    max_y: i32,
}

impl Range2 {
    pub fn new(max_x: i32, max_y: i32) -> Self {
        Self {
            x: 0,
            y: 0,
            max_x,
            max_y,
        }
    }
}

impl Iterator for Range2 {
    type Item = (i32, i32);

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
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    samples_per_pixel: i32,
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
