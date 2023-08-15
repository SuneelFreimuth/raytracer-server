#[allow(dead_code)]
use std::collections::{HashMap, HashSet};
use std::env::args;
use std::fs::File;
use std::io::BufReader;
use std::iter::zip;
use std::panic;
use std::process::exit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

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

mod geometry;
use geometry::{MeshLoadError, Ray, Vec3};

mod scene;
use scene::{LoadTomlError, Scene};
use tungstenite::error::ProtocolError;

const PORT: &str = "8080";
const WIDTH: usize = 600;
const HEIGHT: usize = 450;
const SCENE_NAMES: [&str; 3] = ["cornell_box", "cubes", "flying_unicorn"];

#[tokio::main]
async fn main() {
    let args = args().collect::<Vec<String>>();
    if args.len() < 2 {
        panic!("Usage: raytracer-server <scenes directory>");
    }

    let scene_dir = &args[1];

    let scenes = Arc::new(HashMap::from_iter(SCENE_NAMES.map(|name| {
        (
            name.to_string(),
            load_scene(&format!("{scene_dir}/{name}.toml")),
        )
    })));

    let connections = Arc::new(RwLock::new(HashSet::<String>::new()));
    let listener = TcpListener::bind(format!("127.0.0.1:{PORT}"))
        .await
        .unwrap();

    println!("Listening on port {PORT}.");
    while let Ok((connection, _)) = listener.accept().await {
        let websocket = accept_async(connection).await.unwrap();

        let id = unique_connection_id(&connections.read().unwrap());
        connections.write().unwrap().insert(id.clone());

        let connections = Arc::clone(&connections);
        let scenes = Arc::clone(&scenes);
        tokio::spawn(async move {
            handle_connection(&id, &scenes, websocket).await;
            connections.write().unwrap().remove(&id);
        });
    }
}

fn load_scene(path: &String) -> Scene {
    let mut f = BufReader::new(File::open(path).unwrap());
    match Scene::from_toml(&mut f) {
        Ok(scene) => scene,
        Err(LoadTomlError::Io(err)) | Err(LoadTomlError::MeshLoad(MeshLoadError::IO(err))) => {
            eprintln!("Failed to load scene {path}: {err}");
            exit(1);
        }
        Err(_) => {
            eprintln!("Failed to load scene {path}.");
            exit(1);
        }
    }
}

fn unique_connection_id(connections: &HashSet<String>) -> String {
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

type Outgoing = Buffer<SplitSink<WebSocketStream<TcpStream>, Message>, Message>;

const SEND_BUFFER_SIZE: usize = 1_000;

async fn handle_connection(
    id: &String,
    scenes: &Arc<HashMap<String, Scene>>,
    websocket: WebSocketStream<TcpStream>,
) {
    println!("[{id}] Accepted connection.");

    let (outgoing, mut incoming) = websocket.split();
    let outgoing = Arc::new(Mutex::new(outgoing.buffer(SEND_BUFFER_SIZE)));

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
                        render_to_websocket(
                            &scene,
                            WIDTH,
                            HEIGHT,
                            16,
                            &outgoing,
                            &render_in_progress,
                        )
                        .await;
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

struct CancellationToken {
    cancelled: AtomicBool,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self {
            cancelled: AtomicBool::default(),
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
enum ClientMessage {
    Render { scene: String },
    StopRendering,
}

async fn render_to_websocket(
    scene: &Scene,
    width: usize,
    height: usize,
    samples_per_pixel: usize,
    outgoing: &Arc<Mutex<Outgoing>>,
    render_in_progress: &Arc<AtomicBool>,
) {
    join_all(Range2::new(WIDTH, HEIGHT).map(|(y, x)| {
        let outgoing = Arc::clone(outgoing);
        let render_in_progress = Arc::clone(render_in_progress);
        async move {
            if !render_in_progress.load(Ordering::SeqCst) {
                return;
            }

            let pixel = sample_pixel(x, height - y - 1, width, height, samples_per_pixel, scene);

            let mut outgoing = outgoing.lock().await;
            let msg = Message::Binary(serialize_pixel(x, y, pixel));
            if let Err(err) = outgoing.feed(msg).await {
                eprintln!("{err}");
                return;
            }
        }
    }))
    .await;
    {
        let mut outgoing = outgoing.lock().await;
        outgoing.flush().await.unwrap();
    }
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

fn serialize_pixel(x: usize, y: usize, Vec3 { x: r, y: g, z: b }: Vec3) -> Vec<u8> {
    let mut msg: Vec<u8> = Vec::with_capacity(8);
    msg.push(0);
    msg.write_u16::<LittleEndian>(x as u16).unwrap();
    msg.write_u16::<LittleEndian>(y as u16).unwrap();
    msg.push(r as u8);
    msg.push(g as u8);
    msg.push(b as u8);
    msg
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
