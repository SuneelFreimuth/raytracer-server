use std::collections::{hash_map, HashMap, HashSet};
#[allow(dead_code)]
use std::env::args;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::ops::Deref;
use std::process::exit;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::thread::{self, sleep, Scope, ScopedJoinHandle};
use std::time::{Duration, Instant};

use image::codecs::png::PngEncoder;
use image::ColorType;
use image::ImageEncoder;
use pixels::{Pixels, SurfaceTexture};
use rand::random;
use rand::seq::{IteratorRandom, SliceRandom};
use serde::Deserialize;
use tokio_tungstenite::WebSocketStream;
use tokio_util::sync::CancellationToken;
use winit::{
    dpi::LogicalSize,
    event::Event,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use futures_util::{stream::TryStreamExt, StreamExt};
use futures::lock::Mutex as AsyncMutex;

use byteorder::{LittleEndian, WriteBytesExt};

mod geometry;
use geometry::{Ray, Vec3};

mod scene;
use scene::Scene;

use std::net::{TcpStream};
use std::thread::spawn;
use tungstenite::{accept, Error, Message, WebSocket};

const CONFIG_FILE: &str = "config.toml";

const PORT: &str = "8080";

const WIDTH: usize = 600;
const HEIGHT: usize = 450;

#[tokio::main]
async fn main() {
    let server = TcpListener::bind(format!("127.0.0.1:{PORT}")).await.unwrap();

    let scenes = Arc::new(HashMap::from([
        ("cornell_box", {
            let mut f = BufReader::new(File::open("scenes/cornell_box.toml").unwrap());
            Scene::from_toml(&mut f).unwrap()
        }),
        ("cubes", {
            let mut f = BufReader::new(File::open("scenes/cubes.toml").unwrap());
            Scene::from_toml(&mut f).unwrap()
        }),
        ("flying_unicorn", {
            let mut f = BufReader::new(File::open("scenes/flying_unicorn.toml").unwrap());
            Scene::from_toml(&mut f).unwrap()
        }),
    ]));

    let connections = Arc::new(RwLock::new(HashSet::<String>::new()));

    while let Ok((stream, addr)) = server.accept().await {
        let id = unique_connection_id(connections.read().unwrap().deref());
        let scenes = scenes.as_ref();
        let websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
        handle_connection(&id, &scenes, websocket).await;
        connections.write().unwrap().remove(&id);
        connections.write().unwrap().insert(id.clone());
    }
}

fn unique_connection_id(connections: &HashSet<String>) -> String {
    const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";
    loop {
        let id: String = ALPHABET
            .chars()
            .choose_multiple(&mut rand::thread_rng(), 6)
            .into_iter()
            .collect();
        if !connections.contains(&id) {
            return id;
        }
    }
}

async fn handle_connection(
    id: &String,
    scenes: &HashMap<&str, Scene>,
    websocket: WebSocketStream<TcpStream>,
) {
    println!("[{id}] Accepted connection.");
    let (outgoing, incoming) = websocket.split();
    
    let token = CancellationToken::new();

    loop {
        let msg = {
            let mut websocket = websocket.lock().unwrap();
            match websocket {
                Ok(msg) => msg,
                Err(Error::ConnectionClosed) => {
                    println!("[{id}] Connection closed.");
                    break;
                }
                Err(err) => {
                    println!("[{id}] Unexpected error reading message: {err}");
                    break;
                }
            }
        };

        println!("[{id}] New message: '{msg}'");

        if let Message::Text(text) = msg {
            let msg: ClientMessage =
                serde_json::from_str(text.as_str()).expect("failed to parse message");
            match msg {
                ClientMessage::Render { scene } => {
                    let token = token.clone();
                    tokio::spawn(async move {
                        println!("[{id}] Rendering...");
                        render_to_websocket(
                            scenes.get(scene.as_str()).unwrap(),
                            WIDTH,
                            HEIGHT,
                            16,
                            &websocket,
                            token,
                        );
                        println!("[{id}] Done rendering.");
                        lock_and_use(&websocket, |mut websocket| {
                            websocket.send(Message::Binary(vec![1]));
                        });
                    });
                }
                ClientMessage::StopRendering => {
                    token.cancel();
                }
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
enum ClientMessage {
    Render { scene: String },
    StopRendering,
}

fn lock_and_use<T: ?Sized, R, F>(m: &Mutex<T>, f: F) -> R
where
    F: Fn(MutexGuard<'_, T>) -> R,
{
    let lock = m.lock().unwrap();
    f(lock)
}

fn gamma_correct(v: &Vec3) -> Vec3 {
    v.clamp(0., 1.).powf(1.0 / 2.2) * 255.0 + Vec3::repeat(0.5)
}

fn render_to_websocket(
    scene: &Scene,
    width: usize,
    height: usize,
    samples_per_pixel: usize,
    websocket: &Arc<Mutex<WebSocket<TcpStream>>>,
    token: CancellationToken
) {
    let w = width as f64;
    let h = height as f64;
    let cx = Vec3::new(w * 0.5135 / h, 0., 0.);
    let cy = cx.cross(&scene.camera.dir).norm() * 0.5135;
    let num_samples = samples_per_pixel / 4;
    let pixels_rendered = Arc::new(Mutex::new(0u64));

    let parallelism: usize = thread::available_parallelism()
        .expect("could not query number of cores")
        .into();
    let start = Instant::now();
    thread::scope(|s| {
        for i in 0..parallelism {
            let websocket = websocket.clone();
            let pixels_rendered = Arc::clone(&pixels_rendered);
            let token = token.clone();
            s.spawn(move || {
                let min_y = height * i / parallelism;
                let max_y = height * (i + 1) / parallelism;
                for y in min_y..max_y {
                    let y = height - y - 1;
                    for x in 0..width {
                        if token.is_cancelled() {
                            return;
                        }
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

                                    let d = cx
                                        * (((sx as f64 + 0.5 + dx) / 2. + x as f64) / w - 0.5)
                                        + cy * (((sy as f64 + 0.5 + dy) / 2. + y as f64) / h - 0.5)
                                        + scene.camera.dir;

                                    rad += scene
                                        .received_radiance(&Ray::new(scene.camera.pos, d.norm()))
                                        * (1. / num_samples as f64);
                                }
                                pixel += rad.clamp(0., 1.) * 0.25;
                            }
                        }
                        pixel = gamma_correct(&pixel);

                        let Vec3 { x: r, y: g, z: b } = pixel;

                        lock_and_use(&websocket, |mut websocket| {
                            let y = height - y - 1;
                            // websocket.write(message("pixel", &format!("{x},{y},{r},{g},{b}")));
                            let mut msg: Vec<u8> = Vec::with_capacity(8);
                            msg.push(0);
                            msg.write_u16::<LittleEndian>(x as u16).unwrap();
                            msg.write_u16::<LittleEndian>(y as u16).unwrap();
                            msg.push(r as u8);
                            msg.push(g as u8);
                            msg.push(b as u8);
                            websocket.write(Message::Binary(msg));
                        });

                        {
                            let mut pixels_rendered = pixels_rendered.lock().unwrap();
                            *pixels_rendered += 1;
                            print!(
                                "\rRendering at {samples_per_pixel} spp ({:.1}%)",
                                *pixels_rendered as f64 / (w * h) * 100.
                            );
                        }
                    }
                }
            });
        }
    });
    lock_and_use(&websocket, |mut w| w.flush().unwrap());
    print!("\n");
    let duration = Instant::now() - start;
    println!("Rendered in {:.1} seconds.", duration.as_secs_f64());
}

fn render(
    scene: &Scene,
    width: usize,
    height: usize,
    samples_per_pixel: usize,
    buffer: Arc<RwLock<Vec<u8>>>,
) {
    let w = width as f64;
    let h = height as f64;
    let cx = Vec3::new(w * 0.5135 / h, 0., 0.);
    let cy = cx.cross(&scene.camera.dir).norm() * 0.5135;
    let num_samples = samples_per_pixel / 4;
    let pixels_rendered = Arc::new(Mutex::new(0u64));

    let parallelism: usize = thread::available_parallelism()
        .expect("could not query number of cores")
        .into();
    let start = Instant::now();
    thread::scope(|s| {
        for i in 0..parallelism {
            let pixels_rendered = Arc::clone(&pixels_rendered);
            let buffer = Arc::clone(&buffer);
            s.spawn(move || {
                let min_y = height * i / parallelism;
                let max_y = height * (i + 1) / parallelism;
                for y in min_y..max_y {
                    let y = height - y - 1;
                    for x in 0..width {
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

                                    let d = cx
                                        * (((sx as f64 + 0.5 + dx) / 2. + x as f64) / w - 0.5)
                                        + cy * (((sy as f64 + 0.5 + dy) / 2. + y as f64) / h - 0.5)
                                        + scene.camera.dir;

                                    rad += scene
                                        .received_radiance(&Ray::new(scene.camera.pos, d.norm()))
                                        * (1. / num_samples as f64);
                                }
                                pixel += rad.clamp(0., 1.) * 0.25;
                            }
                        }
                        pixel = gamma_correct(&pixel);

                        {
                            let y = height - y - 1;
                            let mut buffer = buffer.write().unwrap();
                            let i = y * width + x;
                            buffer[4 * i] = pixel.x as u8;
                            buffer[4 * i + 1] = pixel.y as u8;
                            buffer[4 * i + 2] = pixel.z as u8;
                            buffer[4 * i + 3] = 255;
                        }

                        lock_and_use(&pixels_rendered, |mut pixels_rendered| {
                            *pixels_rendered += 1;
                            print!(
                                "\rRendering at {samples_per_pixel} spp ({:.1}%)",
                                *pixels_rendered as f64 / (w * h) * 100.
                            );
                        });
                    }
                }
            });
        }
    });
    print!("\n");
    let duration = Instant::now() - start;
    println!("Rendered in {:.1} seconds.", duration.as_secs_f64());
}

fn dump_to_image(config: &Config, buffer: Arc<RwLock<Vec<u8>>>) {
    let Config {
        width,
        height,
        image_path,
        ..
    } = config;

    let f = file_open_create(image_path.as_ref().unwrap().as_str());
    let encoder = PngEncoder::new(f);
    let buffer = buffer.read().unwrap();

    encoder
        .write_image(
            buffer.as_slice(),
            *width as u32,
            *height as u32,
            ColorType::Rgba8,
        )
        .expect("could not write image");
}

fn file_open_create(path: &str) -> BufWriter<File> {
    match File::create(path) {
        Ok(f) => BufWriter::new(f),
        Err(err) => {
            eprintln!("Could not open {path}: {err}");
            exit(1);
        }
    }
}

fn file_open_read(path: &str) -> BufReader<File> {
    match File::open(path) {
        Ok(f) => BufReader::new(f),
        Err(err) => {
            eprintln!("Could not open {path}: {err}");
            exit(1);
        }
    }
}

fn render_to_window(config: &Config, scene: &Scene, buffer: Arc<RwLock<Vec<u8>>>) {
    let Config {
        width,
        height,
        samples_per_pixel,
        ..
    } = *config;
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(width as u32, height as u32);
        WindowBuilder::new()
            .with_title("Raytracer")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let window_size = window.inner_size();
    let mut pixels = {
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(width as u32, height as u32, surface_texture)
            .expect("could not create pixel buffer")
    };

    thread::scope(|s| {
        let buf_ref = Arc::clone(&buffer);
        s.spawn(move || {
            render(scene, width, height, samples_per_pixel, buf_ref);
        });

        event_loop.run(move |event, _, control_flow| {
            if let Event::RedrawRequested(_) = event {
                let f = pixels.frame_mut();
                f.copy_from_slice(&*buffer.read().unwrap());
                if let Err(err) = pixels.render() {
                    eprintln!("{:?}", err);
                    *control_flow = ControlFlow::Exit;
                }
            }

            if input.update(&event) {
                if input.close_requested() {
                    *control_flow = ControlFlow::Exit;
                }
                window.request_redraw();
            }
        });
    });
}

#[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    Image,
    Window,
}

#[derive(Deserialize)]
struct Config {
    width: usize,
    height: usize,
    samples_per_pixel: usize,
    scene: String,
    show_window: bool,
    image_path: Option<String>,
}

enum ConfigLoadError {
    Io(io::Error),
    Toml(toml::de::Error),
}

impl Config {
    pub fn default() -> Self {
        Self {
            width: 480,
            height: 360,
            samples_per_pixel: 4,
            scene: String::new(),
            show_window: false,
            image_path: None,
        }
    }

    pub fn overwrite_from_args(&mut self, args: &Vec<String>) {
        for i in (1..args.len()).step_by(2) {
            match args[i].as_str() {
                "--spp" => {
                    let spp = args[i + 1].parse::<usize>().unwrap();
                    if spp >= 4 {
                        self.samples_per_pixel = spp;
                    }
                }
                "--scene" => {
                    self.scene = args[i + 1].clone();
                }
                "-w" | "--width" => {
                    self.width = args[i + 1].parse::<usize>().unwrap();
                }
                "-h" | "--height" => {
                    self.height = args[i + 1].parse::<usize>().unwrap();
                }
                "--window" => {
                    self.show_window = true;
                }
                "--image" => {
                    self.image_path = Some(args[i + 1].to_string());
                }
                _ => {}
            }
        }
    }

    pub fn from_toml<R: Read>(r: &mut R) -> Result<Self, ConfigLoadError> {
        let mut document = String::new();
        r.read_to_string(&mut document)
            .map_err(ConfigLoadError::Io)?;
        toml::from_str(&document).map_err(ConfigLoadError::Toml)
    }
}
