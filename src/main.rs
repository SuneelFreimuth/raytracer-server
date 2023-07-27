#[allow(dead_code)]
use std::env::args;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read};
use std::process::exit;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Instant;

use image::codecs::png::PngEncoder;
use image::ColorType;
use image::ImageEncoder;
use pixels::{Pixels, SurfaceTexture};
use rand::random;
use serde::Deserialize;
use winit::{
    dpi::LogicalSize,
    event::Event,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;

mod geometry;
mod scene;

use geometry::{Vec3, Ray};

use geometry::MeshLoadError;
use scene::Scene;

const CONFIG_FILE: &str = "config.toml";

fn main() {
    let mut config_file = file_open_read(CONFIG_FILE);
    let mut config = match Config::from_toml(&mut config_file) {
        Ok(c) => c,
        Err(err) => {
            match err {
                ConfigLoadError::Io(err) => {
                    eprintln!("I/O error: {err}")
                }
                ConfigLoadError::Toml(err) => {
                    eprintln!("Failed to parse {CONFIG_FILE}: {err}")
                }
            }
            return;
        }
    };
    config.overwrite_from_args(&args().collect());

    let f = File::open(&config.scene).expect("could not open file");
    let mut scene = match Scene::from_toml(&mut BufReader::new(f)) {
        Ok(scene) => scene,
        Err(err) => {
            match err {
                scene::LoadTomlError::Io(err)
                | scene::LoadTomlError::MeshLoad(MeshLoadError::IO(err)) => {
                    eprintln!("I/O error: {err}")
                }
                scene::LoadTomlError::MeshLoad(MeshLoadError::Parse(err)) => {
                    eprintln!("Failed to load mesh {err}")
                }
                scene::LoadTomlError::Parse(err) => eprintln!("Could not parse TOML: {err}"),
            }
            return;
        }
    };
    scene.name = config.image_path.clone().unwrap_or(config.scene.clone());

    let buffer = Arc::new(RwLock::new(vec![0; 4 * config.width * config.height]));

    if config.show_window {
        render_to_window(&config, &scene, Arc::clone(&buffer));
    } else {
        render(
            &scene,
            config.width,
            config.height,
            config.samples_per_pixel,
            Arc::clone(&buffer),
        );
    }

    if config.image_path.is_some() {
        dump_to_image(&config, buffer);
    }
}

fn gamma_correct(v: &Vec3) -> Vec3 {
    v.clamp(0., 1.).powf(1.0 / 2.2) * 255.0 + Vec3::repeat(0.5)
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
