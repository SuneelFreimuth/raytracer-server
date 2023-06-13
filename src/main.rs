#[allow(dead_code)]
use std::f64::consts::{FRAC_1_PI, PI};
use std::io::{stdout, BufRead, BufReader, BufWriter, Read, Write};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, spawn};
use std::time::Instant;
use std::{fs::File, io};

mod body;
mod ppm;
mod util;
mod vec3;

use pixels::{Error, Pixels, SurfaceTexture};
use rand::random;
use rand::seq::index::sample;
use rayon::prelude::*;
use winit::{
    dpi::LogicalSize,
    event::Event,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;

use body::{Body, Hit, Mesh, Plane, Sphere};
use util::map;
use vec3::*;

use crate::body::Node;
use crate::ppm::Image;

const WIDTH: usize = 480;
const HEIGHT: usize = 360;
const MAXVAL: u64 = 255;
static mut NUM_SAMPLES: usize = 4;

#[derive(Clone, Debug)]
struct Object {
    emitted: Vec3,
    brdf: BRDF,
    body: Body,
}

impl Object {
    pub fn new_sphere(r: f64, pos: Vec3, emitted: Vec3, brdf: BRDF) -> Self {
        Self {
            emitted,
            brdf,
            body: Body::Sphere(Sphere { pos, r }),
        }
    }
}

pub fn create_local_coord(n: &Vec3) -> (Vec3, Vec3, Vec3) {
    let w = n.clone();
    let u = (if w.x.abs() > 0.1 {
        Vec3::new(0., 1., 0.)
    } else {
        Vec3::new(1., 0., 0.)
    })
    .cross(&w)
    .norm();
    let v = w.cross(&u);
    (u, v, w)
}

#[derive(Debug, Copy, Clone)]
pub enum BRDF {
    Diffuse(Vec3),
    Specular(Vec3),
    Phong {
        kd: f64,
        ks: f64,
        power: i32,
        color_d: Vec3,
        color_s: Vec3
    },
}

impl BRDF {
    pub fn eval(&self, n: &Vec3, o: &Vec3, i: &Vec3) -> Vec3 {
        match self {
            Self::Diffuse(kd) => kd * FRAC_1_PI,
            Self::Specular(ks) => {
                if i.equal_within(&o.flip_across(&n), 0.001) {
                    ks / n.dot(i)
                } else {
                    Vec3::zero()
                }
            }
            Self::Phong { ks, kd, power, color_d, color_s } => {
                let reflection = i.flip_across(n);
                color_d * *kd * FRAC_1_PI
                    + color_s * *ks * (power + 2) as f64 / (2. * PI) * o.dot(&reflection).max(0.).powi(*power)
            }
        }
    }

    // Returns:
    // - sample proportional to the BRDF
    // - the BRDF's pdf
    pub fn sample(&self, n: &Vec3, o: &Vec3) -> (Vec3, f64) {
        match self {
            Self::Diffuse(_) => {
                let z = random::<f64>().sqrt();
                let r = (1.0 - z * z).sqrt();
                let phi = 2.0 * PI * random::<f64>();
                let x = r * phi.cos();
                let y = r * phi.sin();
                let (u, v, w) = create_local_coord(n);
                let i = (u * x + v * y + w * z).norm();
                (i, n.dot(&i) * FRAC_1_PI)
            }
            Self::Specular(_) => (o.flip_across(n), 1.0),
            Self::Phong { kd, power, .. } => {
                let p = *power as f64;
                let u = random::<f64>();
                if u < *kd {
                    // Diffuse sample
                    let xi1 = random::<f64>();
                    let xi2 = random::<f64>();
                    let i = Vec3 {
                        x: (1. - xi1).sqrt() * (2. * PI * xi2).cos(),
                        y: (1. - xi1).sqrt() * (2. * PI * xi2).sin(),
                        z: xi1.sqrt()
                    };
                    (i, i.z * FRAC_1_PI)
                } else {
                    // Specular sample
                    let xi1 = random::<f64>();
                    let xi2 = random::<f64>();
                    let i = Vec3 {
                        x: (1. - xi1.powf(2. / (p + 1.))).sqrt() * (2. * PI * xi2).cos(),
                        y: (1. - xi1.powf(2. / (p + 1.))).sqrt() * (2. * PI * xi2).sin(),
                        z: xi1.powf(1. / (p + 1.)),
                    };
                    (
                        i,
                        (p + 1.) / (2. * PI) * i.z.powi(*power)
                    )
                }
            }
        }
    }
}

pub struct Scene {
    camera: Ray,
    objects: Vec<Object>,
}

fn gamma_correct(v: &Vec3) -> Vec3 {
    v.clamp(0., 1.).powf(1.0 / 2.2) * 255.0 + Vec3::repeat(0.5)
}

const MAX_BOUNCES: u64 = 5;
const SURVIVAL_PROBABILITY: f64 = 0.9;

fn concat<T: Clone>(a: Vec<T>, b: Vec<T>) -> Vec<T> {
    let mut c = a.clone();
    for el in b {
        c.push(el);
    }
    c
}

impl Scene {
    fn received_radiance(&self, r: &Ray) -> Vec3 {
        if let Some(hit) = self.trace_ray(r) {
            let obj = &self.objects[hit.id as usize];
            obj.emitted + self.reflected_radiance(&hit, &-r.dir, 1)
        } else {
            Vec3::zero()
        }
    }

    fn reflected_radiance(&self, hit: &Hit, o: &Vec3, depth: u64) -> Vec3 {
        let Hit { pos: x, n, .. } = hit;
        let obj = &self.objects[hit.id];
        let p = if depth <= MAX_BOUNCES {
            1.0
        } else {
            SURVIVAL_PROBABILITY
        };

        if let BRDF::Specular(_) = obj.brdf {
            let mut rad = Vec3::zero();

            if random::<f64>() < p {
                let (i, pdf) = obj.brdf.sample(n, o);
                if let Some(hit) = self.trace_ray(&Ray::new(*x, i)) {
                    rad = self.objects[hit.id].emitted
                        + self
                            .reflected_radiance(&hit, o, depth + 1)
                            .mult(&obj.brdf.eval(n, o, &i))
                            * n.dot(&i)
                            / (pdf * p);
                }
            }

            rad
        } else {
            let (y, ny, pdf) = self.sample_light_source();
            let i = (y - x).norm();
            let r_sqr = (y - x).dot(&(y - x));
            let visibility = self.mutually_visible(&x, &y);
            let mut rad = self.objects[7].emitted.mult(&obj.brdf.eval(n, o, &i))
                * visibility
                * n.dot(&i)
                * ny.dot(&-i)
                / (r_sqr * pdf);

            if random::<f64>() < p {
                let (i, pdf) = obj.brdf.sample(n, o);
                if let Some(hit) = self.trace_ray(&Ray::new(*x, i)) {
                    rad += self
                        .reflected_radiance(&hit, &-i, depth + 1)
                        .mult(&obj.brdf.eval(n, o, &i))
                        * n.dot(&i)
                        / (pdf * p);
                }
            }

            rad
        }
    }

    fn sample_light_source(&self) -> (Vec3, Vec3, f64) {
        match self.objects[7].body {
            Body::Sphere(Sphere { pos: center, r }) => {
                let xi1 = random::<f64>();
                let xi2 = random::<f64>();

                let z = 2. * xi1 - 1.;
                let x = (1.0 - z * z).sqrt() * (2. * PI * xi2).cos();
                let y = (1.0 - z * z).sqrt() * (2. * PI * xi2).sin();

                let n = Vec3::new(x, y, z).norm();
                let sample = center + n * r;
                let pdf = 1.0 / (4.0 * PI * r * r);
                (sample, n, pdf)
            }
            _ => unimplemented!(),
        }
    }

    fn mutually_visible(&self, x: &Vec3, y: &Vec3) -> f64 {
        const ERR_MARGIN: f64 = 0.001;
        let diff = y - x;
        let x_to_y = Ray {
            pos: *x,
            dir: diff.norm(),
        };
        match self.trace_ray(&x_to_y) {
            Some(hit) => {
                if hit.t + ERR_MARGIN >= diff.mag() {
                    1.0
                } else {
                    0.0
                }
            }
            None => 1.0,
        }
    }

    fn trace_ray(&self, ray: &Ray) -> Option<Hit> {
        let mut nearest_hit: Option<Hit> = None;
        for (i, obj) in self.objects.iter().enumerate() {
            if let Some(mut hit) = obj.body.intersect(ray) {
                hit.id = i;
                match nearest_hit {
                    Some(nh) if hit.t < nh.t => {
                        nearest_hit = Some(hit);
                    }
                    None => {
                        nearest_hit = Some(hit);
                    }
                    _ => {}
                }
            }
        }
        nearest_hit
    }
}

trait Renderer {
    fn render(&self, scene: &Scene);
}

struct PPMRenderer {
    width: usize,
    height: usize,
    samples_per_pixel: usize,
    file: File,
}

impl Renderer for PPMRenderer {
    fn render(&self, scene: &Scene) {
        let PPMRenderer {
            width,
            height,
            samples_per_pixel,
            file,
        } = self;
        let w = *width as f64;
        let h = *height as f64;
        let cx = Vec3::new(w * 0.5135 / h, 0., 0.);
        let cy = cx.cross(&scene.camera.dir).norm() * 0.5135;
        let num_samples = samples_per_pixel / 4;

        let img = ppm::Image::new(self.width, self.height, 255);
        let img_guarded = Arc::new(Mutex::new(img));
        let img_guard_clone = Arc::clone(&img_guarded);
        let completion = Arc::new(Mutex::new(0u64));

        (0..*height).into_par_iter().for_each(move |y| {
            let img_guarded = &img_guard_clone;
            let y = height - y - 1;
            for x in 0..*width {
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
                let mut img = img_guarded.lock().unwrap();
                img.set(height - y - 1, x, gamma_correct(&pixel));
            }

            let mut completion = completion.lock().unwrap();
            *completion += 1;
            print!(
                "\rRendering at {samples_per_pixel} spp ({:.1}%)",
                *completion as f64 / h * 100.
            );
        });
        print!("\n");

        let img = img_guarded.lock().unwrap();
        img.dump(&mut BufWriter::new(file))
            .expect("failed to dump to file");

        // for y in 0..img.height {
        //     for x in 0..img.width {
        //         let i = y * img.width + x;
        //         img.set(y, x, gamma_correct(&pixels[i]));
        //     }
        // }
    }
}

struct WindowRenderer {
    width: usize,
    height: usize,
    samples_per_pixel: usize,
}

impl WindowRenderer {
    fn render_row_sync(&self, scene: &Scene, buf: &RwLock<Vec<u8>>, y: usize) {
        let WindowRenderer {
            width,
            height,
            samples_per_pixel,
        } = self;
        let w = *width as f64;
        let h = *height as f64;
        let cx = Vec3::new(w * 0.5135 / h, 0., 0.);
        let cy = cx.cross(&scene.camera.dir).norm() * 0.5135;
        let num_samples = samples_per_pixel / 4;

        for x in 0..*width {
            let mut pixel = Vec3::zero();
            for sy in 0..2 {
                for sx in 0..2 {
                    let y = HEIGHT - y - 1;
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
            let mut buf = buf.write().unwrap();
            let row = &mut buf[4 * WIDTH * y..4 * WIDTH * (y + 1)];
            pixel = gamma_correct(&pixel);
            row[4 * x] = pixel.x as u8;
            row[4 * x + 1] = pixel.y as u8;
            row[4 * x + 2] = pixel.z as u8;
            row[4 * x + 3] = 255;
        }
    }
}

impl Renderer for WindowRenderer {
    fn render(&self, scene: &Scene) {
        let event_loop = EventLoop::new();
        let mut input = WinitInputHelper::new();
        let window = {
            let size = LogicalSize::new(480., 360.);
            WindowBuilder::new()
                .with_title("Raytracer")
                .with_inner_size(size)
                .with_min_inner_size(size)
                .build(&event_loop)
                .unwrap()
        };

        let window_size = window.inner_size();
        let mut pixels = {
            let surface_texture =
                SurfaceTexture::new(window_size.width, window_size.height, &window);
            Pixels::new(self.width as u32, self.height as u32, surface_texture)
                .expect("could not create pixel buffer")
        };

        let buf = Arc::new(RwLock::new(vec![0u8; 4 * self.width * self.height]));
        let buf_ref = Arc::clone(&buf);
        thread::scope(|s| {
            s.spawn(move || {
                let now = Instant::now();
                (0..self.height).into_par_iter().for_each(|y| {
                    self.render_row_sync(&scene, &buf_ref, y);
                });
                let elapsed = now.elapsed();
                println!("Rendered in {:.1} seconds.", elapsed.as_secs_f64());
            });

            event_loop.run(move |event, _, control_flow| {
                if let Event::RedrawRequested(_) = event {
                    let f = pixels.frame_mut();
                    f.copy_from_slice(&*buf.read().unwrap());
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
}

fn main() -> io::Result<()> {
    let left_wall = BRDF::Diffuse(Vec3::new(0.75, 0.25, 0.25));
    let right_wall = BRDF::Diffuse(Vec3::new(0.25, 0.25, 0.75));
    let other_wall = BRDF::Diffuse(Vec3::new(0.75, 0.75, 0.75));
    let black_surf = BRDF::Diffuse(Vec3::repeat(0.0));
    let bright_surf = BRDF::Diffuse(Vec3::repeat(0.9));
    let shiny_surf = BRDF::Specular(Vec3::repeat(0.999));

    let f = BufReader::new(File::open("assets/crewmate.obj")?);
    let mut mesh = Mesh::load(f).expect("could not open model");
    mesh.scale(0.3);
    mesh.translate(&Vec3::new(30., -70., 75.));
    mesh.rotate_y(0.5);
    mesh.accelerate();

    // let f = BufReader::new(File::open("assets/chair.obj")?);
    // let mut mesh = Mesh::load(f).expect("could not open model");
    // mesh.scale(25.);
    // mesh.translate(&Vec3::new(30., 20.01, 65.));
    // mesh.rotate_y(0.5);
    // mesh.accelerate();

    let scene = Scene {
        camera: Ray {
            pos: Vec3::new(50., 52., 295.6),
            dir: Vec3::new(0., -0.042612, -1.).norm(),
        },
        objects: vec![
            // Left
            Object {
                brdf: left_wall,
                emitted: Vec3::zero(),
                body: Body::Plane(Plane {
                    pos: Vec3::new(1., 0., 0.),
                    n: Vec3::new(1., 0., 0.),
                }),
            },
            // Right
            Object {
                brdf: right_wall,
                emitted: Vec3::zero(),
                body: Body::Plane(Plane {
                    pos: Vec3::new(99., 0., 0.),
                    n: Vec3::new(-1., 0., 0.),
                }),
            },
            // Back
            Object {
                emitted: Vec3::zero(),
                brdf: other_wall,
                body: Body::Plane(Plane {
                    pos: Vec3::new(0., 0., 0.),
                    n: Vec3::new(0., 0., -1.),
                }),
            },
            // Bottom
            Object {
                brdf: other_wall,
                emitted: Vec3::zero(),
                body: Body::Plane(Plane {
                    pos: Vec3::new(0., 0., 0.),
                    n: Vec3::new(0., 1., 0.),
                }),
            },
            // Top
            Object {
                brdf: other_wall,
                emitted: Vec3::zero(),
                body: Body::Plane(Plane {
                    pos: Vec3::new(0., 81.6, 0.),
                    n: Vec3::new(0., -1., 0.),
                }),
            },
            // Ball 1
            Object {
                // brdf: bright_surf,
                brdf: BRDF::Phong {
                    kd: 0.4,
                    ks: 0.05,
                    power: 50,
                    color_d: Vec3::new(1.0, 0.0, 0.0),
                    color_s: Vec3::repeat(1.0)
                },
                emitted: Vec3::zero(),
                body: Body::Mesh(mesh),
                // body: Body::Sphere(Sphere {
                //     pos: Vec3::new(27., 16.5, 47.),
                //     r: 16.5,
                // }),
            },
            // Ball 2
            Object {
                brdf: shiny_surf,
                // brdf: BRDF::Phong {
                //     kd: 0.2,
                //     ks: 0.4,
                //     power: 50,
                //     color_d: Vec3::repeat(0.7),
                //     color_s: Vec3::repeat(1.0)
                // },
                emitted: Vec3::zero(),
                body: Body::Sphere(Sphere {
                    pos: Vec3::new(73., 16.5, 68.),
                    r: 16.5,
                }),
            },
            // Light
            Object {
                brdf: black_surf,
                emitted: Vec3::repeat(50.),
                body: Body::Sphere(Sphere {
                    pos: Vec3::new(50., 70., 81.6),
                    r: 5.,
                }),
            },
            Object {
                brdf: bright_surf,
                emitted: Vec3::zero(),
                body: Body::Plane(Plane {
                    pos: Vec3::new(0., 0., 320.),
                    n: Vec3::new(0., 0., -1.)
                })
            }
        ],
    };

    let args: Vec<String> = std::env::args().collect();
    // let renderer = PPMRenderer {
    //     width: WIDTH,
    //     height: HEIGHT,
    //     samples_per_pixel: if let Some(arg) = args.get(1) {
    //         arg.parse().unwrap()
    //     } else {
    //         4
    //     },
    //     file: File::create("image.ppm").expect("could not open image file for writing")
    // };
    let renderer = WindowRenderer {
        width: WIDTH,
        height: HEIGHT,
        samples_per_pixel: if let Some(arg) = args.get(1) {
            arg.parse().unwrap()
        } else {
            4
        },
    };
    renderer.render(&scene);

    Ok(())
}
