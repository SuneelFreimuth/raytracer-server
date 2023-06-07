use std::f64::consts::{FRAC_1_PI, PI};
use std::rc::Rc;
use std::sync::{Mutex, Arc};
use std::time::Instant;
use std::{fs::File, io};
use std::thread::{spawn, JoinHandle};

mod ppm;
mod util;
mod vec3;

use rand::random;
use rayon::prelude::IntoParallelIterator;
use vec3::*;

use crate::util::map;

const WIDTH: usize = 480;
const HEIGHT: usize = 360;
const MAXVAL: u64 = 255;
const NUM_SAMPLES: usize = 64;
const NUM_THREADS: usize = 4;

#[derive(Copy, Clone, Debug)]
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

#[derive(Copy, Clone, Debug)]
enum Body {
    Sphere(Sphere),
}

#[derive(Copy, Clone, Debug)]
struct Sphere {
    pos: Vec3,
    r: f64,
}

#[derive(Debug, Clone, Copy)]
struct Hit {
    pub t: f64,
    pub pos: Vec3,
    pub n: Vec3,
    pub id: usize,
}

impl Body {
    pub fn intersect(&self, ray: &Ray) -> Option<Hit> {
        match self {
            Self::Sphere(sphere) => {
                let op = sphere.pos - ray.pos;
                let eps = 1e-4;
                let b = op.dot(&ray.dir);

                let mut det = b * b - op.dot(&op) + sphere.r * sphere.r;
                if det < 0. {
                    return None;
                }
                det = det.sqrt();

                let t = b - det;
                if t > eps {
                    let pos = ray.eval(t);
                    let n = (pos - sphere.pos).norm();
                    return Some(Hit {
                        t,
                        pos,
                        n: if n.dot(&-ray.dir) >= 0. { n } else { -n },
                        id: 1000000 // One morbillion
                    });
                }

                let t = b + det;
                if t > eps {
                    let pos = ray.eval(t);
                    let n = (pos - sphere.pos).norm();
                    return Some(Hit {
                        t,
                        pos,
                        n: if n.dot(&-ray.dir) >= 0. { n } else { -n },
                        id: 1000000 // One morbillion
                    });
                }

                None
            }
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
        }
    }
}

pub struct Scene {
    objects: Vec<Object>,
}

fn gamma_correct(v: &Vec3) -> Vec3 {
    v.clamp(0., 1.).powf(1.0 / 2.2) * 255.0 + Vec3::repeat(0.5)
}

const MAX_BOUNCES: u64 = 5;
const SURVIVAL_PROBABILITY: f64 = 0.9;

impl Scene {
    pub fn render(&self, img: &mut ppm::Image, camera: &Ray) {
        let width = img.width;
        let height = img.height;
        let w = width as f64;
        let h = height as f64;
        let cx = Vec3::new(w * 0.5135 / h, 0., 0.);
        let cy = cx.cross(&camera.dir).norm() * 0.5135;
        let num_samples = NUM_SAMPLES / 4;
        
        let mut pixels = vec![Vec3::zero(); img.width * img.height];
        (0..height)
            .into_par_iter()
            .map(|y| {
                for x in 0..width {
                    let i = (height - y - 1) * width + x;

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
                                    + camera.dir;

                                rad = rad
                                    + self.received_radiance(&Ray::new(camera.pos, d.norm()))
                                        * (1. / num_samples as f64);
                            }
                            pixels[i] = pixels[i] + rad.clamp(0., 1.) * 0.25;
                        }
                    }
                }
                print!(
                    "\rRendering at {} spp ({:.1}%)",
                    num_samples * 4,
                    100. * y as f64 / h
                );
            });
        print!("\n");

        for y in 0..img.height {
            for x in 0..img.width {
                let i = y * img.width + x;
                img.set(y, x, gamma_correct(&pixels[i]));
            }
        }
    }

    fn received_radiance(&self, r: &Ray) -> Vec3 {
        if let Some(hit) = self.trace_ray(r) {
            let obj = &self.objects[hit.id as usize];
            obj.emitted + self.reflected_radiance(&hit, &-r.dir, 1)
        } else {
            Vec3::zero()
        }
    }

    fn reflected_radiance(&self, hit: &Hit, o: &Vec3, depth: u64) -> Vec3 {
        let p = if depth <= MAX_BOUNCES {
            1.0
        } else {
            SURVIVAL_PROBABILITY
        };
        
        let obj = &self.objects[hit.id];
        let Hit { pos: x, n, .. } = hit;

        if random::<f64>() < p {
            let (i, pdf) = obj.brdf.sample(n, o);
            if let Some(next_hit) = self.trace_ray(&Ray::new(*x, i)) {
                return obj.emitted +
                    self.reflected_radiance(&next_hit, &-i, depth + 1)
                        .mult(&obj.brdf.eval(n, o, &i)) * PI / p;
            }
        }

        Vec3::zero()
    }

    fn trace_ray(&self, ray: &Ray) -> Option<Hit> {
        let mut nearest_hit: Option<Hit> = None;
        for (i, obj) in self.objects.iter().enumerate() {
            if let Some(mut hit) = obj.body.intersect(ray) {
                hit.id = i;
                match nearest_hit {
                    Some(nh) if hit.t < nh.t => {
                        nearest_hit = Some(hit);
                    },
                    None => {
                        nearest_hit = Some(hit);
                    },
                    _ => {}
                }
            }
        }
        nearest_hit
    }
}

fn main() -> io::Result<()> {
    let left_wall = BRDF::Diffuse(Vec3::new(0.75, 0.25, 0.25));
    let right_wall = BRDF::Diffuse(Vec3::new(0.25, 0.25, 0.75));
    let other_wall = BRDF::Diffuse(Vec3::new(0.75, 0.75, 0.75));
    let black_surf = BRDF::Diffuse(Vec3::new(0.0, 0.0, 0.0));
    let bright_surf = BRDF::Diffuse(Vec3::new(0.9, 0.9, 0.9));

    let scene = Scene {
        objects: vec![
            // Left
            Object::new_sphere(
                1e5,
                Vec3::new(1e5 + 1., 40.8, 81.6),
                Vec3::zero(),
                left_wall,
            ),
            // Right
            Object::new_sphere(
                1e5,
                Vec3::new(-1.0e5 + 99., 40.8, 81.6),
                Vec3::zero(),
                right_wall,
            ),
            // Back
            Object::new_sphere(1e5, Vec3::new(50., 40.8, 1e5), Vec3::zero(), other_wall),
            // Bottom
            Object::new_sphere(1e5, Vec3::new(50., 1e5, 81.6), Vec3::zero(), other_wall),
            // Top
            Object::new_sphere(
                1e5,
                Vec3::new(50., -1e5 + 81.6, 81.6),
                Vec3::zero(),
                other_wall,
            ),
            // Ball 1
            Object::new_sphere(16.5, Vec3::new(27., 16.5, 47.), Vec3::zero(), bright_surf),
            // Ball 2
            Object::new_sphere(16.5, Vec3::new(73., 16.5, 78.), Vec3::zero(), bright_surf),
            // Light
            Object::new_sphere(
                5.0,
                Vec3::new(50., 70.0, 81.6),
                Vec3::new(50., 50., 50.),
                black_surf,
            ),
        ],
    };

    let mut img = ppm::Image::new(WIDTH, HEIGHT, MAXVAL);
    let cam = Ray {
        pos: Vec3::new(50., 52., 295.6),
        dir: Vec3::new(0., -0.042612, -1.).norm(),
    };

    let now = Instant::now();
    scene.render(&mut img, &cam);
    let elapsed = now.elapsed();
    println!("Rendered in {:.1} seconds.", elapsed.as_secs_f64());

    let mut f = File::create("image.ppm")?;
    // let mut img = ppm::Image::new(WIDTH, HEIGHT, MAXVAL);
    // for y in 0..img.height {
    //     for x in 0..img.width {
    //         img.set(y, x, Vec3::new(
    //             x as f64 / img.width as f64 * img.maxval,
    //             y as f64 / img.height as f64 * img.maxval,
    //             0.,
    //         ));
    //     }
    // }
    img.dump(&mut f)?;

    Ok(())
}
