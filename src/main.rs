use std::f64::consts::{FRAC_1_PI, PI};
use std::io::{BufReader, BufWriter, Write, stdout};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::thread::{spawn, JoinHandle};
use std::time::Instant;
use std::{fs::File, io};

mod ppm;
mod util;
mod vec3;
mod octree;

use obj::{load_obj, Obj};
use rand::random;
use rand::seq::index::sample;
use rayon::prelude::*;

use vec3::*;

use util::map;
use octree::BoundingBox;

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

#[derive(Clone, Debug)]
enum Body {
    Sphere(Sphere),
    Plane(Plane),
    Mesh(Mesh),
}

#[derive(Copy, Clone, Debug)]
struct Sphere {
    pos: Vec3,
    r: f64,
}

#[derive(Copy, Clone, Debug)]
struct Plane {
    pos: Vec3,
    n: Vec3,
}

#[derive(Clone, Debug)]
struct Mesh {
    vertices: Vec<Vec3>,
    normals: Vec<Vec3>,
    indices: Vec<usize>,
}

struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3
}

impl Triangle {
    pub fn normal(&self) -> Vec3 {
        (self.v2 - self.v0).cross(&(self.v1 - self.v0)).norm()
    }

    pub fn intersect(&self, ray: &Ray) -> Option<Hit> {
        // Moller-Trumbore ray intersection algorithm.
        let n = self.normal();
        if n.dot(r.dir) < 0.0001 {
            return None
        }

        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        let b = ray.pos - self.v0;

        let det = determinant3(&-ray.dir, &e1, &e2);

        let t = determinant3(b, &e1, &e2) / det;
        let u = determinant3(b, &e1, &e2) / det;
        let v = determinant3(b, &e1, &e2) / det;

        if ((u + v) - 

        None
    }
}

struct TriangleIterator<'a> {
    mesh: &'a Mesh,
    i: usize
}

impl<'a> Iterator for TriangleIterator<'a> {
    type Item = Triangle;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.mesh.num_triangles() {
            let t = self.mesh.get_triangle(self.i);
            self.i += 1;
            Some(t)
        } else {
            None
        }
    }
}

impl Mesh {
    pub fn num_triangles(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn get_triangle(&self, i: usize) -> Triangle {
        let v0 = self.vertices[self.indices[i / 3]];
        let v1 = self.vertices[self.indices[i / 3 + 1]];
        let v2 = self.vertices[self.indices[i / 3 + 2]];
        Triangle { v0, v1, v2 }
    }

    pub fn triangles(&self) -> TriangleIterator {
        TriangleIterator {
            mesh: self,
            i: 0
        }
    }
}

#[derive(Copy, Debug, Clone)]
struct Hit {
    pub t: f64,
    pub pos: Vec3,
    pub n: Vec3,
    pub id: usize,
}

impl Body {
    pub fn new_mesh(path: &str) -> obj::ObjResult<Self> {
        let f = BufReader::new(File::open(path)?);
        let model: Obj<obj::Vertex, usize> = load_obj(f)?;
        Ok(Self::Mesh(Mesh {
            vertices: model.vertices.iter().map(|v| v.position.into()).collect(),
            normals: model.vertices.iter().map(|v| Vec3::from(v.normal).norm()).collect(),
            indices: model.indices,
        }))
    }

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
                        id: 1000000, // One morbillion
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
                        id: 1000000, // One morbillion
                    });
                }

                None
            }
            Self::Plane(Plane { pos, n }) => {
                let d_dot_n = ray.dir.dot(n);
                if d_dot_n.abs() < 0.0001 {
                    return None;
                }
                let t = (pos - &ray.pos).dot(n) / ray.dir.dot(n);
                if t >= 0. {
                    // println!("{:?} hit plane at {:?} with normal {:?}",
                    //     ray, ray.eval(t), n);
                    let n = if n.dot(&-ray.dir) >= 0. { *n } else { -*n };
                    Some(Hit {
                        t,
                        pos: ray.eval(t) + n * 0.00001,
                        n,
                        id: 1000000,
                    })
                } else {
                    None
                }
            }
            Self::Mesh(mesh) => {
                for tri in mesh.triangles() {
                    if let Some(hit) = tri.intersect(ray) {
                        return Some(hit)
                    }
                }
                None
            },
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

    pub fn is_specular(&self) -> bool {
        match self {
            Self::Specular(_) => true,
            _ => false,
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

fn concat<T: Clone>(a: Vec<T>, b: Vec<T>) -> Vec<T> {
    let mut c = a.clone();
    for el in b {
        c.push(el);
    }
    c
}

impl Scene {
    pub fn render(&self, img: &mut ppm::Image, camera: &Ray) {
        let width = img.width as usize;
        let height = img.height as usize;
        let w = width as f64;
        let h = height as f64;
        let cx = Vec3::new(w * 0.5135 / h, 0., 0.);
        let cy = cx.cross(&camera.dir).norm() * 0.5135;
        let num_samples = unsafe { NUM_SAMPLES / 4 };

        let pixels = (0..height)
            .into_par_iter()
            .map(move |y| {
                let y = height - y - 1;
                let mut row = vec![Vec3::zero(); width];
                for x in 0..width {
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

                                rad += self.received_radiance(&Ray::new(camera.pos, d.norm()))
                                    * (1. / num_samples as f64);
                            }
                            row[x] += rad.clamp(0., 1.) * 0.25;
                        }
                    }
                }
                row
            })
            .reduce(|| Vec::new(), |agg, x| concat(agg, x));
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
        let Hit { pos: x, n, .. } = hit;
        let obj = &self.objects[hit.id];
        let p = if depth <= MAX_BOUNCES {
            1.0
        } else {
            SURVIVAL_PROBABILITY
        };

        if obj.brdf.is_specular() {
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

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        unsafe {
            NUM_SAMPLES = args[1]
                .parse::<usize>()
                .expect("if provided, first argument must be an integer.");
        }
    }

    let left_wall = BRDF::Diffuse(Vec3::new(0.75, 0.25, 0.25));
    let right_wall = BRDF::Diffuse(Vec3::new(0.25, 0.25, 0.75));
    let other_wall = BRDF::Diffuse(Vec3::new(0.75, 0.75, 0.75));
    let black_surf = BRDF::Diffuse(Vec3::repeat(0.0));
    let bright_surf = BRDF::Diffuse(Vec3::repeat(0.9));
    let shiny_surf = BRDF::Specular(Vec3::repeat(0.999));

    let mesh = Object {
        brdf: bright_surf,
        emitted: Vec3::zero(),
        body: Body::new_mesh("assets/chair2.obj").expect("could not open chair"),
        // body: Body::Sphere(Sphere {
        //     pos: Vec3::new(27., 16.5, 47.),
        //     r: 16.5,
        // }),
    };

    if let Body::Mesh(ref mesh) = mesh.body {
        println!("Mesh has {} vertices, {} normals, and {} indices.", mesh.vertices.len(), mesh.normals.len(), mesh.indices.len());
        println!("mesh bounding box: {:?}", BoundingBox::enclose(&mesh.vertices));
    }

    let scene = Scene {
        objects: vec![
            // Left
            Object {
                brdf: left_wall,
                emitted: Vec3::zero(),
                body: Body::Plane(Plane {
                    pos: Vec3::new(1., 40.8, 81.6),
                    n: Vec3::new(1., 0., 0.),
                }),
                // body: Body::Sphere(Sphere {
                //     pos: Vec3::new(1e5 + 1., 40.8, 81.6),
                //     r: 1e5,
                // }),
            },
            // Right
            Object {
                brdf: right_wall,
                emitted: Vec3::zero(),
                body: Body::Plane(Plane {
                    pos: Vec3::new(99., 40.8, 81.6),
                    n: Vec3::new(-1., 0., 0.),
                }),
                // body: Body::Sphere(Sphere {
                //     pos: Vec3::new(-1.0e5 + 99., 40.8, 81.6),
                //     r: 1e5,
                // }),
            },
            // Back
            Object {
                emitted: Vec3::zero(),
                brdf: other_wall,
                body: Body::Plane(Plane {
                    pos: Vec3::new(50., 40.8, 0.),
                    n: Vec3::new(0., 0., -1.),
                }),
            },
            // Bottom
            Object {
                brdf: other_wall,
                emitted: Vec3::zero(),
                body: Body::Plane(Plane {
                    pos: Vec3::new(50., 0., 81.6),
                    n: Vec3::new(0., 1., 0.),
                }),
            },
            // Top
            Object {
                brdf: other_wall,
                emitted: Vec3::zero(),
                body: Body::Plane(Plane {
                    pos: Vec3::new(50., 81.6, 81.6),
                    n: Vec3::new(0., -1., 0.),
                }),
            },
            // Ball 1
            mesh,
            // Object {
            //     brdf: bright_surf,
            //     emitted: Vec3::zero(),
            //     body: Body::Sphere(Sphere {
            //         pos: Vec3::new(27., 16.5, 47.),
            //         r: 16.5,
            //     }),
            // },
            // Ball 2
            Object {
                brdf: shiny_surf,
                emitted: Vec3::zero(),
                body: Body::Sphere(Sphere {
                    pos: Vec3::new(73., 16.5, 78.),
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
            }
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

    let f = File::create("image.ppm")?;
    let mut w = BufWriter::new(f);
    let now = Instant::now();
    img.dump(&mut w)?;
    let elapsed = now.elapsed();
    println!("Dumped to file in {:.5} seconds.", elapsed.as_secs_f64());

    Ok(())
}
