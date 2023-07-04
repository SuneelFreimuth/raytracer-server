use std::f64::consts::{FRAC_1_PI, PI};
use std::fs::File;
use std::io::{self, Read, BufReader};

use rand::random;
use serde::Deserialize;

use crate::geometry::{Geometry, Sphere, Plane, Mesh, Hit, MeshLoadError};
use crate::vec3::{Vec3, Ray};
use crate::config::USE_MIS;

#[derive(Clone, Debug)]
pub struct Object {
    emitted: Vec3,
    brdf: BRDF,
    geometry: Geometry,
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
        color_s: Vec3,
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
            Self::Phong {
                ks,
                kd,
                power,
                color_d,
                color_s,
            } => {
                let reflection = i.flip_across(n);
                color_d * *kd * FRAC_1_PI
                    + color_s * *ks * (power + 2) as f64 / (2. * PI)
                        * o.dot(&reflection).max(0.).powi(*power)
            }
        }
    }

    pub fn sample_incoming(&self, n: &Vec3, o: &Vec3) -> (Vec3, f64) {
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
            Self::Phong { kd, ks, power, .. } => {
                let p = *power as f64;
                let u = random::<f64>();
                if u < *kd {
                    // Diffuse sample
                    let xi1 = random::<f64>();
                    let xi2 = random::<f64>();
                    let i = Vec3 {
                        x: (1. - xi1).sqrt() * (2. * PI * xi2).cos(),
                        y: (1. - xi1).sqrt() * (2. * PI * xi2).sin(),
                        z: xi1.sqrt(),
                    };
                    (i, n.dot(&i) * FRAC_1_PI)
                } else if *kd <= u && u < kd + ks {
                    // Specular sample
                    let xi1 = random::<f64>();
                    let xi2 = random::<f64>();
                    let i = Vec3 {
                        x: (1. - xi1.powf(2. / (p + 1.))).sqrt() * (2. * PI * xi2).cos(),
                        y: (1. - xi1.powf(2. / (p + 1.))).sqrt() * (2. * PI * xi2).sin(),
                        z: xi1.powf(1. / (p + 1.)),
                    };
                    (i, (p + 1.) / (2. * PI) * i.z.powi(*power))
                } else {
                    // No contribution
                    (Vec3::zero(), 1.0)
                }
            }
        }
    }
}

pub struct Scene {
    pub camera: Ray,
    pub objects: Vec<Object>,
    pub light_source: usize,
}

const MAX_BOUNCES: u64 = 5;
const SURVIVAL_PROBABILITY: f64 = 0.9;

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

impl Scene {
    pub fn new(camera: Ray, objects: Vec<Object>) -> Self {
        Self {
            light_source: 'find_source: {
                for (i, obj) in objects.iter().enumerate() {
                    if !obj.emitted.equal_within(&Vec3::zero(), 0.00001) {
                        break 'find_source i;
                    }
                }
                unreachable!();
            },
            camera,
            objects,
        }
    }

    pub fn from_toml<R: Read>(r: &mut R) -> Result<Self, LoadTomlError> {
        type E = LoadTomlError;

        let mut document = String::new();
        r.read_to_string(&mut document).map_err(E::Io)?;
        let spec: SceneSpec = toml::from_str(&document).map_err(E::Parse)?;
        spec.load().map_err(E::MeshLoad)
    }

    pub fn received_radiance(&self, r: &Ray) -> Vec3 {
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
                let (i, pdf) = obj.brdf.sample_incoming(n, o);
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
            let mut rad = if USE_MIS {
                let mut rad_direct = Vec3::zero();

                let (y, ny, mut pdf_light) = self.sample_light_source();
                let i = (y - x).norm();
                if self.mutually_visible(&x, &y) {
                    pdf_light *= (y - x).dot(&(y - x)) / ny.dot(&-i);
                    let (_, pdf_brdf) = obj.brdf.sample_incoming(n, &i);
                    rad_direct += self.objects[self.light_source]
                        .emitted
                        .mult(&obj.brdf.eval(n, o, &i))
                        * n.dot(&i)
                        / (pdf_light + pdf_brdf);
                }

                let (i, pdf_brdf) = obj.brdf.sample_incoming(n, o);
                if let Some(hit) = self.trace_ray(&Ray::new(*x, i)) {
                    if hit.id == self.light_source {
                        let (y, ny, mut pdf_light) = self.sample_light_source();
                        pdf_light *= (y - x).dot(&(y - x)) / ny.dot(&-i);
                        rad_direct += self.objects[self.light_source]
                            .emitted
                            .mult(&obj.brdf.eval(n, o, &i))
                            * n.dot(&i)
                            / (pdf_brdf + pdf_light);
                    }
                }

                rad_direct
            } else {
                let (y, ny, pdf) = self.sample_light_source();
                let i = (y - x).norm();
                let r_sqr = (y - x).dot(&(y - x));
                let visibility = self.indicate_visibility(&x, &y);
                self.objects[self.light_source]
                    .emitted
                    .mult(&obj.brdf.eval(n, o, &i))
                    * visibility
                    * n.dot(&i)
                    * ny.dot(&-i)
                    / (r_sqr * pdf)
            };

            if random::<f64>() < p {
                let (i, pdf_brdf) = obj.brdf.sample_incoming(n, o);
                if let Some(hit) = self.trace_ray(&Ray::new(*x, i)) {
                    rad += self
                        .reflected_radiance(&hit, &-i, depth + 1)
                        .mult(&obj.brdf.eval(n, o, &i))
                        * n.dot(&i)
                        / (pdf_brdf * p);
                }
            }

            rad
        }
    }

    fn sample_light_source(&self) -> (Vec3, Vec3, f64) {
        match self.objects[self.light_source].geometry {
            Geometry::Sphere(Sphere { pos: center, r }) => {
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

    fn indicate_visibility(&self, x: &Vec3, y: &Vec3) -> f64 {
        if self.mutually_visible(&x, &y) {
            1.
        } else {
            0.
        }
    }

    fn mutually_visible(&self, x: &Vec3, y: &Vec3) -> bool {
        const ERR_MARGIN: f64 = 0.001;
        let diff = y - x;
        let x_to_y = Ray {
            pos: *x,
            dir: diff.norm(),
        };
        if let Some(hit) = self.trace_ray(&x_to_y) {
            hit.t + ERR_MARGIN >= diff.mag()
        } else {
            true
        }
    }

    fn trace_ray(&self, ray: &Ray) -> Option<Hit> {
        let mut nearest_hit: Option<Hit> = None;
        for (i, obj) in self.objects.iter().enumerate() {
            if let Some(mut hit) = obj.geometry.intersect(ray) {
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

#[derive(Deserialize)]
pub struct SceneSpec {
    camera: RaySpec,
    objects: Vec<ObjectSpec>,
}

#[derive(Deserialize)]
pub struct RaySpec {
    pos: [f64; 3],
    dir: [f64; 3],
}

#[derive(Deserialize)]
pub struct ObjectSpec {
    emitted: Option<[f64; 3]>,
    brdf: BRDFSpec,
    geometry: GeometrySpec,
    transforms: Option<Vec<TransformSpec>>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum BRDFSpec {
    Diffuse {
        kd: [f64; 3],
    },
    Specular {
        ks: [f64; 3],
    },
    Phong {
        kd: f64,
        ks: f64,
        color_d: [f64; 3],
        color_s: [f64; 3],
        power: usize,
    },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum GeometrySpec {
    Sphere { pos: [f64; 3], r: f64 },
    Plane { pos: [f64; 3], n: [f64; 3] },
    Mesh { path: String },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformSpec {
    Translate([f64; 3]),
    Scale(f64),
    RotateX(f64),
    RotateY(f64),
    RotateZ(f64),
}

pub enum LoadTomlError {
    Io(io::Error),
    Parse(toml::de::Error),
    MeshLoad(MeshLoadError),
}

impl SceneSpec {
    pub fn load(self) -> Result<Scene, MeshLoadError> {
        let camera = Ray {
            pos: self.camera.pos.into(),
            dir: self.camera.dir.into(),
        };

        let objects: Result<Vec<Object>, MeshLoadError> = self
            .objects
            .into_iter()
            .map(|spec| {
                Ok(Object {
                    emitted: spec.emitted.map(|e| e.into()).unwrap_or_else(|| Vec3::zero()),
                    brdf: match spec.brdf {
                        BRDFSpec::Diffuse { kd } => BRDF::Diffuse(kd.into()),
                        BRDFSpec::Specular { ks } => BRDF::Specular(ks.into()),
                        BRDFSpec::Phong {
                            kd,
                            ks,
                            color_d,
                            color_s,
                            power,
                        } => BRDF::Phong {
                            kd,
                            ks,
                            color_d: color_d.into(),
                            color_s: color_s.into(),
                            power: power as i32,
                        },
                    },
                    geometry: {
                        let mut geom = match spec.geometry {
                            GeometrySpec::Sphere { pos, r } => {
                                Geometry::Sphere(Sphere { pos: pos.into(), r })
                            }
                            GeometrySpec::Plane { pos, n } => Geometry::Plane(Plane {
                                pos: pos.into(),
                                n: n.into(),
                            }),
                            GeometrySpec::Mesh { path } => {
                                let f = File::open(path).map_err(MeshLoadError::IO)?;
                                let r = BufReader::new(f);
                                Geometry::Mesh(Mesh::load(r)?)
                            }
                        };
                        for t in spec.transforms.unwrap_or_else(Vec::new) {
                            match t {
                                TransformSpec::Translate([x, y, z]) => {
                                    geom.translate(&Vec3 { x, y, z });
                                }
                                TransformSpec::Scale(s) => {
                                    geom.scale(s);
                                }
                                TransformSpec::RotateX(angle) => {
                                    geom.rotate_x(angle);
                                }
                                TransformSpec::RotateY(angle) => {
                                    geom.rotate_y(angle);
                                }
                                TransformSpec::RotateZ(angle) => {
                                    geom.rotate_z(angle);
                                }
                            }
                        }
                        if let Geometry::Mesh(ref mut mesh) = geom {
                            mesh.accelerate();
                        }
                        geom
                    },
                })
            })
            .collect();

        Ok(Scene::new(camera, objects?))
    }
}
