use std::iter::Iterator;
use std::io::BufRead;
use obj::{load_obj, Obj};

use crate::vec3::{determinant3, Vec3, Ray};

#[derive(Clone, Debug)]
pub enum Body {
    Sphere(Sphere),
    Plane(Plane),
    Mesh(Mesh),
}

#[derive(Copy, Clone, Debug)]
pub struct Sphere {
    pub pos: Vec3,
    pub r: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct Plane {
    pub pos: Vec3,
    pub n: Vec3,
}

#[derive(Clone, Debug)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub indices: Vec<usize>,
    pub octree: Octree,
    pub bounding_box: BoundingBox,
}

#[derive(Copy, Debug, Clone)]
pub struct Hit {
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
                let mut nearest_hit: Option<Hit> = None;
                for tri in mesh.triangles() {
                    if let Some(hit) = tri.intersect(ray) {
                        match nearest_hit {
                            Some(nh) => {
                                if hit.t < nh.t {
                                    nearest_hit = Some(hit);
                                }
                            }
                            None => {
                                nearest_hit = Some(hit);
                            }
                        }
                    }
                }
                nearest_hit
            }
        }
    }
}

pub struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
}

impl Triangle {
    pub fn normal(&self) -> Vec3 {
        (self.v2 - self.v0).cross(&(self.v1 - self.v0)).norm()
    }

    pub fn intersect(&self, ray: &Ray) -> Option<Hit> {
        // Moller-Trumbore ray intersection algorithm.
        let n = self.normal();
        if n.dot(&ray.dir).abs() < 0.0001 {
            return None;
        }

        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        let b = ray.pos - self.v0;

        let det = determinant3(&-ray.dir, &e1, &e2);

        // TODO: Inline for better perf?
        let t = determinant3(&b, &e1, &e2) / det;
        let u = determinant3(&-ray.dir, &b, &e2) / det;
        let v = determinant3(&-ray.dir, &e1, &b) / det;

        if u < 0. || u > 1. || v < 0. || u + v > 1. {
            return None;
        }

        if t > 0.0001 {
            let n = if n.dot(&-ray.dir) >= 0. { n } else { -n };
            Some(Hit {
                t,
                pos: ray.eval(t) + 0.00001 * n,
                n,
                id: 10000000,
            })
        } else {
            None
        }
    }
}

#[test]
fn test_triangle_intersection() {
    // Triangle in XY-plane
    let t = Triangle {
        v0: Vec3::new(-1., -1., 0.),
        v1: Vec3::new(1., -1., 0.),
        v2: Vec3::new(0., 1., 0.),
    };
    // Ray 1 unit off the XY-plane pointing directly into the triangle
    let r = Ray {
        pos: Vec3::new(0., 0., 1.),
        dir: Vec3::new(0., 0., -1.),
    };
    let result = t.intersect(&r);
    match result {
        Some(hit) => {
            if (hit.t - 1.).abs() >= 0.0001 {
                panic!("distance should be 1, got {:.6}", hit.t);
            }
            if !hit.n.equal_within(&-r.dir, 0.00001) {
                panic!(
                    "hit normal should be the opposite of the ray direction, got {:?}",
                    hit.n
                );
            }
        }
        None => panic!("ray should have hit"),
    }

    let r = Ray {
        pos: Vec3::new(0., 0., 1.),
        dir: Vec3::new(0., -5., 1.),
    };
    assert!(t.intersect(&r).is_none());
}

pub struct TriangleIterator<'a> {
    mesh: &'a Mesh,
    i: usize,
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
    pub fn new(vertices: Vec<Vec3>, normals: Vec<Vec3>, indices: Vec<usize>) -> Self {
        let mut triangles: Vec<Triangle> = Vec::new();
        let mut i = 0;
        while i < indices.len() {
            triangles.push(Triangle {
                v0: vertices[indices[i]],
                v1: vertices[indices[i + 1]],
                v2: vertices[indices[i + 2]],
            });
            i += 3;
        }
        Self {
            bounding_box: BoundingBox::enclose(&vertices),
            vertices,
            normals,
            indices,
            octree: Octree::new(triangles),
        }
    }

    pub fn load<F: BufRead>(f: F) -> obj::ObjResult<Self> {
        let model: Obj<obj::Vertex, usize> = load_obj(f)?;
        let vertices = model.vertices.iter().map(|v| v.position.into()).collect();
        Ok(Mesh::new(
            vertices,
            model
                .vertices
                .iter()
                .map(|v| Vec3::from(v.normal).norm())
                .collect(),
            model.indices,
        ))
    }

    pub fn cube(p: &Vec3, s: f64) -> Self {
        let vertices = vec![
            Vec3::new(p.x, p.y, p.z),
            Vec3::new(p.x, p.y, p.z + s),
            Vec3::new(p.x, p.y + s, p.z),
            Vec3::new(p.x, p.y + s, p.z + s),
            Vec3::new(p.x + s, p.y, p.z),
            Vec3::new(p.x + s, p.y, p.z + s),
            Vec3::new(p.x + s, p.y + s, p.z),
            Vec3::new(p.x + s, p.y + s, p.z + s),
        ];
        Mesh::new(
            vertices,
            vec![], // TODO
            vec![
                1, 3, 7, 1, 5, 7, // Front
                0, 2, 6, 0, 4, 6, // Back
                0, 1, 3, 0, 2, 3, // Left
                4, 5, 7, 4, 6, 7, // Right
                2, 3, 7, 2, 6, 7, // Top
                0, 1, 5, 0, 4, 5, // Bottom
            ]
        )
    }

    pub fn num_triangles(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn get_triangle(&self, i: usize) -> Triangle {
        let v0 = self.vertices[self.indices[i * 3]];
        let v1 = self.vertices[self.indices[i * 3 + 1]];
        let v2 = self.vertices[self.indices[i * 3 + 2]];
        Triangle { v0, v1, v2 }
    }

    pub fn triangles(&self) -> TriangleIterator {
        TriangleIterator { mesh: self, i: 0 }
    }

    // Defined as the centroid
    pub fn center(&self) -> Vec3 {
        self.bounding_box.center()
    }

    pub fn scale(&mut self, s: f64) {
        let center = self.center();
        for v in self.vertices.iter_mut() {
            *v = center + (*v - center) * s;
        }
        self.bounding_box.min = self.bounding_box.min + (self.bounding_box.min - center) * s;
        self.bounding_box.max = self.bounding_box.max + (self.bounding_box.max - center) * s;
    }

    pub fn translate(&mut self, trans: &Vec3) {
        for v in self.vertices.iter_mut() {
            *v += trans;
        }
        self.bounding_box.min += trans;
        self.bounding_box.max += trans;
    }

    pub fn rotate_y(&mut self, angle: f64) {
        let center = self.center();
        for v in self.vertices.iter_mut() {
            *v = center + (*v - center).rotate_y(angle);
        }
        self.fit_bounds();
    }

    pub fn fit_bounds(&mut self) {
        self.bounding_box = BoundingBox::enclose(&self.vertices);
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3
}

impl BoundingBox {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn enclose(points: &Vec<Vec3>) -> Self {
        let mut min = Vec3::repeat(f64::INFINITY);
        let mut max = Vec3::repeat(-f64::INFINITY);
        for p in points {
            if p.x < min.x {
                min.x = p.x;
            }
            if p.x > max.x {
                max.x = p.x;
            }

            if p.y < min.y {
                min.y = p.y;
            }
            if p.y > max.y {
                max.y = p.y;
            }

            if p.z < min.z {
                min.z = p.z;
            }
            if p.z > max.z {
                max.z = p.z;
            }
        }
        Self { min, max }
    }

    pub fn center(&self) -> Vec3 {
        (self.max + self.min) / 2.
    }
}

#[derive(Debug, Clone)]
pub struct Octree {
    nodes: Vec<Node>,
    bounding_box: BoundingBox,
}

#[derive(Debug, Clone)]
enum Node {
    Internal { children: Vec<usize> },
    Leaf { triangles: Vec<usize> }
}

impl Octree {
    const MAX_DEPTH: usize = 10;
    const MAX_TRIS_IN_NODE: usize = 9;

    pub fn new(triangles: Vec<Triangle>) -> Self {
        Self {
            nodes: Vec::new(),
            bounding_box: BoundingBox::new(Vec3::zero(), Vec3::zero())
        }
    }
}