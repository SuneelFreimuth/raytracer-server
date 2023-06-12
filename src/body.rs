use obj::{load_obj, Obj};
use std::io::BufRead;
use std::iter::Iterator;

use crate::vec3::{determinant3, Ray, Vec3};

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
    pub octree: Option<Octree>,
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
            Self::Mesh(mesh) => mesh.intersect(ray),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle {
    a: Vec3,
    b: Vec3,
    c: Vec3,
}

impl Triangle {
    pub fn normal(&self) -> Vec3 {
        (self.c - self.a).cross(&(self.b - self.a)).norm()
    }

    pub fn intersect(&self, ray: &Ray) -> Option<Hit> {
        // Moller-Trumbore ray intersection algorithm.
        let n = self.normal();
        if n.dot(&ray.dir).abs() < 0.0001 {
            return None;
        }

        let e1 = self.b - self.a;
        let e2 = self.c - self.a;
        let b = ray.pos - self.a;

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
        a: Vec3::new(-1., -1., 0.),
        b: Vec3::new(1., -1., 0.),
        c: Vec3::new(0., 1., 0.),
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
            let t = self.mesh.triangle(self.i);
            self.i += 1;
            Some(t)
        } else {
            None
        }
    }
}

impl Mesh {
    pub fn new(vertices: Vec<Vec3>, normals: Vec<Vec3>, indices: Vec<usize>) -> Self {
        let mut triangles: Vec<(usize, Triangle)> = Vec::new();
        let mut i = 0;
        while i < indices.len() {
            triangles.push((
                i / 3,
                Triangle {
                    a: vertices[indices[i]],
                    b: vertices[indices[i + 1]],
                    c: vertices[indices[i + 2]],
                },
            ));
            i += 3;
        }
        Self {
            bounding_box: BoundingBox::enclose(&vertices),
            vertices,
            normals,
            indices,
            octree: None,
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

    pub fn accelerate(&mut self) {
        self.octree = Some(Octree::build(self));
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
            ],
        )
    }

    pub fn num_triangles(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn triangle(&self, i: usize) -> Triangle {
        let a = self.vertices[self.indices[i * 3]];
        let b = self.vertices[self.indices[i * 3 + 1]];
        let c = self.vertices[self.indices[i * 3 + 2]];
        Triangle { a, b, c }
    }

    pub fn triangles(&self) -> TriangleIterator {
        TriangleIterator { mesh: self, i: 0 }
    }

    pub fn intersect(&self, ray: &Ray) -> Option<Hit> {
        if let Some(octree) = &self.octree {
            octree.intersect(ray)
        } else {
            // Brute force
            let mut nearest_hit: Option<Hit> = None;
            for tri in self.triangles() {
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

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
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

    pub fn enclose_triangle(triangle: &Triangle) -> Self {
        let Triangle { a, b, c } = triangle;
        let mut min = a.clone();
        let mut max = a.clone();

        for p in [b, c] {
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

    pub fn enclose_triangles(triangles: &Vec<Triangle>) -> Self {
        let mut min = Vec3::repeat(f64::INFINITY);
        let mut max = Vec3::repeat(-f64::INFINITY);
        for Triangle { a, b, c } in triangles {
            for p in [a, b, c] {
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
        }
        Self { min, max }
    }

    fn _overlaps(&self, b: &Self) -> bool {
        self.min.x <= b.max.x
            && self.max.x >= b.min.x
            && self.min.y <= b.max.y
            && self.max.y >= b.min.y
            && self.min.z <= b.max.z
            && self.max.z >= b.min.z
    }

    pub fn overlaps(&self, b: &Self) -> bool {
        self._overlaps(b) || b._overlaps(self)
    }

    pub fn contains(&self, p: &Vec3) -> bool {
        self.min.x <= p.x
            && p.x <= self.max.x
            && self.min.y <= p.y
            && p.y <= self.max.y
            && self.min.z <= p.z
            && p.z <= self.max.z
    }

    pub fn intersect(&self, r: &Ray) -> Option<f64> {
        const EPS: f64 = 0.0000001;
        let BoundingBox { min, max } = self;

        // Intersects left?
        let t = (min.x - r.pos.x) / r.dir.x;
        if t >= EPS {
            let p = r.eval(t);
            if min.y <= p.y && p.y <= max.y && min.z <= p.z && p.z <= max.z {
                return Some(t);
            }
        }

        // Intersects right?
        let t = (max.x - r.pos.x) / r.dir.x;
        if t >= EPS {
            let p = r.eval(t);
            if min.y <= p.y && p.y <= max.y && min.z <= p.z && p.z <= max.z {
                return Some(t);
            }
        }

        // Intersects bottom?
        let t = (min.y - r.pos.y) / r.dir.y;
        if t >= EPS {
            let p = r.eval(t);
            if min.x <= p.x && p.x <= max.x && min.z <= p.z && p.z <= max.z {
                return Some(t);
            }
        }

        // Intersects top?
        let t = (max.y - r.pos.y) / r.dir.y;
        if t >= EPS {
            let p = r.eval(t);
            if min.x <= p.x && p.x <= max.x && min.z <= p.z && p.z <= max.z {
                return Some(t);
            }
        }

        // Intersects back?
        let t = (min.z - r.pos.z) / r.dir.z;
        if t >= EPS {
            let p = r.eval(t);
            if min.x <= p.x && p.x <= max.x && min.y <= p.y && p.y <= max.y {
                return Some(t);
            }
        }

        // Intersects front?
        let t = (max.z - r.pos.z) / r.dir.z;
        if t >= EPS {
            let p = r.eval(t);
            if min.x <= p.x && p.x <= max.x && min.y <= p.y && p.y <= max.y {
                return Some(t);
            }
        }

        None
    }

    fn intersect_line_segment(&self, a: &Vec3, b: &Vec3) -> bool {
        let r = Ray::new(*a, (b - a).norm());

        if let Some(t) = self.intersect(&r) {
            if t <= (b - a).mag() {
                return true;
            }
        }
        false
    }

    pub fn overlaps_triangle(&self, t: &Triangle) -> bool {
        let b = Self::enclose(&vec![t.a, t.b, t.c]);
        self.overlaps(&b)

        // let Triangle { a, b, c } = t;
        // if self.contains(a) || self.contains(b) || self.contains(c) {
        //     return true;
        // }

        // self.intersect_line_segment(a, b)
        //     || self.intersect_line_segment(a, c)
        //     || self.intersect_line_segment(b, c)
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) / 2.
    }

    pub fn octant(&self, i: usize) -> Self {
        let BoundingBox { min, max } = self;
        let center = self.center();
        match i {
            0 => Self::new(min.clone(), center),
            1 => Self::new(
                Vec3::new(min.x, min.y, center.z),
                Vec3::new(center.x, center.y, max.z),
            ),
            2 => Self::new(
                Vec3::new(min.x, center.y, min.z),
                Vec3::new(center.x, max.y, center.z),
            ),
            3 => Self::new(
                Vec3::new(min.x, center.y, center.z),
                Vec3::new(center.x, max.y, max.z),
            ),
            4 => Self::new(
                Vec3::new(center.x, min.y, min.z),
                Vec3::new(max.x, center.y, center.z),
            ),
            5 => Self::new(
                Vec3::new(center.x, min.y, center.z),
                Vec3::new(max.x, center.y, max.z),
            ),
            6 => Self::new(
                Vec3::new(center.x, center.y, min.z),
                Vec3::new(max.x, max.y, center.z),
            ),
            7 => Self::new(center, max.clone()),
            _ => panic!("octant index out of bounds"),
        }
    }

    pub fn octants(&self) -> [Self; 8] {
        [
            self.octant(0),
            self.octant(1),
            self.octant(2),
            self.octant(3),
            self.octant(4),
            self.octant(5),
            self.octant(6),
            self.octant(7),
        ]
    }
}

#[test]
fn test_octants() {
    let b = BoundingBox::new(Vec3::new(-1., -1., -1.), Vec3::new(1., 1., 1.));
    assert_eq!(
        b.octants(),
        [
            BoundingBox::new(Vec3::new(-1., -1., -1.), Vec3::new(0., 0., 0.)),
            BoundingBox::new(Vec3::new(-1., -1., 0.), Vec3::new(0., 0., 1.)),
            BoundingBox::new(Vec3::new(-1., 0., -1.), Vec3::new(0., 1., 0.)),
            BoundingBox::new(Vec3::new(-1., 0., 0.), Vec3::new(0., 1., 1.)),
            BoundingBox::new(Vec3::new(0., -1., -1.), Vec3::new(1., 0., 0.)),
            BoundingBox::new(Vec3::new(0., -1., 0.), Vec3::new(1., 0., 1.)),
            BoundingBox::new(Vec3::new(0., 0., -1.), Vec3::new(1., 1., 0.)),
            BoundingBox::new(Vec3::new(0., 0., 0.), Vec3::new(1., 1., 1.)),
        ]
    );
}

#[derive(Debug, Clone)]
pub struct Octree {
    pub nodes: Vec<Node>,
    pub bounding_box: BoundingBox,
}

#[derive(Debug, Clone)]
pub enum Node {
    Parent { children: [Option<usize>; 8] },
    Leaf { triangles: Vec<(usize, Triangle)> },
}

impl Octree {
    const MAX_DEPTH: usize = 10;
    const SMALL_NODE: usize = 9;

    pub fn build(mesh: &Mesh) -> Self {
        let mut nodes: Vec<Node> = Vec::new();
        Self::_build(
            mesh,
            mesh.bounding_box,
            mesh.triangles().enumerate().collect(),
            &mut nodes,
            1,
        );
        Self {
            nodes,
            bounding_box: mesh.bounding_box,
        }
    }

    fn _build(
        mesh: &Mesh,
        bounding_box: BoundingBox,
        triangles: Vec<(usize, Triangle)>,
        nodes: &mut Vec<Node>,
        depth: usize,
    ) -> Option<usize> {
        if triangles.len() == 0 {
            return None;
        }

        if triangles.len() <= Self::SMALL_NODE || depth >= Self::MAX_DEPTH {
            let i_new = Self::create_node(nodes, Node::Leaf { triangles });
            return Some(i_new);
        }

        let octants = bounding_box.octants();
        let mut octant_triangles: [Vec<(usize, Triangle)>; 8] = [
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        ];

        for (id, t) in triangles {
            for (i, octant) in octants.iter().enumerate() {
                if octant.overlaps_triangle(&t) {
                    octant_triangles[i].push((id, t));
                }
            }
        }

        let mut children: [Option<usize>; 8] = [None; 8];
        let i_new = Self::create_node(nodes, Node::Parent { children });

        for i in 0..8 {
            children[i] = Self::_build(
                mesh,
                octants[i],
                octant_triangles[i].clone(),
                nodes,
                depth + 1,
            );
        }

        nodes[i_new] = Node::Parent { children };

        Some(i_new)
    }

    pub fn depth(&self) -> i64 {
        self._depth_recurse(0)
    }

    pub fn _depth_recurse(&self, i: usize) -> i64 {
        if let Node::Parent { children } = self.nodes[i] {
            1 + children
                .iter()
                .map(|c| match c {
                    Some(n) => self._depth_recurse(*n),
                    None => -1,
                })
                .max()
                .unwrap()
        } else {
            0
        }
    }

    pub fn intersect(&self, ray: &Ray) -> Option<Hit> {
        if self.nodes.is_empty() {
            None
        } else {
            self._intersect_recurse(&self.nodes[0], self.bounding_box, ray)
        }
    }

    fn _intersect_recurse(
        &self,
        node: &Node,
        bounding_box: BoundingBox,
        ray: &Ray,
    ) -> Option<Hit> {
        match node {
            Node::Parent { children } => {
                let mut octant_search_order = [0, 1, 2, 3, 4, 5, 6, 7];
                let octants = self.bounding_box.octants();
                let dist_to_ray = |octant: &BoundingBox| (octant.center() - ray.pos).mag();
                for i in 1..octant_search_order.len() {
                    let mut j = i;
                    while j > 0
                        && dist_to_ray(&octants[octant_search_order[j - 1]])
                            > dist_to_ray(&octants[octant_search_order[j]])
                    {
                        octant_search_order.swap(j, j - 1);
                        j -= 1;
                    }
                }

                let octants = bounding_box.octants();
                for i in octant_search_order {
                    let octant = octants[i];
                    if let Some(n) = children[i] {
                        if octant.intersect(ray).is_some() {
                            if let Some(hit) = self._intersect_recurse(
                                &self.nodes[n],
                                octant,
                                ray,
                            ) {
                                return Some(hit);
                            }
                        }
                    }
                }
                None
            }
            Node::Leaf { triangles } => {
                let mut nearest_hit: Option<Hit> = None;
                for (_, tri) in triangles {
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

    fn create_node(nodes: &mut Vec<Node>, n: Node) -> usize {
        nodes.push(n);
        nodes.len() - 1
    }
}
