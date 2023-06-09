use std::iter::Iterator;

use crate::vec3::Vec3;

#[derive(Copy, Clone, Debug)]
pub struct BoundingBox {
    min: Vec3,
    max: Vec3
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
}

// pub struct Octree {
//     nodes: Vec<Node>
// }

// enum Node {
//     Internal { children: Vec<usize> },
//     Leaf { triangles: Vec<usize> }
// }

// impl Octree {
//     const MAX_DEPTH: usize = 10;
//     const MAX_TRIS_IN_NODE: usize = 9;

//     pub fn 
// }