use std::fs::File;
use std::io;
use std::io::Write;

use crate::vec3::Vec3;

pub struct Image {
    pub width: usize,
    pub height: usize,
    pub maxval: f64,
    pub pixels: Vec<Vec3>,
}

impl Image {
    pub fn new(width: usize, height: usize, maxval: u64) -> Self {
        Self {
            width,
            height,
            maxval: maxval as f64,
            pixels: vec![Vec3::zero(); width * height],
        }
    }

    pub fn dump<T: Write>(&self, f: &mut T) -> io::Result<()> {
        writeln!(f, "P3")?;
        writeln!(f, "{} {}", self.width, self.height)?;
        writeln!(f, "{}", self.maxval)?;
        for r in 0..self.height {
            for c in 0..self.width {
                let Vec3 { x, y, z } = self.pixels[self.width * r + c];
                write!(f, "{} {} {} ", x as u64, y as u64, z as u64)?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }

    pub fn set(&mut self, r: usize, c: usize, color: Vec3) {
        self.pixels[self.width * r + c] = color.clamp(0., self.maxval);
    }
}
