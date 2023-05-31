use std::fs::File;
use std::io;
use std::io::Write;

use crate::vec3::Vec3;

pub struct Image {
    pub width: usize,
    pub height: usize,
    pub maxval: u32,
    pixels: Vec<Vec3>,
}

impl Image {
    pub fn new(width: usize, height: usize, maxval: u32) -> Self {
        Self {
            width,
            height,
            maxval,
            pixels: vec![Vec3::zero(); width * height],
        }
    }

    pub fn dump(&self, f: &mut File) -> io::Result<()> {
        writeln!(f, "P3")?;
        writeln!(f, "{} {}", self.width, self.height)?;
        writeln!(f, "{}", self.maxval)?;
        for r in 0..self.height {
            for c in 0..self.width {
                let mut color = self.pixels[self.width * r + c];
                color = color.clamp(0.0, self.maxval as f64);
                write!(f, "{} {} {}  ", color.x, color.y, color.z)?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }

    pub fn set(&mut self, r: usize, c: usize, color: Vec3) {
        self.pixels[self.width * r + c] = color;
    }
}
