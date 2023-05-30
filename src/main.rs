mod vec3;

use vec3::Vec3;

use std::io::*;
use std::{fs::File, io};

const WIDTH: usize = 100;
const HEIGHT: usize = 100;
const MAXVAL: u32 = 255;

fn clamp<T: PartialOrd>(x: T, low: T, hi: T) -> T {
    if x < low {
        low
    } else if x > hi {
        hi
    } else {
        x
    }
}

struct PPMImage {
    width: usize,
    height: usize,
    maxval: u32,
    pixels: Vec<Vec3>,
}

impl PPMImage {
    pub fn new(width: usize, height: usize, maxval: u32) -> Self {
        Self {
            width,
            height,
            maxval,
            pixels: Vec::with_capacity(width * height),
        }
    }

    pub fn dump(&self, f: &mut File) -> io::Result<()> {
        writeln!(f, "P3\n")?;
        writeln!(f, "{} {}", self.width, self.height)?;
        for r in 0..self.height {
            for c in 0..self.width {
                let mut color = self.pixels[self.width * r + c];
                color = color.clamp(0, self.maxval);
                write!(f, "{} {} {}", color.x, color.y, color.z)?;
            }
        }
        Ok(())
    }

    pub fn set(&mut self, r: usize, c: usize, color: Vec3) {
        self.pixels[self.width * r + c] = color;
    }
}

fn main() -> io::Result<()> {
    let img = PPMImage::new(WIDTH, HEIGHT, MAXVAL);
    let mut f = File::create("image.ppm")?;
    img.dump(&mut f)?;

    Ok(())
}
