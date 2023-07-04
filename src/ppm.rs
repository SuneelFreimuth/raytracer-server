use std::io;
use std::io::Write;

pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![0; 3 * width * height],
        }
    }

    pub fn dump<W: Write>(&self, w: &mut W) -> io::Result<()> {
        writeln!(w, "P3")?;
        writeln!(w, "{} {}", self.width, self.height)?;
        writeln!(w, "255")?;
        for r in 0..self.height {
            for c in 0..self.width {
                let i = 3 * (self.width * r + c);
                let [r, g, b] = self.pixels[i..i + 3] else { unreachable!() };
                write!(w, "{r} {g} {b} ")?;
            }
            write!(w, "\n")?;
        }
        Ok(())
    }

    // Color must be an rgb triple `&[r, g, b]`.
    pub fn set(&mut self, r: usize, c: usize, color: &[u8]) {
        assert!(color.len() == 3);
        let i = 3 * (self.width * r + c);
        self.pixels[i..i + 3].copy_from_slice(color);
    }
}
