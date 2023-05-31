use std::{fs::File, io};

mod vec3;
mod util;
mod ppm;

use vec3::*;

const WIDTH: usize = 200;
const HEIGHT: usize = 200;
const MAXVAL: u32 = 255;

// Camera position & direction
// const Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).normalize());

// const CAMERA: Ray = Ray {
//     pos: Vec3::new(50., 52., 295.6),
//     dir: Vec3::new(0., -0.042612, -1.).norm(),
// };

// static mut rng: rand::ThreadRng = rand::thread_rng();

pub enum Shape {
    Sphere,
}

pub struct Scene {
    objects: Vec<Shape>
}

impl Scene {
    pub fn render(&self, img: &mut ppm::Image, camera: &Ray) {
        // let cx = Vec3::repeat(img.width as f64 * 0.5135 / img.height as f64);
        // let cy = cx.cross(&camera.dir).norm() * 0.5135;
        // let num_samples = 4;
        // for y in 0..img.height {
        //     for x in 0..img.width {
        //         let i = (img.height - y - 1)*img.width + x;

        //         for sy in 0..2 {
        //             for sx in 0..2 {
        //                 let mut rad = Vec3::zero();
        //                 for s in 0..num_samples {
        //                     let r1 =  
        //                 }
        //             }
        //         }
        //     }
        // }

        // Vec cx = Vec(w*.5135/h), cy = (cx.cross(cam.d)).normalize()*.5135;
        // for ( int y = 0; y < h; y++ ) {
        //     for ( int x = 0; x < w; x++ ) {
        //         const int i = (h - y - 1)*w + x;

        //         for ( int sy = 0; sy < 2; ++sy ) {
        //             for ( int sx = 0; sx < 2; ++sx ) {
        //                 Vec r;
        //                 for ( int s = 0; s<samps; s++ ) {
        //                     double r1 = 2*rng(), dx = r1<1 ? sqrt(r1)-1 : 1-sqrt(2-r1);
        //                     double r2 = 2*rng(), dy = r2<1 ? sqrt(r2)-1 : 1-sqrt(2-r2);
        //                     Vec d = cx*(((sx+.5 + dx)/2 + x)/w - .5) +
        //                         cy*(((sy+.5 + dy)/2 + y)/h - .5) + cam.d;
        //                     r = r + receivedRadiance(Ray(cam.o, d.normalize()), 1, true)*(1./samps);
        //                 }
        //                 c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z))*.25;
        //             }
        //         }
        //     }
        // }
    }
}


fn main() -> io::Result<()> {
    let mut img = ppm::Image::new(WIDTH, HEIGHT, MAXVAL);
    // let scene = Scene {
    //     objects: vec![
            
    //     ],
    // }
    // for r in 0..img.width {
    //     for c in 0..img.height {
    //         let l = c as f64 / img.width as f64 * img.maxval as f64;
    //         img.set(r, c, Vec3::new(l, l, l));
    //     }
    // }

    let mut f = File::create("image.ppm")?;
    img.dump(&mut f)?;

    Ok(())
}
