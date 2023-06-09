# Pathtracer

A multithreaded CPU-driven raytracer using pathtracing.

![Raytraced scene of the Cornell Box with a diffuse and a specular ball.](cornell-box.png)

To run:
```shell
cargo run --profile release # 4 samples per pixel
cargo run --profile release -- <samples_per_pixel>
```
