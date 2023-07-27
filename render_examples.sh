#!/bin/sh
set -e

cargo build --profile release
mkdir -p examples
for name in cornell_box cubes
do
    ./target/release/raytracer --image "examples/$name.png" --spp 64 --scene "scenes/$name.toml"
done