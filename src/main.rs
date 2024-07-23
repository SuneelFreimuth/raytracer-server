#[allow(dead_code)]
use std::collections::{HashMap};
use std::env::args;
use std::fs::File;
use std::io::BufReader;
use std::process::exit;

mod server;
mod geometry;
mod scene;
use crate::server::Server;
use crate::geometry::{MeshLoadError};
use crate::scene::{LoadTomlError, Scene};

const PORT: &str = "8080";
const SCENE_NAMES: [&str; 3] = ["cornell_box", "cubes", "flying_unicorn"];

#[tokio::main]
async fn main() {
    let args = args().collect::<Vec<String>>();
    if args.len() < 2 {
        eprintln!("Usage: raytracer-server <scenes directory>");
        return;
    }

    let scene_dir = &args[1];

    let scenes = HashMap::from_iter(SCENE_NAMES.map(|name| {
        (
            name.to_string(),
            load_scene(&format!("{scene_dir}/{name}.toml")),
        )
    }));

    let server = Server::new(scenes);
    server.listen(PORT).await;
}

fn load_scene(path: &String) -> Scene {
    let mut f = BufReader::new(File::open(path).unwrap());
    match Scene::from_toml(&mut f) {
        Ok(scene) => scene,
        Err(LoadTomlError::Io(err)) | Err(LoadTomlError::MeshLoad(MeshLoadError::IO(err))) => {
            eprintln!("Failed to load scene {path}: {err}");
            exit(1);
        }
        Err(_) => {
            eprintln!("Failed to load scene {path}.");
            exit(1);
        }
    }
}

