pub enum Target {
    Image,
    Window,
}

pub const WIDTH: usize = 600;
pub const HEIGHT: usize = 450;
pub const USE_MIS: bool = true;
pub const RENDER_TO: Target = Target::Window;
pub const PPM_FILE: &str = if USE_MIS {
    "with_mis.ppm"
} else {
    "without_mis.ppm"
};
