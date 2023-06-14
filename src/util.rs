pub fn clamp<T: PartialOrd>(x: T, low: T, hi: T) -> T {
    if x < low {
        low
    } else if x > hi {
        hi
    } else {
        x
    }
}

pub fn map(x: f64, low0: f64, hi0: f64, low1: f64, hi1: f64) -> f64 {
    ((x - low0) / (hi0 - low0) + low1) * (hi1 - low1)
}