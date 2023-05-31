use num::Num;

pub fn clamp<T: PartialOrd>(x: T, low: T, hi: T) -> T {
    if x < low {
        low
    } else if x > hi {
        hi
    } else {
        x
    }
}

pub fn map<T: Num + Copy>(x: T, low0: T, hi0: T, low1: T, hi1: T) -> T {
    ((x - low0) / (hi0 - low0) + low1) * (hi1 - low1)
}