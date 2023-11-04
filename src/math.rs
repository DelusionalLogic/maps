pub trait Sqrt {
    type Output;

    fn sqrt(self) -> Self::Output;
}

impl Sqrt for f32 {
    type Output = f32;

    fn sqrt(self) -> Self::Output {
        return self.sqrt();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

impl <T> Vector2<T> {
    pub fn new(x: T, y: T) -> Self {
        return Vector2 {
            x,
            y,
        }
    }
}

impl <T> Vector2<T>
    where T: std::ops::AddAssign<T> + Copy {

    pub fn addf(&mut self, val: T) {
        self.x += val;
        self.y += val;
    }
}

impl <T> Vector2<T>
    where T: std::ops::DivAssign<T> + Copy {

    pub fn divf(&mut self, val: T) {
        self.x /= val;
        self.y /= val;
    }
}

impl <T> Vector2<T>
    where T: std::ops::MulAssign<T> + Copy {

    pub fn mulf(&mut self, val: T) {
        self.x *= val;
        self.y *= val;
    }
}

impl <T> Vector2<T>
    where T: std::ops::AddAssign<T> + Copy {

    pub fn addv2(&mut self, other: &Vector2<T>) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl <T> Vector2<T>
    where T: std::ops::SubAssign<T> + Copy {

    pub fn subv2(&mut self, other: &Vector2<T>) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

impl <T> Vector2<T>
    where T: std::ops::Neg<Output = T> + Copy {

    pub fn negate(&mut self) {
        self.x = -self.x;
        self.y = -self.y;
    }
}

impl <T> Vector2<T>
    where T: std::ops::Neg<Output = T> + Copy {

    pub fn normal(&mut self) {
        let x = self.x;
        self.x = self.y;
        self.y = -x;
    }
}

impl <T> Vector2<T>
    where T: std::ops::Mul<Output = T> + std::ops::Sub<Output = T> + Copy {

    pub fn cross(&self, other: &Self) -> T {
        return self.x * other.y - self.y * other.x;
    }
}

impl <T> Vector2<T>
    where T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy {

    pub fn dot(&self, other: &Self) -> T {
        return self.x * other.x + self.y * other.y;
    }
}

impl <T> Vector2<T>
    where T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy {

    pub fn len_squared(&self) -> T {
        return self.x * self.x + self.y * self.y;
    }
}

impl <T> Vector2<T>
    where T: Sqrt + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy {

    pub fn len(&self) -> <T as Sqrt>::Output {
        return self.len_squared().sqrt();
    }
}

impl <T> Vector2<T>
    where T: Sqrt<Output = T> + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::DivAssign<T> + Copy {

    pub fn unit(&mut self) {
        let len = self.len();
        self.x /= len;
        self.y /= len;
    }
}


impl Vector2<f32> {
    pub fn apply_transform(&mut self, mat: &[f32; 9]) {
        self.x = mat[0] * self.x + mat[1] * self.y + mat[2];
        self.y = mat[3] * self.x + mat[4] * self.y + mat[5];
    }

    pub fn mulv2(&mut self, other: &Self) {
        self.x *= other.x;
        self.y *= other.y;
    }

    pub fn min(&mut self, other: &Self) {
        self.x = self.x.min(other.x);
        self.y = self.y.min(other.y);
    }

    pub fn max(&mut self, other: &Self) {
        self.x = self.x.max(other.x);
        self.y = self.y.max(other.y);
    }

    pub fn angle(&mut self) -> f32 {
        return f32::atan2(self.y, self.x);
    }
}

impl Vector2<f64> {
    pub fn apply_transform(&mut self, mat: &[f64; 9]) {
        self.x = mat[0] * self.x + mat[1] * self.y + mat[2];
        self.y = mat[3] * self.x + mat[4] * self.y + mat[5];
    }

    pub fn mulv2(&mut self, other: &Self) {
        self.x *= other.x;
        self.y *= other.y;
    }

    pub fn angle(&mut self) -> f64 {
        return f64::atan2(self.y, self.x);
    }
}

pub struct Mat4 {
    pub data: [f64; 16],
}

pub const MAT4_IDENTITY: Mat4 = Mat4{ data: [
           1.0,    0.0,    0.0, 0.0,
           0.0,    1.0,    0.0, 0.0,
           0.0,    0.0,    1.0, 0.0,
           0.0,    0.0,    0.0, 1.0,
]};

pub struct Mat3 {
    pub data: [f64; 9],
}

impl Mat4 {
    pub fn ortho(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Self {
        return Mat4{
            data: [
                2.0/(right-left),              0.0,            0.0, -(right+left)/(right-left),
                             0.0, 2.0/(top-bottom),            0.0, -(top+bottom)/(top-bottom),
                             0.0,              0.0, 2.0/(far-near), -(far+near  )/(far-near  ),
                             0.0,              0.0,            1.0,                          1.0,
            ]
        };
    }

    pub fn scale_2d(x_factor: f64, y_factor: f64) -> Self {
        return Mat4{ data: [
              x_factor,        0.0,        0.0, 0.0,
                   0.0,   y_factor,        0.0, 0.0,
                   0.0,        0.0,        1.0, 0.0,
                   0.0,        0.0,        0.0, 1.0,
        ]};
    }

    pub fn rotate_2d(angle: f64) -> Self {
        return Mat4{ data: [
            angle.cos(), -angle.sin(),        0.0, 0.0,
            angle.sin(),  angle.cos(),        0.0, 0.0,
                    0.0,          0.0,        1.0, 0.0,
                    0.0,          0.0,        0.0, 1.0,
        ]};
    }

    pub fn translate(x: f64, y: f64) -> Self {
        return Mat4{ data: [
               1.0,    0.0,    0.0,   x,
               0.0,    1.0,    0.0,   y,
               0.0,    0.0,    1.0, 0.0,
               0.0,    0.0,    0.0, 1.0,
        ]};
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut out = [0.0; 16];

        for row in 0..4 {
            let row_offset = row * 4;
            for column in 0..4 {
                out[row_offset + column] =
                    (self.data[row_offset + 0] * other.data[column + 0]) +
                    (self.data[row_offset + 1] * other.data[column + 4]) +
                    (self.data[row_offset + 2] * other.data[column + 8]) +
                    (self.data[row_offset + 3] * other.data[column + 12]);
            }
        }

        return Mat4{
            data: out
        };
    }
}

impl Clone for Mat4 {
    fn clone(&self) -> Self {
        return Mat4{
            data: self.data.clone(),
        }
    }
}

impl Into<Mat3> for &Mat4 {
    fn into(self) -> Mat3 {
        return Mat3{ data: [
            self.data[0],  self.data[1],  self.data[3],
            self.data[4],  self.data[5],  self.data[7],
            self.data[12], self.data[13], self.data[15],
        ]};
    }
}

impl Into<[f32; 16]> for Mat4 {
    fn into(self) -> [f32; 16] {
        return [
            self.data[0] as f32,  self.data[1] as f32,  self.data[2] as f32,  self.data[3] as f32,
            self.data[4] as f32,  self.data[5] as f32,  self.data[6] as f32,  self.data[7] as f32,
            self.data[8] as f32,  self.data[9] as f32,  self.data[10] as f32, self.data[11] as f32,
            self.data[12] as f32, self.data[13] as f32, self.data[14] as f32, self.data[15] as f32,
        ];
    }
}

impl Into<[f32; 16]> for &Mat4 {
    fn into(self) -> [f32; 16] {
        return [
            self.data[0] as f32,  self.data[1] as f32,  self.data[2] as f32,  self.data[3] as f32,
            self.data[4] as f32,  self.data[5] as f32,  self.data[6] as f32,  self.data[7] as f32,
            self.data[8] as f32,  self.data[9] as f32,  self.data[10] as f32, self.data[11] as f32,
            self.data[12] as f32, self.data[13] as f32, self.data[14] as f32, self.data[15] as f32,
        ];
    }
}

impl Into<[f32; 9]> for Mat3 {
    fn into(self) -> [f32; 9] {
        return [
            self.data[0] as f32, self.data[1] as f32, self.data[2] as f32,
            self.data[3] as f32, self.data[4] as f32, self.data[5] as f32,
            self.data[6] as f32, self.data[7] as f32, self.data[8] as f32,
        ];
    }
}
