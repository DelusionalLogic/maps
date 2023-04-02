#[derive(Debug, Clone, Copy)]
pub struct Vector2 {
    pub x: f32,
    pub y: f32,
}

impl Vector2 {
    pub fn new(x: f32, y: f32) -> Self {
        return Vector2 {
            x,
            y,
        }
    }

    pub fn addf(&mut self, val: f32) {
        self.x += val;
        self.y += val;
    }

    pub fn divf(&mut self, val: f32) {
        self.x /= val;
        self.y /= val;
    }

    pub fn addv2(&mut self, other: &Vector2) {
        self.x += other.x;
        self.y += other.y;
    }

    pub fn subv2(&mut self, other: &Vector2) {
        self.x -= other.x;
        self.y -= other.y;
    }

    pub fn normal(&mut self) {
        let x = self.x;
        self.x = self.y;
        self.y = -x;
    }

    pub fn len(&self) -> f32 {
        return (self.x.powi(2) + self.y.powi(2)).sqrt();
    }

    pub fn unit(&mut self) {
        let len = self.len();
        self.x /= len;
        self.y /= len;
    }

    pub fn apply_transform(&mut self, mat: &[f32; 9]) {
        self.x = mat[0] * self.x + mat[1] * self.y + mat[2];
        self.y = mat[3] * self.x + mat[4] * self.y + mat[5];
    }

    pub fn mulv2(&mut self, other: &Vector2) {
        self.x *= other.x;
        self.y *= other.y;
    }
}

