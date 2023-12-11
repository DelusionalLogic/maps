use freetype::Face;

use crate::math::Vector2;

pub struct CharMetric {
    pub size: Vector2<f32>,
    pub bearing: Vector2<f32>,
    pub advance: i16,
}

pub struct FontMetric {
    pub chars: Vec<CharMetric>,
}

impl FontMetric {
    pub fn load(face: &Face) -> Self {

        let mut chars = Vec::with_capacity(128);

        for i in 0..128 {
            face.load_char(i, freetype::face::LoadFlag::RENDER).unwrap();
            let glyph = face.glyph();
            let metrics = glyph.metrics();
            let size = Vector2::new(metrics.width as f32 /64.0, metrics.height as f32 /64.0);
            let bearing = Vector2::new(
                metrics.horiBearingX as f32 /64.0,
                -metrics.horiBearingY as f32 /64.0,
            );
            let advance = metrics.horiAdvance >> 6;

            chars.push(CharMetric {
                size,
                bearing,
                advance: advance as i16,
            });
        }

        return FontMetric {
            chars,
        };
    }

    pub fn size_str(&self, text: &[u8]) -> (Vector2<f32>, Vector2<f32>) {
        let mut advance = 0.0;
        let mut max = Vector2::new(0.0, 0.0);
        let mut min = Vector2::new(0.0, 0.0);

        for c in text {
            if *c >= 128 { continue; }
            let char = &self.chars[*c as usize];

            let mut char_min = Vector2::new(advance as f32, 0.0);
            char_min += char.bearing;

            let mut char_max = char_min.clone();
            char_max += char.size;

            min.min(&char_min);
            max.max(&char_max);
            advance += char.advance as f32;
        }

        return (min, max);
    }
}
