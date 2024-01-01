use std::{rc::Rc, collections::HashMap};

use freetype::Face;

use crate::{math::{Vector2, Transform, Mat3}, mapbox::{pmtile::RenderCommand, GlVert}};

pub struct Atlas {
    pub gl_texture: u32,
}

impl Drop for Atlas {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteTextures(1, &self.gl_texture)
        }
    }
}

pub struct LoadedTexture {
    pub atlas: Rc<Atlas>,
    pub uv_min: Vector2<f32>,
    pub uv_max: Vector2<f32>,
}

pub struct TexInfo {
    pub bearing: Vector2<f32>,
    pub size: Vector2<f32>,
    pub texture: LoadedTexture,
}

pub struct CharMetric<'a> {
    pub size: Vector2<f32>,
    pub bearing: Vector2<f32>,
    pub advance: i16,

    pub texdata: &'a TexInfo,
}

pub struct FontMetric {
    face: Face,

    pub characters: HashMap<usize, TexInfo>,
}

// @HACK @COMPLETE: This iterates over unicode values. In reality we probably want to use
// harfbuzz or something to split the string inteo renderable chunks
impl FontMetric {
    pub fn load(face: Face) -> Self {
        return FontMetric {
            face,
            characters: HashMap::new(),
        };
    }

    pub fn size_char(&mut self, char: usize) -> CharMetric {
        self.face.load_char(char, freetype::face::LoadFlag::RENDER).unwrap();
        let glyph = self.face.glyph();
        let metrics = glyph.metrics();
        let size = Vector2::new(metrics.width as f32 /64.0, metrics.height as f32 /64.0);
        let bearing = Vector2::new(
            metrics.horiBearingX as f32 /64.0,
            -metrics.horiBearingY as f32 /64.0,
        );
        let advance = metrics.horiAdvance >> 6;

        return CharMetric {
            size,
            bearing,
            advance: advance as i16,

            texdata: self.load_char(char),
        };
    }

    pub fn load_char(&mut self, char: usize) -> &TexInfo {
        if !self.characters.contains_key(&char) {
            dbg!("Generate for", char);
            self.face.load_char(char, freetype::face::LoadFlag::RENDER).unwrap();
            let glyph = self.face.glyph();
            glyph.render_glyph(freetype::render_mode::RenderMode::Sdf).unwrap();
            let bitmap = glyph.bitmap();

            let mut textures = [0; 1];
            unsafe {
                gl::GenTextures(1, textures.as_mut_ptr());
                gl::BindTexture(gl::TEXTURE_2D, textures[0]);
                gl::TexStorage2D(
                    gl::TEXTURE_2D, 1,
                    gl::R8,
                    bitmap.width(), bitmap.rows()
                );
                gl::TexSubImage2D(
                    gl::TEXTURE_2D, 0,
                    0, 0,
                    bitmap.width(), bitmap.rows(),
                    gl::RED, gl::UNSIGNED_BYTE,
                    bitmap.buffer().as_ptr() as _,
                );
            };

            let atlas = Atlas{
                gl_texture: textures[0],
            };

            let info = TexInfo {
                texture: LoadedTexture { atlas: atlas.into(), uv_min: Vector2::new(0.0, 0.0), uv_max: Vector2::new(1.0, 1.0) },
                bearing: Vector2::new(glyph.bitmap_left() as f32, -glyph.bitmap_top() as f32),
                size: Vector2::new(bitmap.width() as f32, bitmap.rows() as f32)
            };

            self.characters.insert(char, info);

            return self.characters.get(&char).unwrap();
        }

        return self.characters.get(&char).unwrap();
    }

    pub fn size_str(&mut self, text: &[u8]) -> (Vector2<f32>, Vector2<f32>) {
        let mut advance = 0.0;
        let mut max = Vector2::new(0.0, 0.0);
        let mut min = Vector2::new(0.0, 0.0);

        for c in text {
            let char = self.size_char(*c as usize);

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

    pub fn width(&mut self, text: &str) -> f32 {
        let mut width = 0.0;
        for c in text.chars() {
            let char = self.size_char(c as usize);
            width += char.advance as f32;
        }

        return width;
    }

    pub fn layout_text(&mut self, verts: &mut Vec<GlVert>, cmd: &mut Vec<RenderCommand>, text: &str, t: &Transform, pos: Vector2<f32>) -> (Vector2<f32>, Vector2<f32>) {
        let mut pen = Vector2::new(0.0, 0.0);

        let t2d: Mat3 = t.mat().into();
        let t2d32: [f32; 9] = t2d.into();

        let mut emax = Vector2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut emin = Vector2::new(f32::INFINITY, f32::INFINITY);

        for c in text.chars() {
            let sze = self.size_char(c as usize);
            let texture = sze.texdata;

            let min = {
                let mut vec = pen.clone();
                vec += texture.bearing;
                vec
            };
            let max = {
                let mut vec = min.clone();
                vec += texture.size;
                vec
            };

            let ps = [
                {
                    let mut vec = min.clone();
                    vec.apply_transform(&t2d32);
                    vec
                }, {
                    let mut vec = Vector2::new(min.x, max.y);
                    vec.apply_transform(&t2d32);
                    vec
                }, {
                    let mut vec = max.clone();
                    vec.apply_transform(&t2d32);
                    vec
                }, {
                    let mut vec = Vector2::new(max.x, min.y);
                    vec.apply_transform(&t2d32);
                    vec
                },
            ];

            // There's an analytical way of figuring this out, but i'm lazy.
            for p in ps {
                emax.max(&p);
                emin.min(&p);
            }

            for (p, u, v) in [
                (ps[0], 0.0, 0.0),
                (ps[1], 0.0, 1.0),
                (ps[3], 1.0, 0.0),
                (ps[3], 1.0, 0.0),
                (ps[1], 0.0, 1.0),
                (ps[2], 1.0, 1.0),
            ] {
                verts.push(GlVert{x: p.x, y: p.y, norm_x: u, norm_y: v, sign: 0});
            }

            pen.x += sze.advance as f32;

            cmd.push(RenderCommand::PositionedLetter(c, pos, 6));

        }

        return (emin, emax);
    }
}
