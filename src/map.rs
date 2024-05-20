use crate::font::FontMetric;
use crate::mapbox::GlVert;
use crate::mapbox::RawTile;
use crate::math::Transform;
use crate::math::Vector2;
use crate::triangulate;

pub struct Label {
    pub rank: u8,

    pub pos: Vector2<f32>,

    pub min: Vector2<f32>,
    pub max: Vector2<f32>,

    pub cmds: usize,

    pub not_before: f32,
    pub opacity: f32,
    pub opacity_from: Option<usize>,
}

pub enum RenderCommand {
    Simple(usize),
    Target(Vector2<f32>, usize),
    PositionedLetter(char, Vector2<f32>, usize),
}

pub struct GlLayer {
    pub vao: u32,
    pub vbo: u32,
    pub unlabeled: usize,
    pub commands: Vec<RenderCommand>,
    pub blocks: Vec<usize>,
    pub labels: Vec<Label>,
}

impl Drop for GlLayer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteVertexArrays(1, &self.vao);
            gl::DeleteBuffers(1, &self.vbo);
        }
    }
}

pub trait Renderer {
    type Layer;

    fn upload_layer(data: &Vec<crate::mapbox::GlVert>) -> Self::Layer;
    fn upload_multi_layer(vertex_data: &Vec<crate::mapbox::GlVert>, cmd: Vec<RenderCommand>, unlabeled: usize, labels: Vec<Label>) -> Self::Layer;
}

pub struct GL { }

impl Renderer for GL {
    type Layer = GlLayer;

    fn upload_layer(data: &Vec<crate::mapbox::GlVert>) -> GlLayer {
        return Self::upload_multi_layer(data, vec![RenderCommand::Simple(data.len())], 1, Vec::new());
    }

    fn upload_multi_layer(vertex_data: &Vec<crate::mapbox::GlVert>, cmd: Vec<RenderCommand>, unlabeled: usize, labels: Vec<Label>) -> Self::Layer {
        let mut vao = 0;
        let mut vbo = 0;

        unsafe {
            gl::GenVertexArrays(1, &mut vao);
            gl::GenBuffers(1, &mut vbo);

            gl::BindVertexArray(vao);

            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(gl::ARRAY_BUFFER, (vertex_data.len() * std::mem::size_of::<GlVert>()) as _, vertex_data.as_ptr().cast(), gl::STATIC_DRAW);

            gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<GlVert>() as i32, 0 as *const _);
            gl::EnableVertexAttribArray(0);
            gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<GlVert>() as i32, (2 * std::mem::size_of::<f32>()) as *const _);
            gl::EnableVertexAttribArray(1);
            gl::VertexAttribPointer(2, 1, gl::BYTE, gl::FALSE, std::mem::size_of::<GlVert>() as i32, (4 * std::mem::size_of::<f32>()) as *const _);
            gl::EnableVertexAttribArray(2);

            gl::BindBuffer(gl::ARRAY_BUFFER, 0);
            gl::BindVertexArray(0);
        }

        let mut blocks = Vec::with_capacity(labels.len() + 1);
        blocks.push(unlabeled);
        for l in &labels {
            blocks.push(l.cmds);
        }

        return GlLayer{
            vao,
            vbo,
            unlabeled,
            commands: cmd,
            blocks,
            labels
        };
    }
}

pub const LAYERTYPE_MAX: usize = 11;
#[derive(Clone, Copy)]
pub enum LayerType {
    EARTH,
    ROADS,
    HIGHWAYS,
    MAJOR,
    MEDIUM,
    MINOR,
    BUILDINGS,
    WATER,
    FARMLAND,
    AREAS,
    POINTS,
}

pub struct FaderSlot {
    pub key: String,
}

impl FaderSlot {
    pub fn new(key: String) -> Self {
        return FaderSlot {
            key,
        }
    }
}

pub struct Tile<R: Renderer> {
    pub tid: u64,
    pub x: u64,
    pub y: u64,
    pub z: u8,
    pub extent: u16,
    pub fades: Vec<FaderSlot>,

    pub layers: [Option<R::Layer>; LAYERTYPE_MAX],
}

pub struct LineBuilder {
    pub verts: Vec<GlVert>,
}

impl <'a> LineBuilder {
    pub fn new() -> Self {
        return LineBuilder{
            verts: Vec::new(),
        };
    }

    pub fn add_point(&mut self, lv: Vector2<f32>, v1: Vector2<f32>, connect_previous: bool) {
        let cx = lv.x;
        let cy = lv.y;

        let mut ltov = v1.clone();
        ltov -= lv;

        let mut normal = ltov.clone();
        normal.normal();
        normal.unit();

        let bend_norm_x;
        let bend_norm_y;

        if connect_previous {
            let len = self.verts.len();

            let last_normx = self.verts[len-2].norm_x;
            let last_normy = self.verts[len-2].norm_y;

            let mut join_x = last_normx + normal.x;
            let mut join_y = last_normy + normal.y;
            let join_len = f32::sqrt(f32::powi(join_x, 2) + f32::powi(join_y, 2));
            join_x /= join_len;
            join_y /= join_len;

            let cos_angle = normal.x * join_x + normal.y * join_y;
            let l = 1.0 / cos_angle;
            // Don't do a miter for very sharp corners
            if l < 2.0 {
                bend_norm_x = join_x * l;
                bend_norm_y = join_y * l;

                self.verts[len-4].norm_x = bend_norm_x;
                self.verts[len-4].norm_y = bend_norm_y;
                self.verts[len-3].norm_x = -bend_norm_x;
                self.verts[len-3].norm_y = -bend_norm_y;
                self.verts[len-2].norm_x = bend_norm_x;
                self.verts[len-2].norm_y = bend_norm_y;
            } else {
                // @HACK @COMPLETE: Do another type  of join here. Right now it's just disconnected
                bend_norm_x = normal.x;
                bend_norm_y = normal.y;
            }
        } else {
            bend_norm_x = normal.x;
            bend_norm_y = normal.y;
        }

        // Now construct the tris
        self.verts.push(GlVert { x:   cx, y:   cy, norm_x:  bend_norm_x, norm_y:  bend_norm_y, sign: 1 });
        self.verts.push(GlVert { x:   cx, y:   cy, norm_x: -bend_norm_x, norm_y: -bend_norm_y, sign: -1 });
        self.verts.push(GlVert { x: v1.x, y: v1.y, norm_x:  normal.x, norm_y:  normal.y, sign: 1 });

        self.verts.push(GlVert { x: v1.x, y: v1.y, norm_x: -normal.x, norm_y: -normal.y, sign: -1 });
        self.verts.push(GlVert { x: v1.x, y: v1.y, norm_x:  normal.x, norm_y:  normal.y, sign: 1 });
        self.verts.push(GlVert { x:   cx, y:   cy, norm_x: -bend_norm_x, norm_y: -bend_norm_y, sign: -1 });
    }
}


fn polygon_area_two(polys: &[crate::mapbox::GlVert]) -> f32 {
    let start = 0;
    let end = polys.len();

    let mut area = 0.0;
    for i in start..end-1 {
        area += polys[i].x*polys[i+1].y - polys[i].y*polys[i+1].x;
    }
    area += polys[end-1].x*polys[start].y - polys[end-1].y*polys[start].x;

    return area;
}

fn compile_polygon_layer<R: Renderer>(raw_tile: &crate::mapbox::PolyGeom, _z: u8) -> R::Layer {
    let poly_start = &raw_tile.start;
    let polys = &raw_tile.data;
    let mut tri_polys = Vec::new();

    // @HACK: This normal calculation is bad and halfbaked. I'm leaving it in for now, but
    // the first sign of problems and i'm gutting it.
    let mut normals = Vec::with_capacity(polys.len());
    for i in 0..poly_start.len() {
        let start = poly_start[i].pos;
        let end = if i < poly_start.len()-1 {
            poly_start[i+1].pos
        } else {
            polys.len()
        };

        for j in start..end {
            let prev = if j == start {
                polys[end-1]
            } else {
                polys[j-1]
            };
            let current = polys[j];
            let next = if j == end-1 {
                polys[start]
            } else {
                polys[j+1]
            };

            let prev = Vector2::new(prev.x, prev.y);
            let current = Vector2::new(current.x, current.y);
            let mut next = Vector2::new(next.x, next.y);

            let mut into = current.clone();
            into -= prev;

            next -= current;

            into.unit();
            next.unit();

            into.normal();
            next.normal();

            into += next;
            into.unit();

            let cos_angle = into.dot(&next);
            let l = 1.0 / cos_angle;

            into *= l;

            normals.push(into);
        }
    }
    assert!(normals.len() == polys.len());

    let mut exterior_rings = Vec::with_capacity(poly_start.len());
    // Find the clockwise polygons
    if !poly_start.is_empty() {
        for i in 0..poly_start.len() {
            let start = poly_start[i].pos;
            let end = if i < poly_start.len()-1 {
                poly_start[i+1].pos
            } else {
                polys.len()
            };

            let area = polygon_area_two(&polys[start..end]);
            if area.is_sign_positive() {
                exterior_rings.push(i);
            }

            // I don't entirely understand why we don't need to reverse the clockwise polys here,
            // but the output _looks_ fine.
        }
    }

    let mut offset = 0;
    for i in 0..exterior_rings.len() {
        let ring = exterior_rings[i];
        let polys_start = poly_start[ring].pos;

        let polys_end = if i < exterior_rings.len()-1 {
            // Everything between here and the next exterior polygon is part of this polygon
            poly_start[exterior_rings[i+1]].pos
        } else {
            // The final one contains the rest of the polys
            polys.len()
        };

        let point_it = polys[polys_start..polys_end].iter()
            .map(|i| (i.x as f64, i.y as f64));
        let normals = &normals[polys_start..polys_end];

        let (tris, vid) = triangulate::triangulate(poly_start.iter().filter(|x| x.pos >= offset).map(|x| x.pos-offset), point_it, polys_end-polys_start).unwrap();
        let triangulation = tris
            .trim().unwrap();

        for tri in &triangulation.tris {
            let p1 = triangulation.verts[tri[0]];
            let p2 = triangulation.verts[tri[1]];
            let p3 = triangulation.verts[tri[2]];
            // @HACK: We know that verts are added in the order we pass them and that they
            // are never reordered. That's supposed to be an implementation detail, but
            // ends up important here
            let n1 = normals[vid[tri[0]].unwrap()];
            let n2 = normals[vid[tri[1]].unwrap()];
            let n3 = normals[vid[tri[2]].unwrap()];

            tri_polys.push(GlVert { x:   p1.x as f32, y:   p1.y as f32, norm_x: n1.x, norm_y: n1.y, sign: 0 });
            tri_polys.push(GlVert { x:   p2.x as f32, y:   p2.y as f32, norm_x: n2.x, norm_y: n2.y, sign: 0 });
            tri_polys.push(GlVert { x:   p3.x as f32, y:   p3.y as f32, norm_x: n3.x, norm_y: n3.y, sign: 0 });
        }
        offset += polys_end - polys_start;
    }

    return R::upload_layer(&tri_polys);
}

fn compile_region_layer<R: Renderer>(raw_tile: &crate::mapbox::PointGeom, strings: &Vec<String>, fades: &mut Vec<FaderSlot>, font: &mut FontMetric, font_scale: f32, _z: u8) -> R::Layer {

    assert!(raw_tile.name.iter().all(|x| x.is_some()));

    // Allocate unique fade handles for each unique point in the layer. The specification disallows
    // the strings table from containing values that are byte for byte identical, we can therefore
    // be sure that any points with the same name will have the same string id, and we can skip the
    // string lookup.
    let allocated_fades = {
        let mut index: Vec<usize> = (0..raw_tile.name.len()).collect();
        index.sort_by_key(|x| raw_tile.name[*x]);

        let mut allocated_fades = vec![0; raw_tile.name.len()];
        let mut current = None;
        for i in index {
            if current != Some(raw_tile.name[i]) {
                fades.push(FaderSlot::new(strings[raw_tile.name[i].unwrap()].clone()));
                current = Some(raw_tile.name[i]);
            }

            allocated_fades[i] = fades.len()-1;
        }
        allocated_fades
    };

    let mut t = Transform::identity();
    t.scale(&Vector2::new(font_scale, font_scale));

    let mut verts = Vec::new();
    let mut labels = Vec::new();
    let mut draw_commands = Vec::new();
    for (i, v) in raw_tile.data.iter().enumerate() {
        let size_before = draw_commands.len();

        let width = font.width(&strings[raw_tile.name[i].unwrap()]);

        let mut t = t.clone();
        t.translate(&Vector2::new(-width/2.0, 0.0));

        let (min, max) = font.layout_text(&mut verts, &mut draw_commands, &strings[raw_tile.name[i].unwrap()], &t, *v);
        labels.push(Label{
            cmds: draw_commands.len() - size_before,
            min, max,
            rank: 0,
            pos: *v,
            not_before: 0.0,
            opacity: 0.0,
            opacity_from: Some(allocated_fades[i]),
        });
    }
    return R::upload_multi_layer(&verts, draw_commands, 0, labels);
}

fn compile_line_layer<R: Renderer>(raw_tile: &crate::mapbox::LineGeom, strings: &[String], font: &mut FontMetric, font_scale: f32, rank_start: u8) -> R::Layer {
    let mut line = LineBuilder::new();

    for i in 0..raw_tile.start.len() {
        let start = raw_tile.start[i].pos;
        let end = if i+1 < raw_tile.start.len() {
            raw_tile.start[i+1].pos
        } else {
            raw_tile.data.len()
        };

        for j in 0..end-start-1 {
            let p1 = raw_tile.data[start + j];
            let p2 = raw_tile.data[start + j+1];

            line.add_point(p1, p2, j!=0);
        }
    }
    let mut cmd = vec![RenderCommand::Simple(line.verts.len())];

    let mut labels = Vec::new();
    // Generate labels
    for i in 0..raw_tile.start.len() {
        let start = raw_tile.start[i].pos;
        let end = if i+1 < raw_tile.start.len() {
            raw_tile.start[i+1].pos
        } else {
            raw_tile.data.len()
        };

        // Calculate the length of the linestring
        let mut len = 0.0;
        for j in 0..end-start-1 {
            let p1 = raw_tile.data[start + j];
            let p2 = raw_tile.data[start + j+1];

            let v1 = Vector2 {
                x: p1.x as f64,
                y: p1.y as f64,
            };
            let mut v2 = Vector2 {
                x: p2.x as f64,
                y: p2.y as f64,
            };
            v2 -= v1;
            len += v2.len();
        }

        len /= 2.0;

        const DISTANCE_BETWEEN_LABELS: f64 = 100.0;
        let mut next = len % DISTANCE_BETWEEN_LABELS;
        let num_labels = ((len / DISTANCE_BETWEEN_LABELS) as u64) % 8;

        const RANK_SEQ: [u8; 8] = [3, 2, 3, 1, 3, 2, 3, 0];
        let mut rank_step = 7-num_labels as usize;

        let text = raw_tile.name[i];
        if text.is_none() {
            continue
        }
        let text = &strings[text.unwrap()];

        // Walk the line, placing labels as we go
        for j in 0..end-start-1 {
            let p1 = raw_tile.data[start + j];
            let p2 = raw_tile.data[start + j+1];

            let v1 = Vector2 {
                x: p1.x as f64,
                y: p1.y as f64,
            };
            let mut v2 = Vector2 {
                x: p2.x as f64,
                y: p2.y as f64,
            };
            v2 -= v1;

            let segment_len = v2.len();
            v2 /= segment_len;

            let mut orientation = v2.angle();
            // Rotate the labels if they would be upside down
            if orientation.abs() > std::f64::consts::TAU/4.0 {
                orientation -= orientation.signum() * std::f64::consts::TAU/2.0;
            }

            while next <= segment_len {
                assert!(next >= 0.0);

                let mut pos = v2.clone();
                pos *= next;
                pos += v1;

                let rank = RANK_SEQ[rank_step] + rank_start;
                // let text = rank.to_string();
                let width = font.width(&text);


                // @FIX @UX This basically stops us from placing labels on bendy bits. We
                // might want to allow hanging off the end if the bend isn't very sharp.
                 if next < (width*font_scale) as f64/2.0 {
                     // If there's not enough space for the label, push it in so that there
                     // is.
                     next = (width*font_scale) as f64/2.0;
                     continue;
                 } else if segment_len - next < (width*font_scale) as f64/2.0 {
                     // If there's not enough remaining space on the segment, push it into
                     // the next segment (plus half the width of the string to make sure
                     // there's room for it)
                     next = segment_len + (width*font_scale) as f64/2.0;
                     continue;
                 }

                let mut t = Transform::identity();
                t.rotate(orientation);
                t.scale(&Vector2::new(font_scale, font_scale));
                t.translate(&Vector2::new(-width/2.0, 0.0));

                let size_before = cmd.len();
                let (min, max) = font.layout_text(&mut line.verts, &mut cmd, &text, &t, pos.downcast());

                labels.push(Label{
                    rank: rank as u8,
                    min, max,
                    cmds: cmd.len() - size_before,
                    pos: pos.downcast(),
                    // @HACK: These are just some random numbers
                    not_before: 5000.0 + 10000.0 * 4.0_f32.powi(rank as _) as f32,
                    opacity: 0.0,
                    opacity_from: None,
                });

                rank_step = (rank_step+1) % RANK_SEQ.len();

                next += DISTANCE_BETWEEN_LABELS;
            }

            next -= segment_len;
        }
    }

    return R::upload_multi_layer(&line.verts, cmd, 1, labels);
}

pub fn compile_tile<R: Renderer>(id: u64, font: &mut FontMetric, x: u64, y: u64, z: u8, raw_tile: RawTile) -> Result<Tile<R>, String> {
    let font_size = 2.0_f32.powi(z as i32 - 15);

    let mut fades = Vec::new();

    let layers = [
        Some(compile_polygon_layer::<R>(&raw_tile.earth, z)),
        Some(compile_line_layer::<R>(&raw_tile.roads, &raw_tile.strings, font, font_size*0.25, 4)),
        Some(compile_line_layer::<R>(&raw_tile.highways, &raw_tile.strings, font, font_size*1.0, 0)),
        Some(compile_line_layer::<R>(&raw_tile.major, &raw_tile.strings, font, font_size*0.75, 1)),
        Some(compile_line_layer::<R>(&raw_tile.medium, &raw_tile.strings, font, font_size*0.75, 2)),
        Some(compile_line_layer::<R>(&raw_tile.minor, &raw_tile.strings, font, font_size*0.5, 2)),
        Some(compile_polygon_layer::<R>(&raw_tile.buildings, z)),
        Some(compile_polygon_layer::<R>(&raw_tile.water, z)),
        Some(compile_polygon_layer::<R>(&raw_tile.farmland, z)),
        Some(compile_polygon_layer::<R>(&raw_tile.areas, z)),

        Some(compile_region_layer::<R>(&raw_tile.points, &raw_tile.strings, &mut fades, font, font_size*4.0, z)),
    ];

    // @INCOMPLETE @CLEANUP: The extent here should be read from the file
    return Ok(Tile{
        tid: id,
        x, y, z,
        extent: 4096,
        fades,
        layers,
    });
}

