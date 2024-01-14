use crate::font::FontMetric;
use crate::mapbox::GlVert;
use crate::mapbox::RawTile;
use crate::mapbox::pmtile::Label;
use crate::mapbox::pmtile::LineBuilder;
use crate::mapbox::pmtile::RenderCommand;
use crate::math::Transform;
use crate::math::Vector2;
use crate::mapbox::pmtile::Tile;
use crate::mapbox::pmtile::Renderer;

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

fn compile_polygon_layer<R: Renderer>(raw_tile: &mut crate::mapbox::PolyGeom, _z: u8) -> R::Layer {
    let poly_start = &raw_tile.start;
    let polys = &mut raw_tile.data;
    let mut tri_polys : Vec<GlVert> = vec![];

    // Triangulate poly
    use crate::triangulate;

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

    let mut multipoly_start = Vec::with_capacity(poly_start.len());
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
                multipoly_start.push(i);
            }

            polys[start..end].reverse();
            normals[start..end].reverse();
        }
    }

    let mut offset = 0;
    for i in 0..multipoly_start.len() {
        let multipoly = multipoly_start[i];
        let polys_start = poly_start[multipoly].pos;

        let polys_end = if i == multipoly_start.len()-1 {
            // The final multipoly extends to the end of the polys array
            polys.len()
        } else {
            poly_start[multipoly_start[i+1]].pos
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

fn compile_region_layer<R: Renderer>(raw_tile: &crate::mapbox::PointGeom, strings: &Vec<String>, font: &mut FontMetric, font_scale: f32, _z: u8) -> R::Layer {
    let mut verts = Vec::new();
    let mut labels = Vec::new();
    let mut draw_commands = Vec::new();
    let mut size_before = 0;
    for (i, v) in raw_tile.data.iter().enumerate() {

        // const SIZE : f32 = 5.0;
        // verts.push(super::GlVert { x: -SIZE, y: -SIZE, norm_x: 0.0, norm_y: 0.0, sign: 0 });
        // verts.push(super::GlVert { x: -SIZE, y:  SIZE, norm_x: 0.0, norm_y: 1.0, sign: 0 });
        // verts.push(super::GlVert { x:  SIZE, y: -SIZE, norm_x: 1.0, norm_y: 0.0, sign: 0 });
        // verts.push(super::GlVert { x:  SIZE, y: -SIZE, norm_x: 1.0, norm_y: 0.0, sign: 0 });
        // verts.push(super::GlVert { x: -SIZE, y:  SIZE, norm_x: 0.0, norm_y: 1.0, sign: 0 });
        // verts.push(super::GlVert { x:  SIZE, y:  SIZE, norm_x: 1.0, norm_y: 1.0, sign: 0 });
        // draw_commands.push(RenderCommand::Target(*v, 6));

        let width = font.width(&strings[raw_tile.name[i].unwrap()]);

        let mut t = Transform::identity();
        t.scale(&Vector2::new(font_scale, font_scale));
        t.translate(&Vector2::new(-width/2.0, 0.0));
        // t.translate(&Vector2::new(SIZE, 0.0));
        let (min, max) = font.layout_text(&mut verts, &mut draw_commands, &strings[raw_tile.name[i].unwrap()], &t, *v);
        labels.push(Label{
            cmds: draw_commands.len() - size_before,
            min, max,
            rank: 0,
            pos: *v,
            not_before: 0.0,
        });
        size_before = draw_commands.len();
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

            line.add_point(p1, p2, j!=0, true);
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

        // Calculate the length of the line
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

        const RANK_SEQ: [u8; 8] = [0, 3, 2, 3, 1, 3, 2, 3];
        const RANK_STEPS : usize = RANK_SEQ.len();
        let mut rank_step = 0;

        let text = raw_tile.name[i];
        if text.is_none() {
            continue
        }
        let text = text.unwrap();

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

                let width = font.width(&strings[text]);

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

                let rank = RANK_SEQ[rank_step] + rank_start;

                let size_before = cmd.len();
                let (min, max) = font.layout_text(&mut line.verts, &mut cmd, &strings[text], &t, pos.downcast());

                // line.verts.push(super::GlVert{x: min.x, y: min.y, norm_x: 0.0, norm_y: 0.0, sign: 0});
                // line.verts.push(super::GlVert{x: min.x, y: max.y, norm_x: 0.0, norm_y: 1.0, sign: 0});
                // line.verts.push(super::GlVert{x: max.x, y: min.y, norm_x: 1.0, norm_y: 0.0, sign: 0});
                // line.verts.push(super::GlVert{x: max.x, y: min.y, norm_x: 1.0, norm_y: 0.0, sign: 0});
                // line.verts.push(super::GlVert{x: min.x, y: max.y, norm_x: 0.0, norm_y: 1.0, sign: 0});
                // line.verts.push(super::GlVert{x: max.x, y: max.y, norm_x: 1.0, norm_y: 1.0, sign: 0});
                // cmd.push(RenderCommand::PositionedLetter('x', Vector2::new(0.0, 0.0), 6));

                labels.push(Label{
                    rank: rank as u8,
                    min, max,
                    cmds: cmd.len() - size_before,
                    pos: pos.downcast(),
                    // @HACK: These are just some random numbers
                    not_before: 30000.0 + 20000.0 * 2.0_f32.powi(rank as _) as f32,
                });

                rank_step = (rank_step+1) % RANK_STEPS;

                next += DISTANCE_BETWEEN_LABELS;
            }

            next -= segment_len;
        }
    }

    return R::upload_multi_layer(&line.verts, cmd, 1, labels);
}

pub fn compile_tile<R: Renderer>(id: u64, font: &mut FontMetric, x: u64, y: u64, z: u8, mut raw_tile: RawTile) -> Result<Tile<R>, String> {
    let font_size = 2.0_f32.powi(z as i32 - 15);

    let layers = [
        Some(compile_polygon_layer::<R>(&mut raw_tile.earth, z)),
        Some(compile_line_layer::<R>(&raw_tile.roads, &raw_tile.strings, font, font_size*0.25, 4)),
        Some(compile_line_layer::<R>(&raw_tile.highways, &raw_tile.strings, font, font_size*4.0, 0)),
        Some(compile_line_layer::<R>(&raw_tile.major, &raw_tile.strings, font, font_size*1.0, 1)),
        Some(compile_line_layer::<R>(&raw_tile.medium, &raw_tile.strings, font, font_size*1.0, 2)),
        Some(compile_line_layer::<R>(&raw_tile.minor, &raw_tile.strings, font, font_size*0.5, 2)),
        Some(compile_polygon_layer::<R>(&mut raw_tile.buildings, z)),
        Some(compile_polygon_layer::<R>(&mut raw_tile.water, z)),
        Some(compile_polygon_layer::<R>(&mut raw_tile.farmland, z)),
        Some(compile_polygon_layer::<R>(&mut raw_tile.areas, z)),

        Some(compile_region_layer::<R>(&raw_tile.points, &raw_tile.strings, font, font_size*4.0, z)),
    ];

    // @INCOMPLETE @CLEANUP: The extent here should be read from the file
    return Ok(Tile{
        tid: id,
        x, y, z,
        extent: 4096,
        layers,
    });
}

