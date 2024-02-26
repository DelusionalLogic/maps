extern crate freetype;

use maps::font::FontMetric;
use maps::map::GL;
use maps::map::LayerType;
use maps::map::LineBuilder;
use maps::map::Tile;
use maps::mapbox::GlVert;
use maps::math::GLTransform;
use maps::math::Transform;
use maps::math::Vector2;
use maps::map::GlLayer;
use maps::map::RenderCommand;
use maps::math::lerp;
use maps::math::remap;
use maps::pmtile::PMTile;
use maps::triangulate;
use maps::label;
use maps::math::Mat4;
use maps::math::Mat3;
use maps::world::World;

use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::TryInto;
use std::ffi::CString;

use glfw;
use glfw::Context;
use gl;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;
const TITLE: &str = "Hello From OpenGL World!";

fn compile_shader(vert: &str, frag: &str) -> u32 {
    let vertex_shader = unsafe { gl::CreateShader(gl::VERTEX_SHADER) };
    unsafe {
        gl::ShaderSource(vertex_shader, 1, &vert.as_bytes().as_ptr().cast(), &vert.len().try_into().unwrap());
        gl::CompileShader(vertex_shader);

        let mut success = 0;
        gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success);
        if success == 0 {
            let mut log_len = 0_i32;
            // gl::GetShaderiv(vertex_shader, gl::INFO_LOG_LENGTH, &mut log_len);
            // let mut v: Vec<u8> = Vec::with_capacity(log_len as usize);
            // gl::GetShaderInfoLog(vertex_shader, log_len, &mut log_len, v.as_mut_ptr().cast());
            let mut v: Vec<u8> = Vec::with_capacity(1024);
            gl::GetShaderInfoLog(vertex_shader, 1024, &mut log_len, v.as_mut_ptr().cast());
            v.set_len(log_len.try_into().unwrap());
            panic!("Vertex Shader Compile Error: {}", String::from_utf8_lossy(&v));
        }
    }

    let fragment_shader = unsafe { gl::CreateShader(gl::FRAGMENT_SHADER) };
    unsafe {
        gl::ShaderSource(fragment_shader, 1, &frag.as_bytes().as_ptr().cast(), &frag.len().try_into().unwrap());
        gl::CompileShader(fragment_shader);

        let mut success = 0;
        gl::GetShaderiv(fragment_shader, gl::COMPILE_STATUS, &mut success);
        if success == 0 {
            let mut v: Vec<u8> = Vec::with_capacity(1024);
            let mut log_len = 0_i32;
            gl::GetShaderInfoLog(fragment_shader, 1024, &mut log_len, v.as_mut_ptr().cast());
            v.set_len(log_len.try_into().unwrap());
            panic!("Fragment Shader Compile Error: {}", String::from_utf8_lossy(&v));
        }
    }

    let program;
    unsafe {
        program = gl::CreateProgram();
        gl::AttachShader(program, vertex_shader);
        gl::AttachShader(program, fragment_shader);
        gl::LinkProgram(program);

        let mut success = 0;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
        if success == 0 {
            let mut v: Vec<u8> = Vec::with_capacity(1024);
            let mut log_len = 0_i32;
            gl::GetProgramInfoLog(program, 1024, &mut log_len, v.as_mut_ptr().cast());
            v.set_len(log_len.try_into().unwrap());
            panic!("Program Link Error: {}", String::from_utf8_lossy(&v));
        }

        gl::DetachShader(program, vertex_shader);
        gl::DetachShader(program, fragment_shader);
        gl::DeleteShader(vertex_shader);
        gl::DeleteShader(fragment_shader);
    }

    return program;
}

pub struct LineProg {
    program: u32,
    pre_transform: i32,
    transform: i32,
    width: i32,
    fill_color: i32,
}

pub struct FontProg {
    program: u32,
    texture: i32,
    transform: i32,
    target: i32,
    fill_color: i32,
    line_color: i32,
}

fn load_font() -> FontMetric {
    let freetype = freetype::Library::init().unwrap();

    let font_path = "OpenSans-Regular.ttf";

    let face = freetype.new_face(font_path, 0).unwrap();
    face.set_pixel_sizes(0, 32).unwrap();
    let font = FontMetric::load(face);
    return font;
}

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::Resizable(true));

    glfw.window_hint(glfw::WindowHint::X11ClassName(Some("maps".to_string())));
    glfw.window_hint(glfw::WindowHint::X11InstanceName(Some("maps".to_string())));
    let (mut window, events) = glfw.create_window(WIDTH, HEIGHT, TITLE, glfw::WindowMode::Windowed).unwrap();
    let mut display = {
        let (mouse_x, mouse_y) = window.get_cursor_pos();
        let (cx, cy) = window.get_content_scale();
        let (screen_width, screen_height) = window.get_framebuffer_size();

        DisplayState{
            width: screen_width,
            height: screen_height,
            size_change: true,
            mouse_x: mouse_x * cx as f64,
            mouse_y: mouse_y * cy as f64,
        }
    };

    window.make_current();
    window.set_key_polling(true);
    window.set_framebuffer_size_polling(true);
    window.set_cursor_pos_polling(true);
    window.set_mouse_button_polling(true);
    window.set_scroll_polling(true);
    gl::load_with(|ptr| window.get_proc_address(ptr) as *const _);

    unsafe {
        gl::Viewport(0, 0, display.width, display.height);
        let c = Color(0.4, 0.4, 0.4, 1.0);
        gl::ClearColor(c.0, c.1, c.2, c.3);
    }
    println!("OpenGL version: {}", gl_get_string(gl::VERSION));
    println!("GLSL version: {}", gl_get_string(gl::SHADING_LANGUAGE_VERSION));
    // -------------------------------------------

    let mut font = load_font();

    const FONT_VERT_SHADER: &str = "
        #version 330 core
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 uv_in;

        uniform mat4 transform;

        out vec2 uv;

        void main() {
            vec4 gp = transform * vec4(position, 0.0, 1.0);
            gl_Position = gp;
            uv = uv_in;
        }
    ";

    const FONT_FRAG_SHADER: &str = "
        #version 330 core
        out vec4 color;

        in vec2 uv;

        uniform sampler2D tex;
        uniform bool target;
        uniform vec4 fill_color;
        uniform vec4 line_color;

        const float smoothing = 1.0/16.0;
        const float mid = 0.5;

        void main() {
            if(target) {
                float dist = 1-2*length(uv - vec2(0.5));
                float alpha = smoothstep(0.5 - smoothing, 0.5 + smoothing, dist);
                color = vec4(vec3(alpha * 1.0), alpha);
            } else {
                float dist = texture(tex, uv).r;

                float border_alpha = smoothstep(mid-0.2 - smoothing, mid-0.2 + smoothing, dist);
                color = line_color*border_alpha;

                float alpha = smoothstep(mid - smoothing, mid + smoothing, dist);
                color = mix(color, fill_color, alpha);
            }
        }
    ";



    const VERT_SHADER: &str = "
        #version 330 core
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 normal;
        layout(location = 2) in float sign;
        out vec3 v_bary;
        out float v_dist;
        out vec2 norm;

        uniform mat4 pre_transform;
        uniform mat4 transform;
        uniform float tesselation_width;

        const vec3 barys[3] = vec3[](
            vec3(0, 0, 1),
            vec3(0, 1, 0),
            vec3(1, 0, 0)
        );

        void main() {
            v_bary = barys[gl_VertexID % 3];
            v_dist = sign;

            vec2 dnormal = normal; // * vec2(1.0, -1.0);
            // dnormal = clamp(dnormal, vec2(-1,-1), vec2(1, 1));
            norm = dnormal;
            vec4 gp = (transform * vec4(position, 0.0, 1.0)) + (pre_transform * vec4(dnormal*tesselation_width, 0.0, 0.0));
            gl_Position = gp;
        }
    ";

    const FRAG_SHADER: &str = "
        #version 330 core
        #extension GL_OES_standard_derivatives : enable
        out vec4 color;

        in vec3 v_bary;
        in float v_dist;
        in vec2 norm;

        uniform float tesselation_width;
        const vec4 border_color = vec4(.5, .5, .5, 1);

        uniform vec4 fill_color;

        const float feather = .7;
        const float line_width = 0.7;
        const float line_smooth = .5;
        vec4 edge_color = vec4(0, 0, 0, 1);

        float linearstep(float edge0, float edge1, float x) {
            return  clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
        }

        float edge_factor() {
            vec3 d = fwidth(v_bary);
            vec3 smth = line_smooth * d;
            d *= line_width;
            vec3 f = smoothstep(d, d+smth, v_bary);
            return min(min(f.x, f.y), f.z);
        }

        float center_factor() {
            float d = fwidth(v_dist);
            float smth = line_smooth * d;
            d *= line_width;
            float f = smoothstep(d, d+smth, abs(v_dist));
            return f;
        }

        void main() {
            color = vec4(1.0);
            float dist = abs(v_dist);

            color = fill_color;

            { // Border
                // float border = 2/tesselation_width;
                // float ffactor = linearstep(1-border-feather, 1-border, dist);
                // color = mix(color, border_color, ffactor);
            }

            float invdist = 1-dist;
            float grad = fwidth(invdist);
            float ffactor = clamp(invdist/(feather*grad), 0, 1);
            color *= ffactor;

            // color = mix(edge_color, color, edge_factor());
            // color = mix(vec4(1.0, 1.0, 1.0, 1.0), color, center_factor());
            // color = vec4(.9, .9, .9, 1.0);
        }
    ";

    let shader_program;
    {
        let program = compile_shader(VERT_SHADER, FRAG_SHADER);
        let pre_transform_str = CString::new("pre_transform").unwrap();
        let transform_str = CString::new("transform").unwrap();
        let width_str = CString::new("tesselation_width").unwrap();
        let fill_color_str = CString::new("fill_color").unwrap();
        unsafe {
            shader_program = LineProg{
                program,
                pre_transform: gl::GetUniformLocation(program, pre_transform_str.as_ptr()),
                transform: gl::GetUniformLocation(program, transform_str.as_ptr()),
                width: gl::GetUniformLocation(program, width_str.as_ptr()),
                fill_color: gl::GetUniformLocation(program, fill_color_str.as_ptr()),
            }
        }
    }

    let font_shader = {
        let program = compile_shader(FONT_VERT_SHADER, FONT_FRAG_SHADER);
        let texture = CString::new("tex").unwrap();
        let transform = CString::new("transform").unwrap();
        let target = CString::new("target").unwrap();
        let fill_color_str = CString::new("fill_color").unwrap();
        let line_color_str = CString::new("line_color").unwrap();
        unsafe {
            FontProg {
                program,
                texture: gl::GetUniformLocation(program, texture.as_ptr()),
                transform: gl::GetUniformLocation(program, transform.as_ptr()),
                target: gl::GetUniformLocation(program, target.as_ptr()),
                fill_color: gl::GetUniformLocation(program, fill_color_str.as_ptr()),
                line_color: gl::GetUniformLocation(program, line_color_str.as_ptr()),
            }
        }
    };

    let tris = triangulate::triangulate(
        [0].iter().copied(),
        [(100.0, 100.0), (100.0, 200.0), (200.0, 200.0), (200.0, 100.0)].iter().copied(),
        4,
    ).unwrap().0;

    let mut vertecies : Vec<f32> = Vec::with_capacity(5 * 3 * tris.tris.len());
    for tri in tris.tris {
        {
            let v = tris.verts[tri[0]];
            vertecies.push(v.x as _);
            vertecies.push(v.y as _);
            vertecies.push(0.0);

            vertecies.push(1.0);
            vertecies.push(0.0);
        }
        {
            let v = tris.verts[tri[1]];
            vertecies.push(v.x as _);
            vertecies.push(v.y as _);
            vertecies.push(0.0);

            vertecies.push(0.0);
            vertecies.push(1.0);
        }
        {
            let v = tris.verts[tri[2]];
            vertecies.push(v.x as _);
            vertecies.push(v.y as _);
            vertecies.push(0.0);

            vertecies.push(0.0);
            vertecies.push(0.0);
        }
    }

    let border: Tile<GL> = {
        let mut line = LineBuilder::new();

        let mut lv = Vector2::new(0.0, 0.0);

        {
            let nv = Vector2::new(0.0, 4096.0);
            line.add_point(lv, nv, false);
            lv = nv;
        }

        {
            let nv = Vector2::new(4096.0, 4096.0);
            line.add_point(lv, nv, true);
            lv = nv;
        }

        {
            let nv = Vector2::new(4096.0, 0.0);
            line.add_point(lv, nv, true);
            lv = nv;
        }

        {
            let nv = Vector2::new(0.0, 0.0);
            line.add_point(lv, nv, true);
        }

        let mut vao = 0;
        unsafe { gl::GenVertexArrays(1, &mut vao) };

        let mut vbo = 0;
        unsafe { gl::GenBuffers(1, &mut vbo) };

        unsafe {
            gl::BindVertexArray(vao);

            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(gl::ARRAY_BUFFER, (line.verts.len() * std::mem::size_of::<GlVert>()) as _, line.verts.as_ptr().cast(), gl::STATIC_DRAW);

            gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<GlVert>() as i32, 0 as *const _);
            gl::EnableVertexAttribArray(0);
            gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<GlVert>() as i32, (2 * std::mem::size_of::<f32>()) as *const _);
            gl::EnableVertexAttribArray(1);
            gl::VertexAttribPointer(2, 1, gl::BYTE, gl::FALSE, std::mem::size_of::<GlVert>() as i32, (4 * std::mem::size_of::<f32>()) as *const _);
            gl::EnableVertexAttribArray(2);

            gl::BindBuffer(gl::ARRAY_BUFFER, 0);
            gl::BindVertexArray(0);
        }

        let mut layers: [Option<GlLayer>; 11] = Default::default();
        layers[LayerType::ROADS as usize] = Some(GlLayer{
            vao,
            vbo,
            unlabeled: 1,
            commands: vec![RenderCommand::Simple(line.verts.len())],
            blocks: vec![1],
            labels: Vec::new(),
        });

        Tile{
            tid: 0,
            x: 0, y: 0, z: 0,
            extent: 4096,
            fades: Vec::new(),
            layers,
        }
    };

    // -------------------------------------------


    let overlay_layers = [
        (LayerType::POINTS, true),
        (LayerType::ROADS, false),
        (LayerType::MINOR, false),
        (LayerType::MEDIUM, false),
        (LayerType::MAJOR, false),
        (LayerType::HIGHWAYS, false),
    ];

    struct LabelData {
        opacity: f32,
        goal: f32,
    }
    let mut fader_binds = HashMap::new();

    let mut projection = Transform::from_mat(Mat4::ortho(0.0, display.width as _, display.height as _, 0.0, 0.0, 1.0));
    let mut hold: Option<Vector2<f64>> = None;
    let mut zoom = 0.0;
    let mut scale = 1.0;
    let mut viewport_pos = Vector2::new(0.0_f64, 0.0);
    let mut viewport_size;
    let mut mouse_world = Vector2::new(0.0, 0.0);
    const MAP_SIZE : u64 = 512;
    let mut tiles = World::new(PMTile::new("aalborg.pmtiles"));
    while !window.should_close() {
        glfw.poll_events();

        display.size_change = false;

        let mut dzoom = 0.0;
        for (_, event) in glfw::flush_messages(&events) {
            glfw_handle_event(&mut window, event, &mut display, &mut dzoom);
        }

        if dzoom != 0.0 {
            zoom += dzoom;
            zoom = zoom.clamp(0.0, 100.0);

            let old_scale = scale;
            scale = f64::powf(2.0, zoom.into());

            let v = (scale-old_scale) / (scale*old_scale);
            viewport_pos.x += display.mouse_x * v;
            viewport_pos.y += display.mouse_y * v;
        }

        let scale_level = zoom.floor() as u64;

        {
            // Calculate the mouse position in the world
            mouse_world.x = display.mouse_x as f64;
            mouse_world.y = display.mouse_y as f64;
            mouse_world /= scale as f64;
            mouse_world += viewport_pos;
        }

        if display.size_change {
            projection = Transform::from_mat(Mat4::ortho(0.0, display.width as _, display.height as _, 0.0, 0.0, 1.0));
        }
        viewport_size = Vector2::new(display.width as _, display.height as _);
        viewport_size /= scale;

        if window.get_mouse_button(glfw::MouseButtonLeft) == glfw::Action::Press {
            if let Some(hold) = hold {
                let mut diff = hold.clone();
                diff -= mouse_world;
                viewport_pos += diff;
                mouse_world = hold;
            } else {
                hold = Some(mouse_world);
            }
        } else {
            hold = None;
        }

        let (to_add, _to_remove) = {
            let scale_level = scale_level.min(tiles.source.max_zoom as u64);
            let native_resolution = MAP_SIZE as f64 / 2.0_f64.powi(scale_level as i32);
            let top_left = {
                let mut v = viewport_pos.clone();
                v /= native_resolution;
                v.floor();
                v.quant()
            };
            let bottom_right: Vector2<u64> = {
                let mut v = viewport_pos.clone();
                v += viewport_size;
                v /= native_resolution;
                v.ceil();
                v.quant()
            };

            tiles.update_visible_tiles(top_left.x, top_left.y, bottom_right.x, bottom_right.y, scale_level as u8, &mut font)
        };

        {
            let needs_bind : HashSet<String> = to_add.iter().flat_map(|tid| {
                tiles.get(*tid).unwrap().fades.iter().map(|x| &x.key).cloned()
            }).collect();


            for key in needs_bind {
                if !fader_binds.contains_key(&key) {
                    fader_binds.insert(key, LabelData{ opacity: 0.0, goal: 0.0 });
                }
            }
        }

        let mut world_to_screen = projection.clone();

        world_to_screen.scale(&Vector2::new(scale as f64, scale as f64));
        world_to_screen.translate(&Vector2::new(-viewport_pos.x, -viewport_pos.y));

        unsafe {
            let c = Color(0.0, 0.1, 0.15, 1.0);
            gl::ClearColor(c.0, c.1, c.2, c.3);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        unsafe {
            gl::Enable(gl::BLEND);
            gl::BlendEquationSeparate(gl::FUNC_ADD, gl::MAX);
            gl::BlendFunc(gl::ONE, gl::ONE_MINUS_SRC_ALPHA);
            gl::UseProgram(shader_program.program);
        }

        fn run_commands(shader_program: &LineProg, font_shader: &FontProg, projection: Option<&GLTransform>, tile_transform: &Transform, layer: u32, commands: &[RenderCommand], color: &Color, width: f32, font: &mut FontMetric, inverse_scale: f32, offset: usize) -> usize {
            let mut cursor = offset;

            unsafe {
                gl::BindVertexArray(layer);

                #[derive(PartialEq)]
                enum RenderMode {
                    UNINIT,
                    SIMPLE,
                    TEXT,
                }

                impl RenderMode {
                    fn go_to(&mut self, new: RenderMode, shader_program: &LineProg, font_shader: &FontProg, projection32: Option<&GLTransform>, tile_transform32: &Transform, color: &Color, width: f32) {
                        match new {
                            _ if *self == new => {}
                            RenderMode::UNINIT => panic!("You can't switch to uninit"),
                            RenderMode::SIMPLE => {
                                unsafe {
                                    gl::UseProgram(shader_program.program);

                                    if let Some(projection32) = projection32 {
                                        gl::UniformMatrix4fv(shader_program.pre_transform, 1, gl::TRUE, projection32.as_ptr());
                                    }
                                    gl::UniformMatrix4fv(shader_program.transform, 1, gl::TRUE, tile_transform32.to_gl().as_ptr());
                                    gl::Uniform1f(shader_program.width, width);
                                    let Color(r, g, b, a) = color;
                                    gl::Uniform4f(shader_program.fill_color, *r, *g, *b, *a);
                                }
                            },
                            RenderMode::TEXT => {
                                unsafe {
                                    gl::BlendFunc(gl::ONE, gl::ONE_MINUS_SRC_ALPHA);
                                    gl::Enable(gl::BLEND);

                                    gl::ActiveTexture(gl::TEXTURE0);

                                    gl::UseProgram(font_shader.program);
                                    gl::Uniform1i(font_shader.texture, 0);
                                    let Color(r, g, b, a) = color;
                                    gl::Uniform4f(font_shader.fill_color, *r, *g, *b, *a);
                                    gl::Uniform4f(font_shader.line_color, 0.0 * a, 0.2 * a, 0.3 * a, *a);
                                }
                            },
                        }

                        *self = new;
                    }
                }

                let mut mode = RenderMode::UNINIT;
                for cmd in commands {
                    let x = match cmd {
                        RenderCommand::Simple(x) => {
                            mode.go_to(RenderMode::SIMPLE, shader_program, font_shader, projection, tile_transform, color, width);
                            if mode != RenderMode::SIMPLE {

                                mode = RenderMode::SIMPLE;
                            }

                            x
                        }
                        RenderCommand::Target(t, x) => {
                            mode.go_to(RenderMode::TEXT, shader_program, font_shader, projection, tile_transform, color, width);

                            let mut trans = tile_transform.clone();
                            trans.translate(t);
                            trans.scale(&Vector2::new(inverse_scale, inverse_scale));

                            gl::UniformMatrix4fv(font_shader.transform, 1, gl::TRUE, trans.to_gl().as_ptr());

                            gl::Uniform1i(font_shader.target, 1);

                            x
                        }
                        RenderCommand::PositionedLetter(c, t, x) => {
                            mode.go_to(RenderMode::TEXT, shader_program, font_shader, projection, tile_transform, color, width);

                            let mut trans = tile_transform.clone();
                            trans.translate(t);
                            trans.scale(&Vector2::new(inverse_scale, inverse_scale));

                            gl::UniformMatrix4fv(font_shader.transform, 1, gl::TRUE, trans.to_gl().as_ptr());

                            gl::Uniform1i(font_shader.target, 0);

                            let texture = &font.size_char(*c as usize);
                            gl::BindTexture(gl::TEXTURE_2D, texture.texdata.texture.atlas.gl_texture);

                            x
                        }
                    };

                    gl::DrawArrays(gl::TRIANGLES, cursor as _, *x as _);
                    cursor += x;
                }

                gl::BindVertexArray(0);
            }

            return cursor - offset;
        }

        fn render_poly(shader_program: &LineProg, font_shader: &FontProg, projection32: Option<&GLTransform>, tile_transform: &Transform, layer: &Option<GlLayer>, color: &Color, width: f32, font: &mut FontMetric, inverse_scale: f32) {
            if let Some(layer) = layer {
                run_commands(shader_program, font_shader, projection32, tile_transform, layer.vao, &layer.commands[0..layer.blocks[0]], color, width, font, inverse_scale, 0);
            }
        }

        let mut draw_stuff = Vec::new();
        let mut labels = Vec::new();
        let mut boxes = Vec::with_capacity(labels.len());

        let mut vao_offsets = Vec::with_capacity(tiles.visible.len());
        let mut offsets = Vec::with_capacity(tiles.visible.len());
        let mut tile_to_screen_transforms = Vec::with_capacity(tiles.visible.len());

        let inverse_scale = 1.0/scale as f32 * 2.0_f32.powi(15);
        let text_zoom_scale = if zoom >= 15.5 {
            remap(15.5, 18.5, 6.0, 1.0, zoom)
        } else {
            remap(14.5, 15.5, 8.0, 6.0, zoom)
        };

        for (local_tid, tid) in tiles.visible.iter().enumerate() {
            let tile = tiles.get(*tid).unwrap();

            let grid_step = MAP_SIZE as f64 / 2.0_f64.powi(tile.z as i32);

            let tile_to_world = {
                let xcoord = tile.x as f64 * grid_step;
                let ycoord = tile.y as f64 * grid_step;

                let mut transform = Transform::identity();
                transform.translate(&Vector2::new(xcoord, ycoord));
                transform.scale(&Vector2::new(grid_step/tile.extent as f64, grid_step/tile.extent as f64));

                transform
            };
            // Here we can truncate

            let tile_to_screen = {
                let mut t = world_to_screen.clone();
                t.apply(tile_to_world.mat());
                t
            };

            tile_to_screen_transforms.push(tile_to_screen.clone());

            // Calculate the screen position of the tile and scissor that
            {
                let mut v1 = Vector2::new(0.0, tile.extent as f32);
                let mut v2 = Vector2::new(tile.extent as f32, 0.0);

                // The clipspace transform
                let mvp3d: &Mat4 = (&tile_to_screen).into();
                let mvp2d: Mat3 = mvp3d.into();
                let mvp2d32: [f32; 9] = mvp2d.into();

                v1.apply_transform(&mvp2d32);
                v2.apply_transform(&mvp2d32);

                // The viewport transform
                v1 += 1.0;
                v1 /= 2.0;
                v2 += 1.0;
                v2 /= 2.0;

                let display_size = Vector2::new(display.width as f32, display.height as f32);
                v1 *= display_size;
                v2 *= display_size;

                v2 -= v1;

                unsafe {
                    gl::Enable(gl::SCISSOR_TEST);
                    // Allow a one pixel overlap between tiles to reduce the screen door when
                    // rounding goes poorly
                    gl::Scissor(v1.x as i32, v1.y as i32, v2.x as i32 +1, v2.y as i32 +1);
                }
            }

            unsafe {
                gl::Clear(gl::COLOR_BUFFER_BIT);
            }

            let gl_proj = {
                let mut projection = projection.clone();
                projection.scale(&Vector2::new(scale, scale));
                projection.to_gl()
            };

            render_poly(&shader_program, &font_shader, None, &tile_to_screen, &tile.layers[LayerType::EARTH as usize], &Color(0.1, 0.3, 0.4, 1.0), 0.0, &mut font, 0.0);
            render_poly(&shader_program, &font_shader, None, &tile_to_screen, &tile.layers[LayerType::AREAS as usize], &Color(0.07, 0.27, 0.37, 1.0), 0.0, &mut font, 0.0);
            render_poly(&shader_program, &font_shader, None, &tile_to_screen, &tile.layers[LayerType::FARMLAND as usize], &Color(0.07, 0.27, 0.37, 1.0), 0.0, &mut font, 0.0);
            render_poly(&shader_program, &font_shader, None, &tile_to_screen, &tile.layers[LayerType::BUILDINGS as usize], &Color(0.0, 0.2, 0.3, 1.0), 0.0, &mut font, 0.0);
            render_poly(&shader_program, &font_shader, None, &tile_to_screen, &tile.layers[LayerType::WATER as usize], &Color(0.082, 0.173, 0.267, 1.0), 0.0, &mut font, 0.0);

            {
                let road_layers = [
                    (LayerType::ROADS, Color(0.024, 0.118, 0.173, 1.0), Color(0.176, 0.247, 0.298, 1.0), 0.00001),
                    (LayerType::MINOR, Color(0.024, 0.118, 0.173, 1.0), Color(0.075, 0.196, 0.263, 1.0), 0.00004),
                    (LayerType::MEDIUM, Color(0.024, 0.118, 0.173, 1.0), Color(0.075, 0.196, 0.263, 1.0), 0.00007),
                    (LayerType::MAJOR, Color(0.024, 0.118, 0.173, 1.0), Color(0.075, 0.196, 0.263, 1.0), 0.00009),
                    (LayerType::HIGHWAYS, Color(0.024, 0.118, 0.173, 1.0), Color(0.075, 0.196, 0.263, 1.0), 0.00020),
                ];

                for (layer, bgcolor, _, width)  in &road_layers {
                    let outline_width = 2.0/scale;
                    render_poly(&shader_program, &font_shader, Some(&gl_proj), &tile_to_screen, &tile.layers[*layer as usize], bgcolor, *width + outline_width as f32, &mut font, 0.0);
                }

                for (layer, _, fgcolor, width)  in &road_layers {
                    render_poly(&shader_program, &font_shader, Some(&gl_proj), &tile_to_screen, &tile.layers[*layer as usize], fgcolor, *width, &mut font, 0.0);
                }
            }

            unsafe {
                gl::Disable(gl::SCISSOR_TEST);
            }

            {
                let mut tile_vao_offsets = vec![0; overlay_layers.len()];
                let mut tile_offsets = vec![0; overlay_layers.len()];

                for (layer_id, (layer_type, screen_relative)) in overlay_layers.into_iter().enumerate() {
                    if let Some(layer) = &tile.layers[layer_type as usize] {
                        tile_offsets[layer_id] = layer.blocks[0];
                        // @HACK @PERF Calculate the offset for the first conditional triangle.
                        // This shouldn't really be done here, we should remember this from when we
                        // rendered the unconditional stuff
                        tile_vao_offsets[layer_id] = {
                        let mut vao_offset = 0;
                            for cmd in &layer.commands[0..layer.blocks[0]] {
                                let count = match cmd {
                                    RenderCommand::Simple(x) => x,
                                    RenderCommand::Target(_, x) => x,
                                    RenderCommand::PositionedLetter(_, _, x) => x,
                                };
                                vao_offset += count;
                            }
                            vao_offset
                        };

                        let scale_factor = if screen_relative {
                            inverse_scale
                        } else {
                            text_zoom_scale
                        };

                        for (label_id, label) in layer.labels.iter().enumerate() {
                            // @HACK we place it in the middle of the baseline, but the middle of
                            // the baseline is defined as the centerpoint between the start pen
                            // location and the right side of the bounding box (which is not
                            // necessarily the ending pen location). It might look better to use
                            // the middle of the bounding box in the x direction.
                            labels.push(label);
                            draw_stuff.push((local_tid, layer_id, label_id, screen_relative));

                            // The clipspace transform
                            let mvp3d: &Mat4 = (&tile_to_world).into();
                            let mvp2d: Mat3 = mvp3d.into();
                            let mvp2d32: [f32; 9] = mvp2d.into();

                            let mut min = label.min;
                            let mut max = label.max;
                            min *= scale_factor;
                            max *= scale_factor;
                            min += label.pos;
                            max += label.pos;
                            min.apply_transform(&mvp2d32);
                            max.apply_transform(&mvp2d32);

                            boxes.push(label::Box {
                                min, max
                            });
                        }
                    }
                }
                offsets.push(tile_offsets);
                vao_offsets.push(tile_vao_offsets);
            }

            // {
            //     let mut text_transform = tile_trans.clone();
            //     text_transform.scale(&Vector2::new(4.0, 4.0));
            //     text_transform.translate(&Vector2::new(10.0, 32.0));
            //     draw_ascii(text_transform.clone(), &mut font, format!("X {} Y {} Z {}", tile.x, tile.y, tile.z).as_bytes());
            //     text_transform.translate(&Vector2::new(0.0, 32.0));
            //     draw_ascii(text_transform.clone(), &mut font, format!("TID {} ID {}", tile.tid, mapbox::pmtile::coords_to_id(tile.x, tile.y, tile.z)).as_bytes());
            // }
        }

        {
            // For debugging label culling
            if false {
                for (i, _label) in labels.iter().enumerate() {
                    let bbox = &boxes[i];

                    let mut size = bbox.max.clone();
                    size -= bbox.min;

                    let mut entity_to_screen = world_to_screen.clone();
                    entity_to_screen.translate(&bbox.min);

                    // @HACK This is just because the border layer is a hack as well.
                    entity_to_screen.scale(&Vector2::new(size.x as f32/border.extent as f32, size.y as f32/border.extent as f32));
                    let gl_proj = projection.to_gl();
                    render_poly(&shader_program, &font_shader, Some(&gl_proj), &entity_to_screen, &border.layers[LayerType::ROADS as usize], &Color(0.0, 1.0, 1.0, 1.0), 1.0, &mut font, 0.0);
                }
            }

            let mut desired_visible: Vec<usize> = (0..boxes.len()).collect();

            // Discard labels that are too small
            desired_visible.retain(|i| {
                (labels[*i].not_before as f64) <= scale
            });

            // Cull offscreen labels
            {
                let top_left = viewport_pos;
                let bottom_right = {
                    let mut v = viewport_pos.clone();
                    v += viewport_size;
                    v
                };

                desired_visible.retain(|i| {
                    let bbox = &boxes[*i];

                    bbox.min.x <= bottom_right.x as f32 && bbox.max.x >= top_left.x as f32
                        && bbox.min.y <= bottom_right.y as f32 && bbox.max.y >= top_left.y as f32
                });
            }

            desired_visible.sort_by_key(|i| labels[*i].rank);

            // Select a set that doesn't overlap
            label::select_nooverlap(&boxes, &mut desired_visible);

            desired_visible.sort();

            for (_, x) in fader_binds.iter_mut() {
                x.goal = 0.0;
            }

            let mut visible_cursor = 0;
            let num_labels = labels.len();
            // Set the end goal of all the fades
            // @HACK This also steps the unique fades, that's kinda silly. Maybe we should create
            // fades for those as well somehow?
            for i in 0..num_labels {
                assert!(visible_cursor >= desired_visible.len() || desired_visible[visible_cursor] >= i);

                let (local_tile_id, layer_id, label_id, _) = &draw_stuff[i];
                let tile = tiles.get_mut(tiles.visible[*local_tile_id]).unwrap();
                let layer = tile.layers[overlay_layers[*layer_id].0 as usize].as_mut().unwrap();
                let label = &mut layer.labels[*label_id];

                let target_opacity = if visible_cursor < desired_visible.len() && desired_visible[visible_cursor] == i {
                    visible_cursor += 1;
                    1.0
                } else {
                    0.0
                };

                if let Some(x) = &label.opacity_from {
                    let label_data = fader_binds.get_mut(&tile.fades[*x].key).unwrap();
                    label_data.goal = label_data.goal.max(target_opacity);
                } else {
                    label.opacity = lerp(label.opacity, target_opacity, 0.1);
                }
            }

            // Step the shared fades
            for (_, x) in fader_binds.iter_mut() {
                x.opacity = lerp(x.opacity, x.goal, 0.1);
            }

            // Copy the opacity back into the labels
            // @PERF This seems unnecessary
            for i in 0..num_labels {
                let (local_tile_id, layer_id, label_id, _) = &draw_stuff[i];
                let tile = tiles.get_mut(tiles.visible[*local_tile_id]).unwrap();
                let layer = tile.layers[overlay_layers[*layer_id].0 as usize].as_mut().unwrap();
                let label = &mut layer.labels[*label_id];

                if let Some(x) = &label.opacity_from {
                    let label_data = fader_binds.get_mut(&tile.fades[*x].key).unwrap();
                    label.opacity = label_data.opacity;
                }
            }

            // Do the render
            for i in 0..num_labels {
                let (local_tile_id, layer_id, label_id, screenspace_scale) = &draw_stuff[i];

                let scale_factor = if *screenspace_scale {
                    inverse_scale
                } else {
                    text_zoom_scale
                };

                let tile = tiles.get_mut(tiles.visible[*local_tile_id]).unwrap();
                let layer = tile.layers[overlay_layers[*layer_id].0 as usize].as_mut().unwrap();
                let commands = &layer.commands;
                let vao = &layer.vao;
                let offset = &mut offsets[*local_tile_id][*layer_id];
                let vao_offset = &mut vao_offsets[*local_tile_id][*layer_id];
                let label = &mut layer.labels[*label_id];

                let executed = if label.opacity > f32::EPSILON {
                    run_commands(
                        &shader_program,
                        &font_shader,
                        None,
                        &tile_to_screen_transforms[*local_tile_id],
                        *vao,
                        &commands[*offset..*offset+label.cmds],
                        &Color(label.opacity, label.opacity, label.opacity, label.opacity), 0.0,
                        &mut font,
                        scale_factor,
                        *vao_offset
                    )
                } else {
                    let mut executed = 0;
                    for cmd in &commands[*offset..*offset+label.cmds] {
                        executed += match cmd {
                            RenderCommand::Simple(x) => x,
                            RenderCommand::Target(_, x) => x,
                            RenderCommand::PositionedLetter(_, _, x) => x,
                        }
                    }

                    executed
                };

                *vao_offset += executed;
                *offset += label.cmds;
            }
        }

        // Draw the pois
        for tid in &tiles.visible {
            let tile = tiles.get(*tid).unwrap();
            let mut tile_trans = world_to_screen.clone();

            let scale_factor = 2.0_f64.powi(tile.z as i32 - 15) as f32 * (1.0/scale as f32)*10.0 * 36000.0;
            let grid_step = MAP_SIZE as f64 / 2.0_f64.powi(tile.z as i32);

            let tile_trans = {
                let xcoord = tile.x as f64 * grid_step;
                let ycoord = tile.y as f64 * grid_step;
                // Transform the tilelocal coordinates to global coordinates
                tile_trans.translate(&Vector2::new(xcoord, ycoord));
                tile_trans.scale(&Vector2::new(grid_step/tile.extent as f64, grid_step/tile.extent as f64));

                tile_trans
            };

            render_poly(&shader_program, &font_shader, None, &tile_trans, &tile.layers[LayerType::POINTS as usize], &Color(1.0, 1.0, 1.0, 1.0), 0.0, &mut font, scale_factor);
        }

        // let mut text_transform = projection.clone();
        // text_transform.translate(&Vector2::new(20.0, 20.0));
        // draw_ascii(text_transform.clone(), &mut font, format!("Pos {} {}, size {} {}", viewport_pos.x, viewport_pos.y, display.width as f64/scale, display.height as f64/scale).as_bytes());
        // text_transform.translate(&Vector2::new(0.0, 16.0));
        // draw_ascii(text_transform.clone(), &mut font, format!("ggg Pos {} {}", mouse_world.x, mouse_world.y).as_bytes(), );
        // text_transform.translate(&Vector2::new(0.0, 16.0));
        // draw_ascii(text_transform.clone(), &mut font, format!("Zoom level {} {}", scale, scale_level).as_bytes());

        window.swap_buffers();
    }
}

pub struct Color(f32, f32, f32, f32);

struct DisplayState {
    size_change: bool,
    width: i32,
    height: i32,
    mouse_x: f64,
    mouse_y: f64,
}

pub fn gl_get_string<'a>(name: gl::types::GLenum) -> &'a str {
    let v = unsafe { gl::GetString(name) };
    let v: &std::ffi::CStr = unsafe { std::ffi::CStr::from_ptr(v as *const i8) };
    v.to_str().unwrap()
}

fn glfw_handle_event(window: &mut glfw::Window, event: glfw::WindowEvent, state: &mut DisplayState, zoom: &mut f32) {
    use glfw::WindowEvent as Event;
    use glfw::Key;
    use glfw::Action;

    match event {
        Event::FramebufferSize(w, h) => {
            unsafe{ gl::Viewport(0, 0, w, h) };
            state.size_change = true;
            state.width = w;
            state.height = h;
        },
        Event::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true);
        },
        Event::Key(Key::Q, _, Action::Press, _) => {
            window.set_should_close(true);
        },
        Event::CursorPos(x, y) => {
            let (sx, sy) = window.get_content_scale();
            state.mouse_x = x * sx as f64;
            state.mouse_y = y * sy as f64;
        },
        Event::Scroll(_, y) => {
            *zoom += y as f32 * 0.1;
            // view_pos.y -= state.mouse_y as f32 * zoom_diff;
            // dbg!("X {} Y {} RECI {}", state.mouse_x, state.mouse_y, reci, view_pos.x, view_pos.y, *zoom);
        },
        _ => {},
    }
}
