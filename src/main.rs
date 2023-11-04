mod triangulate;
mod math;
mod mapbox;
mod label;
extern crate freetype;

use math::Vector2;
use math::Mat4;
use math::Mat3;

use std::convert::TryInto;
use std::ffi::CString;

use glfw;
use glfw::Context;
use gl;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;
const TITLE: &str = "Hello From OpenGL World!";

type GLTransform = [f32; 16];

struct Transform {
    mats : [Mat4; 2],
    primary: bool,
}

impl Transform {
    pub fn from_mat(source: Mat4) -> Self {
        return Transform{
            mats: [source, math::MAT4_IDENTITY],
            primary: false,
        };
    }

    pub fn identity() -> Self {
        return Self::from_mat(math::MAT4_IDENTITY);
    }

    fn split(&self) -> (usize, usize) {
        let primary = if self.primary { 1 } else { 0 };
        let secondary = if self.primary { 0 } else { 1 };

        return (primary, secondary);
    }

    fn apply(&mut self, op: &Mat4) {
        let (primary, secondary) = self.split();

        // self.mats[secondary] = op.mul(&self.mats[primary]);
        self.mats[secondary] = self.mats[primary].mul(op);
        self.primary = !self.primary;
    }

    pub fn translate<T: Into<f64> + Copy>(&mut self, offset: &Vector2<T>) {
        self.apply(&Mat4::translate(offset.x.into(), offset.y.into()));
    }

    pub fn rotate<T: Into<f64>>(&mut self, theta: T) {
        self.apply(&Mat4::rotate_2d(theta.into()));
    }

    pub fn scale<T: Into<f64> + Copy>(&mut self, scale: &Vector2<T>) {
        self.apply(&Mat4::scale_2d(scale.x.into(), scale.y.into()));
    }

    pub fn mat(&self) -> &Mat4 {
        let (primary, _) = self.split();
        return &self.mats[primary];
    }

    pub fn to_gl(&self) -> GLTransform {
        let (primary, _) = self.split();
        return (&self.mats[primary]).into();
    }
}

impl<'a> Into<&'a Mat4> for &'a Transform {
    fn into(self) -> &'a Mat4 {
        let (primary, _) = self.split();
        return &self.mats[primary];
    }
}

impl Clone for Transform {
    fn clone(&self) -> Self {
        let (primary, _) = self.split();
        return Transform::from_mat(self.mats[primary].clone());
    }
}

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

struct Character {
    size: Vector2<f32>,
    bearing: Vector2<f32>,
    advance: f32,
    texture: u32,
}

struct FontMap {
    chars: Vec<Character>,

    shader: u32,
    uni_transform: i32,
    uni_texture: i32,

    vao: u32,
    vbo: u32,
}

impl Drop for FontMap {
    fn drop(&mut self) {
        let mut textures = Vec::with_capacity(self.chars.len());
        for char in &self.chars {
            textures.push(char.texture);
        }
        unsafe{ gl::DeleteTextures(textures.len().try_into().unwrap(), textures.as_ptr()) };

        unsafe{ gl::DeleteProgram(self.shader) };

        unsafe{ gl::DeleteVertexArrays(1, &self.vao) };
        unsafe{ gl::DeleteBuffers(1, &self.vbo) };
    }
}

fn load_font() -> FontMap {
    let freetype = freetype::Library::init().unwrap();
    let face = freetype.new_face("/home/delusional/Documents/neocomp/assets/Roboto-Light.ttf", 0).unwrap();

    let mut chars = Vec::with_capacity(128);

    let mut textures = [0; 128];
    unsafe{ gl::GenTextures(textures.len() as i32, textures.as_mut_ptr()) };
    for i in 0..128 {
        unsafe{
            gl::BindTexture(gl::TEXTURE_2D, textures[i]);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        }
    }

    face.set_pixel_sizes(0, 16).unwrap();
    unsafe{ gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1) }
    for i in 0..128 {
        face.load_char(i, freetype::face::LoadFlag::RENDER).unwrap();
        let glyph = face.glyph();
        let bitmap = glyph.bitmap();
        let size = Vector2::new(bitmap.width() as f32, bitmap.rows() as f32);
        let bearing = Vector2::new(
            glyph.bitmap_left() as f32,
            -glyph.bitmap_top() as f32,
        );
        let advance = (glyph.advance().x >> 6) as f32;

        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, textures[i]);
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

        chars.push(Character {
            size,
            bearing,
            advance,
            texture: textures[i],
        });
    }

    const TEXT_VERT: &str = "
        #version 330 core
        layout(location = 0) in vec2 position;

        uniform mat4 transform;

        out vec2 uv;

        void main() {
            uv = position;
            uv.y = uv.y;

            gl_Position = transform * vec4(position, 0.0, 1.0);
        }
    ";

    const TEXT_FRAG: &str = "
        #version 330 core
        in vec2 uv;
        uniform sampler2D tex;

        out vec4 color;

        void main() {
            vec4 tex = texture(tex, uv);
            color = vec4(tex.r);
        }
    ";

    let program = compile_shader(TEXT_VERT, TEXT_FRAG);
    let uni_texture;
    let uni_transform;
    {
        let name = CString::new("tex").unwrap();
        uni_texture = unsafe{ gl::GetUniformLocation(program, name.as_ptr()) };
        let name = CString::new("transform").unwrap();
        uni_transform = unsafe{ gl::GetUniformLocation(program, name.as_ptr()) };
    }

    let verts = [
        0.0 as f32, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ];

    let mut vao = 0;
    unsafe { gl::GenVertexArrays(1, &mut vao) };

    let mut vbo = 0;
    unsafe { gl::GenBuffers(1, &mut vbo) };
    unsafe {
        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(gl::ARRAY_BUFFER, (verts.len() * std::mem::size_of::<f32>()) as _, verts.as_ptr().cast(), gl::STATIC_DRAW);

        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, 2 * std::mem::size_of::<f32>() as i32, 0 as *const _);
        gl::EnableVertexAttribArray(0);

        gl::BindBuffer(gl::ARRAY_BUFFER, 0);
        gl::BindVertexArray(0);
    }

    return FontMap{
        chars,
        shader: program,
        uni_texture,
        uni_transform,

        vao,
        vbo,
    };
}

fn size_ascii(font: &FontMap, text: &[u8]) -> (Vector2<f32>, Vector2<f32>) {
    let mut advance = 0.0;
    let mut max = Vector2::new(0.0, 0.0);
    let mut min = Vector2::new(0.0, 0.0);

    for c in text {
        if *c >= 128 { continue; }
        let char = &font.chars[*c as usize];

        let mut char_min = Vector2::new(advance as f32, 0.0);
        char_min.addv2(&char.bearing);

        let mut char_max = char_min.clone();
        char_max.addv2(&char.size);

        min.min(&char_min);
        max.max(&char_max);
        advance += char.advance;
    }

    return (min, max);
}

fn draw_ascii(projection: Transform, font: &FontMap, text: &[u8]) {
    unsafe {
        gl::BlendFunc(gl::ONE, gl::ONE_MINUS_SRC_ALPHA);
        gl::Enable(gl::BLEND);

        gl::ActiveTexture(gl::TEXTURE0);

        gl::UseProgram(font.shader);
        gl::Uniform1i(font.uni_texture, 0);
    }

    let mut pen = Vector2::new(0.0, 0.0);
    for c in text {
        if *c >= 128 { continue; }
        let char = &font.chars[*c as usize];

        let mut pos = pen.clone();
        pos.addv2(&char.bearing);

        let mut projection = projection.clone();
        projection.translate(&pos);
        projection.scale(&char.size);
        let mvp32: [f32; 16] = projection.to_gl();

        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, char.texture);
            gl::UniformMatrix4fv(font.uni_transform, 1, gl::TRUE, mvp32.as_ptr());
            gl::BindVertexArray(font.vao);
            gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);
        }

        pen.x += char.advance;
    }
}

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::Resizable(true));

    let mut display;
    let (mut window, events) = glfw.create_window(WIDTH, HEIGHT, TITLE, glfw::WindowMode::Windowed).unwrap();
    {
        let (mouse_x, mouse_y) = window.get_cursor_pos();
        let (cx, cy) = window.get_content_scale();
        let (screen_width, screen_height) = window.get_framebuffer_size();
        display = DisplayState{
            width: screen_width,
            height: screen_height,
            size_change: true,
            mouse_x: mouse_x * cx as f64,
            mouse_y: mouse_y * cy as f64,
        }
    }

    window.make_current();
    window.set_key_polling(true);
    window.set_framebuffer_size_polling(true);
    window.set_cursor_pos_polling(true);
    window.set_mouse_button_polling(true);
    window.set_scroll_polling(true);
    gl::load_with(|ptr| window.get_proc_address(ptr) as *const _);

    unsafe {
        gl::Viewport(0, 0, display.width, display.height);
        clear_color(Color(0.4, 0.4, 0.4, 1.0));
    }
    println!("OpenGL version: {}", gl_get_string(gl::VERSION));
    println!("GLSL version: {}", gl_get_string(gl::SHADING_LANGUAGE_VERSION));
    // -------------------------------------------

    let font = load_font();

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

    // unsafe {
    //     gl::BindVertexArray(vao);

    //     gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
    //     gl::BufferData(gl::ARRAY_BUFFER, (vertecies.len() * std::mem::size_of::<f32>()) as _, vertecies.as_ptr().cast(), gl::STATIC_DRAW);

    //     gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 5 * std::mem::size_of::<f32>() as i32, 0 as *const _);
    //     gl::EnableVertexAttribArray(0);
    //     gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, 5 * std::mem::size_of::<f32>() as i32, (3 * std::mem::size_of::<f32>()) as *const _);
    //     gl::EnableVertexAttribArray(1);

    //     gl::BindBuffer(gl::ARRAY_BUFFER, 0);
    //     gl::BindVertexArray(0);
    // }
    // let file = std::fs::File::open("aalborg.mvt").unwrap();
    // let tile = compile_tile(0, 0, 0, file).unwrap();
    let border = mapbox::pmtile::placeholder_tile(0, 0, 0);

    // -------------------------------------------

    let mut projection = Transform::from_mat(Mat4::ortho(0.0, display.width as _, display.height as _, 0.0, 0.0, 1.0));
    let mut hold: Option<Vector2<f64>> = None;
    let mut zoom = 0.0;
    let mut scale = 1.0;
    let mut viewport_pos = Vector2::new(0.0_f64, 0.0);
    let mut mouse_world = Vector2::new(0.0, 0.0);
    const MAP_SIZE : u64 = 512;
    let mut tiles = mapbox::pmtile::LiveTiles::new(mapbox::pmtile::File::new("aalborg.pmtiles"));
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

        let scale_level = (f64::log2(scale).floor() as u64).min(tiles.source.max_zoom as u64);

        {
            // Calculate the mouse position in the world
            mouse_world.x = display.mouse_x as f64;
            mouse_world.y = display.mouse_y as f64;
            mouse_world.divf(scale as f64);
            mouse_world.addv2(&viewport_pos);
        }

        if display.size_change {
            projection = Transform::from_mat(Mat4::ortho(0.0, display.width as _, display.height as _, 0.0, 0.0, 1.0));
        }

        if window.get_mouse_button(glfw::MouseButtonLeft) == glfw::Action::Press {
            if let Some(hold) = hold {
                let mut diff = hold.clone();
                diff.subv2(&mouse_world);
                viewport_pos.addv2(&diff);
                mouse_world = hold;
            } else {
                hold = Some(mouse_world);
            }
        } else {
            hold = None;
        }

        {
            let native_resolution = MAP_SIZE as f64 / 2.0_f64.powi(scale_level as i32);
            let left = ((viewport_pos.x / native_resolution).floor() as i64).max(0) as u64;
            let top = ((viewport_pos.y / native_resolution).floor() as i64).max(0) as u64;
            let right = (((viewport_pos.x + (display.width as f64/scale as f64)) / native_resolution).ceil() as i64).max(0) as u64;
            let bottom = (((viewport_pos.y + (display.height as f64/scale as f64)) / native_resolution).ceil() as i64).max(0) as u64;

            tiles.retrieve_visible_tiles(left, top, right, bottom, scale_level as u8);
        }


        let mut transform = projection.clone();

        transform.scale(&Vector2::new(scale as f64, scale as f64));
        transform.translate(&Vector2::new(-viewport_pos.x, -viewport_pos.y));

        clear_color(Color(0.0, 0.1, 0.15, 1.0));

        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        unsafe {
            gl::Enable(gl::BLEND);
            gl::BlendEquationSeparate(gl::FUNC_ADD, gl::MAX);
            gl::BlendFunc(gl::ONE, gl::ONE_MINUS_SRC_ALPHA);
            gl::UseProgram(shader_program.program);
        }

        for tid in &tiles.visible {
            let mut tile_trans = transform.clone();
            let tile = tiles.active.get(tid).unwrap();

            let grid_step = MAP_SIZE as f64 / 2.0_f64.powi(tile.z as i32);

            let xcoord = tile.x as f64 * grid_step;
            let ycoord = tile.y as f64 * grid_step;
            // Transform the tilelocal coordinates to global coordinates
            tile_trans.translate(&Vector2::new(xcoord, ycoord));
            tile_trans.scale(&Vector2::new(grid_step/tile.extent as f64, grid_step/tile.extent as f64));
            // Here we can truncate


            // Calculate the screen position of the tile and scissor that
            {
                let mut v1 = Vector2::new(0.0, tile.extent as f32);
                let mut v2 = Vector2::new(tile.extent as f32, 0.0);

                // The clipspace transform
                let mvp3d: &Mat4 = (&tile_trans).into();
                let mvp2d: Mat3 = mvp3d.into();
                let mvp2d32: [f32; 9] = mvp2d.into();
                v1.apply_transform(&mvp2d32);
                v2.apply_transform(&mvp2d32);

                // The viewport transform
                v1.addf(1.0);
                v1.divf(2.0);
                v2.addf(1.0);
                v2.divf(2.0);

                let display_size = Vector2::new(display.width as f32, display.height as f32);
                v1.mulv2(&display_size);
                v2.mulv2(&display_size);

                v2.subv2(&v1);

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

            fn render_poly(shader_program: &LineProg, projection32: Option<&GLTransform>, tile_transform32: &GLTransform, layer: &Option<mapbox::pmtile::Layer>, color: &Color, width: f32) {
                if let Some(layer) = layer {
                    unsafe {
                        gl::UseProgram(shader_program.program);

                        if let Some(projection32) = projection32 {
                            gl::UniformMatrix4fv(shader_program.pre_transform, 1, gl::TRUE, projection32.as_ptr());
                        }
                        gl::UniformMatrix4fv(shader_program.transform, 1, gl::TRUE, tile_transform32.as_ptr());
                        gl::BindVertexArray(layer.vao);

                        gl::Uniform1f(shader_program.width, width);
                        let Color(r, g, b, a) = color;
                        gl::Uniform4f(shader_program.fill_color, *r, *g, *b, *a);
                        gl::DrawArrays(gl::TRIANGLES, 0, layer.size as _);

                        gl::BindVertexArray(0);
                    }
                }
            }

            let (gl_proj, gl_trans) = {
                let mut projection = projection.clone();
                projection.scale(&Vector2::new(scale, scale));
                (projection.to_gl(), tile_trans.to_gl())
            };

            render_poly(&shader_program, None, &gl_trans, &tile.layers.earth, &Color(0.1, 0.3, 0.4, 1.0), 0.0);
            render_poly(&shader_program, None, &gl_trans, &tile.layers.areas, &Color(0.07, 0.27, 0.37, 1.0), 0.0);
            render_poly(&shader_program, None, &gl_trans, &tile.layers.farmland, &Color(0.07, 0.27, 0.37, 1.0), 0.0);
            render_poly(&shader_program, None, &gl_trans, &tile.layers.buildings, &Color(0.0, 0.2, 0.3, 1.0), 0.0);
            render_poly(&shader_program, None, &gl_trans, &tile.layers.water, &Color(0.082, 0.173, 0.267, 1.0), 0.0);

            {
                let road_layers = [
                    (&tile.layers.roads, Color(0.024, 0.118, 0.173, 1.0), Color(0.75, 0.196, 0.263, 1.0), 0.00003),
                    (&tile.layers.minor, Color(0.024, 0.118, 0.173, 1.0), Color(0.075, 0.196, 0.263, 1.0), 0.00004),
                    (&tile.layers.medium, Color(0.024, 0.118, 0.173, 1.0), Color(0.075, 0.196, 0.263, 1.0), 0.00007),
                    (&tile.layers.major, Color(0.024, 0.118, 0.173, 1.0), Color(0.075, 0.196, 0.263, 1.0), 0.00009),
                    (&tile.layers.highways, Color(0.024, 0.118, 0.173, 1.0), Color(0.075, 0.196, 0.263, 1.0), 0.00020),
                ];

                for (layer, bgcolor, _, width)  in &road_layers {
                    let outline_width = 1.0/scale;
                    render_poly(&shader_program, Some(&gl_proj), &gl_trans, &layer, bgcolor, *width + outline_width as f32);
                }

                for (layer, _, fgcolor, width)  in &road_layers {
                    render_poly(&shader_program, Some(&gl_proj), &gl_trans, &layer, fgcolor, *width);
                }
            }

            let mut labels = Vec::new();
            let mut bbox = Vec::new();
            let mut size = Vec::new();

            if let Some(roads) = &tile.layers.highways {
                for label in &roads.labels {
                    let (mut min, mut max) = size_ascii(&font, label.text.as_bytes());

                    // Transform to fit the destination size
                    // @SPEED This is fixed per tile. We could precompute it
                    min.mulf(300.0);
                    min.divf(grid_step as f32*tile.extent as f32);
                    max.mulf(300.0);
                    max.divf(grid_step as f32*tile.extent as f32);

                    // @HACK we place it in the middle of the baseline, but the middle of the
                    // baselines is defined as the centerpoint between the start pen location and
                    // the right side of the bounding box (which is not necessarily the ending pen
                    // location). It might look better to use the middle of the bounding box in the
                    // x direction.
                    bbox.push((min, max));
                    labels.push(label);
                    size.push(300.0);
                }
            }

            if let Some(roads) = &tile.layers.roads {
                for label in &roads.labels {
                    let (mut min, mut max) = size_ascii(&font, label.text.as_bytes());

                    // Transform to fit the destination size
                    // @SPEED This is fixed per tile. We could precompute it
                    min.mulf(60.0);
                    min.divf(grid_step as f32*tile.extent as f32);
                    max.mulf(60.0);
                    max.divf(grid_step as f32*tile.extent as f32);

                    // @HACK we place it in the middle of the baseline, but the middle of the
                    // baselines is defined as the centerpoint between the start pen location and
                    // the right side of the bounding box (which is not necessarily the ending pen
                    // location). It might look better to use the middle of the bounding box in the
                    // x direction.
                    bbox.push((min, max));
                    labels.push(label);
                    size.push(60.0);
                }
            }


            let mut boxes = Vec::with_capacity(labels.len());
            // Calculate the axis aligned bouding box of each label
            for (i, label) in labels.iter().enumerate() {
                let (min, max) = bbox[i];

                let mut max = max.clone();
                max.subv2(&min);

                let mut size = Vector2::new(
                    max.x as f64 * label.orientation.cos().abs() + max.y as f64 * label.orientation.sin().abs(),
                    max.x as f64 * label.orientation.sin().abs() + max.y as f64 * label.orientation.cos().abs(),
                );
                size.divf(2.0);

                let mut min = label.pos.clone();
                min.subv2(&size);
                let mut max = label.pos.clone();
                max.addv2(&size);

                boxes.push(label::Box {
                    min, max
                });
            }

            // And select a set that doesn't overlap
            let to_draw = label::select_nooverlap(&boxes);

            for i in to_draw {
                let label = labels[i];
                let size = size[i];
                let (min, max) = bbox[i];

                // Remove labels that overlap a tile boundary
                {
                    let mut max = max.clone();
                    max.subv2(&min);

                    let awidth = max.x as f64 * label.orientation.cos().abs() + max.y as f64 * label.orientation.sin().abs();
                    let aheight = max.x as f64 * label.orientation.sin().abs() + max.y as f64 * label.orientation.cos().abs();

                    if label.pos.x < awidth/2.0 || label.pos.x > 4096.0-awidth/2.0 {
                        continue;
                    }
                    if label.pos.y < aheight/2.0 || label.pos.y > 4096.0-aheight/2.0 {
                        continue;
                    }
                }

                {
                    let mut text_transform = tile_trans.clone();
                    text_transform.translate(&Vector2::new(label.pos.x, label.pos.y));
                    text_transform.rotate(label.orientation);
                    text_transform.translate(&Vector2::new(-max.x / 2.0, -min.y / 2.0));
                    text_transform.scale(&Vector2::new(size/grid_step/tile.extent as f64, size/grid_step/tile.extent as f64));
                    draw_ascii(text_transform.clone(), &font, label.text.as_bytes());
                }
            }

            unsafe {
                gl::Disable(gl::SCISSOR_TEST);
            }

            // unsafe {
            //     gl::UseProgram(shader_program.program);

            //     gl::UniformMatrix4fv(shader_program.transform, 1, gl::TRUE, tile_transform32.as_ptr());
            //     gl::BindVertexArray(border.vao);

            //     gl::Uniform1f(shader_program.width, 10.0);
            //     gl::Uniform4f(shader_program.fill_color, 1.0, 0.0, 0.0, 1.0);
            //     gl::DrawArrays(gl::TRIANGLES, 0, border.vertex_len as _);

            //     gl::BindVertexArray(0);
            // }

            {
                let mut text_transform = tile_trans.clone();
                text_transform.scale(&Vector2::new(8.0, 8.0));
                {
                    let mut text_transform = text_transform.clone();
                    text_transform.translate(&Vector2::new(10.0, 16.0));
                    draw_ascii(text_transform.clone(), &font, format!("X {} Y {} Z {}", tile.x, tile.y, tile.z).as_bytes());
                }
                {
                    let mut text_transform = text_transform.clone();
                    text_transform.translate(&Vector2::new(10.0, 32.0));
                    draw_ascii(text_transform.clone(), &font, format!("TID {} ID {}", tile.tid, mapbox::pmtile::coords_to_id(tile.x, tile.y, tile.z)).as_bytes());
                }
            }
        }

        let mut text_transform = projection.clone();
        text_transform.translate(&Vector2::new(20.0, 20.0));
        draw_ascii(text_transform.clone(), &font, format!("Pos {} {}, size {} {}", viewport_pos.x, viewport_pos.y, display.width as f64/scale, display.height as f64/scale).as_bytes());
        text_transform.translate(&Vector2::new(0.0, 16.0));
        draw_ascii(text_transform.clone(), &font, format!("ggg Pos {} {}", mouse_world.x, mouse_world.y).as_bytes(), );
        text_transform.translate(&Vector2::new(0.0, 16.0));
        draw_ascii(text_transform.clone(), &font, format!("Zoom level {} {}", scale, scale_level).as_bytes());

        window.swap_buffers();
    }
}

pub struct Color(f32, f32, f32, f32);

pub fn clear_color(c: Color) {
    unsafe { gl::ClearColor(c.0, c.1, c.2, c.3) }
}

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
