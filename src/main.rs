mod triangulate;
mod mapbox;

extern crate freetype;

use triangulate::triangulate;
use std::convert::TryInto;
use std::ffi::CString;

use glfw;
use glfw::Context;
use gl;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;
const TITLE: &str = "Hello From OpenGL World!";

#[derive(Debug, Clone, Copy)]
struct Vector2 {
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

fn ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [f32; 16] {
    return [
        2.0/(right-left),              0.0,            0.0, -(right+left)/(right-left),
                     0.0, 2.0/(top-bottom),            0.0, -(top+bottom)/(top-bottom),
                     0.0,              0.0, 2.0/(far-near), -(far+near  )/(far-near  ),
                     0.0,              0.0,            1.0,                          1.0,
    ];
}

fn mat4_multply(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0; 16];

    for row in 0..4 {
        let row_offset = row * 4;
        for column in 0..4 {
            out[row_offset + column] =
                (b[row_offset + 0] * a[column + 0]) +
                (b[row_offset + 1] * a[column + 4]) +
                (b[row_offset + 2] * a[column + 8]) +
                (b[row_offset + 3] * a[column + 12]);
        }
    }

    return out;
}

fn mat4_scale(factor: f32) -> [f32; 16] {
    return [
            factor,        0.0,        0.0, 0.0,
               0.0,     factor,        0.0, 0.0,
               0.0,        0.0,     factor, 0.0,
               0.0,        0.0,        0.0, 1.0,
    ];
}

fn mat4_translate(x: f32, y: f32) -> [f32; 16] {
    return [
           1.0,    0.0,    0.0,   x,
           0.0,    1.0,    0.0,   y,
           0.0,    0.0,    1.0, 0.0,
           0.0,    0.0,    0.0, 1.0,
    ];
}

fn mat4_to_mat3(mat: &[f32; 16]) -> [f32; 9] {
    return [
        mat[0], mat[1], mat[3],
        mat[4], mat[5], mat[7],
        mat[12], mat[13], mat[15],
    ]
}

const IDENTITY: [f32; 16] = [
           1.0,    0.0,    0.0, 0.0,
           0.0,    1.0,    0.0, 0.0,
           0.0,    0.0,    1.0, 0.0,
           0.0,    0.0,    0.0, 1.0,
];

pub struct LineProg {
    program: u32,
    transform: i32,
    width: i32,
    fill_color: i32,
}

struct Character {
    size: Vector2,
    bearing: Vector2,
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
    let face = freetype.new_face("/System/Library/Fonts/Supplemental/Baskerville.ttc", 0).unwrap();

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

    unsafe{ gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1) }
    face.set_pixel_sizes(0, 12).unwrap();
    for i in 0..128 {
        face.load_char(i, freetype::face::LoadFlag::RENDER).unwrap();
        let bitmap = face.glyph().bitmap();
        let size = Vector2::new(bitmap.width() as f32, bitmap.rows() as f32);
        let bearing = Vector2::new(
            face.glyph().bitmap_left() as f32,
            face.glyph().bitmap_top() as f32,
        );
        let advance = (face.glyph().advance().x >> 6) as f32;

        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, textures[i]);
            gl::TexStorage2D(
                gl::TEXTURE_2D, 0,
                gl::RED,
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
            gl_Position = transform * vec4(position, 0.0, 1.0);
        }
    ";

    const TEXT_FRAG: &str = "
        #version 330 core
        in vec2 uv;

        out vec4 color;

        void main() {
            color = vec4(1.0, 0.0, 0.0, 1.0);
        }
    ";

    let program = compile_shader(TEXT_VERT, TEXT_FRAG);
    let uni_texture;
    let uni_transform;
    {
        let name = CString::new("texture").unwrap();
        uni_texture = unsafe{ gl::GetUniformLocation(program, name.as_ptr()) };
        let name = CString::new("transform").unwrap();
        uni_transform = unsafe{ gl::GetUniformLocation(program, name.as_ptr()) };
    }

    let verts = [
        0.0, 0.0,
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

fn draw_text(projection: &[f32;16], font: &FontMap, text: &[u8], pos: &Vector2) {
    unsafe {
        gl::Enable(gl::BLEND);

        gl::UseProgram(font.shader);
        gl::Uniform1i(font.uni_texture, 0);

        gl::ActiveTexture(gl::TEXTURE0);
    }

    let mut pen = *pos;
    for c in text {
        let char = &font.chars[*c as usize];

        let mut pos = pen;
        pos.addv2(&char.bearing);
        pos.y -= char.size.y;

        let mvp = mat4_multply(&mat4_multply(&mat4_translate(pos.x, pos.y), &mat4_scale(200.0)), projection);

        unsafe {
            gl::BindTexture(gl::TEXTURE_2D, char.texture);
            gl::UniformMatrix4fv(font.uni_transform, 1, gl::TRUE, mvp.as_ptr());
            gl::BindVertexArray(font.vao);
            gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);
        }
    }
}

struct Tile {
    x: u64,
    y: u64,
    z: u8,
    extent: u16,
    vao: u32,
    vbo: u32,
    vertex_len: usize,
}

impl Drop for Tile {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteVertexArrays(1, &self.vao);
            gl::DeleteBuffers(1, &self.vbo);
        }
    }
}

fn compile_tile(x: u64, y: u64, z: u8, mut file: std::fs::File) -> Result<Tile, String> {
    let (start, verts) = mapbox::read_one_linestring(&mut mapbox::pbuf::Message::new(&mut file)).unwrap();

    let mut vao = 0;
    unsafe { gl::GenVertexArrays(1, &mut vao) };

    let mut vbo = 0;
    unsafe { gl::GenBuffers(1, &mut vbo) };

    unsafe {
        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(gl::ARRAY_BUFFER, (verts.len() * std::mem::size_of::<mapbox::LineVert>()) as _, verts.as_ptr().cast(), gl::STATIC_DRAW);

        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<mapbox::LineVert>() as i32, 0 as *const _);
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<mapbox::LineVert>() as i32, (2 * std::mem::size_of::<f32>()) as *const _);
        gl::EnableVertexAttribArray(1);
        gl::VertexAttribPointer(2, 1, gl::BYTE, gl::FALSE, std::mem::size_of::<mapbox::LineVert>() as i32, (4 * std::mem::size_of::<f32>()) as *const _);
        gl::EnableVertexAttribArray(2);

        gl::BindBuffer(gl::ARRAY_BUFFER, 0);
        gl::BindVertexArray(0);
    }

    // @INCOMPLETE @CLEANUP: The extent here should be read from the file
    return Ok(Tile{
        x,
        y,
        z,
        extent: 4096,
        vao,
        vbo,
        vertex_len: verts.len(),
    });
}

fn add_point(verts: &mut Vec<mapbox::LineVert>, lv: Vector2, v1: Vector2, connect_previous: bool) {
    let cx = lv.x;
    let cy = lv.y;

    let mut ltov = v1.clone();
    ltov.subv2(&lv);

    let mut normal = ltov.clone();
    normal.normal();
    normal.unit();

    let bend_norm_x;
    let bend_norm_y;

    if connect_previous {
        let len = verts.len();

        let last_normx = verts[len-2].norm_x;
        let last_normy = verts[len-2].norm_y;

        let mut join_x = last_normx + normal.x;
        let mut join_y = last_normy + normal.y;
        let join_len = f32::sqrt(f32::powi(join_x, 2) + f32::powi(join_y, 2));
        join_x /= join_len;
        join_y /= join_len;

        let cos_angle = normal.x * join_x + normal.y * join_y;
        let l = 1.0 / cos_angle;

        bend_norm_x = join_x * l;
        bend_norm_y = join_y * l;

        verts[len-2].norm_x = bend_norm_x;
        verts[len-2].norm_y = bend_norm_y;
        verts[len-3].norm_x = -bend_norm_x;
        verts[len-3].norm_y = -bend_norm_y;
        verts[len-4].norm_x = bend_norm_x;
        verts[len-4].norm_y = bend_norm_y;
    } else {
        bend_norm_x = normal.x;
        bend_norm_y = normal.y;
    }

    // Now construct the tris
    verts.push(mapbox::LineVert { x:   cx, y:   cy, norm_x:  bend_norm_x, norm_y:  bend_norm_y, sign: 1 });
    verts.push(mapbox::LineVert { x:   cx, y:   cy, norm_x: -bend_norm_x, norm_y: -bend_norm_y, sign: -1 });
    verts.push(mapbox::LineVert { x: v1.x, y: v1.y, norm_x:  normal.x, norm_y:  normal.y, sign: 1 });

    verts.push(mapbox::LineVert { x: v1.x, y: v1.y, norm_x: -normal.x, norm_y: -normal.y, sign: -1 });
    verts.push(mapbox::LineVert { x: v1.x, y: v1.y, norm_x:  normal.x, norm_y:  normal.y, sign: 1 });
    verts.push(mapbox::LineVert { x:   cx, y:   cy, norm_x: -bend_norm_x, norm_y: -bend_norm_y, sign: -1 });
}

fn placeholder_tile(x: u64, y: u64, z: u8) -> Tile {
    let mut verts: Vec<mapbox::LineVert> = vec![];

    let mut lv = Vector2::new(0.0, 0.0);

    {
        let nv = Vector2::new(0.0, 4096.0);
        add_point(&mut verts, lv, nv, false);
        lv = nv;
    }

    {
        let nv = Vector2::new(4096.0, 4096.0);
        add_point(&mut verts, lv, nv, true);
        lv = nv;
    }

    {
        let nv = Vector2::new(4096.0, 0.0);
        add_point(&mut verts, lv, nv, true);
        lv = nv;
    }

    {
        let nv = Vector2::new(0.0, 0.0);
        add_point(&mut verts, lv, nv, true);
    }

    let mut vao = 0;
    unsafe { gl::GenVertexArrays(1, &mut vao) };

    let mut vbo = 0;
    unsafe { gl::GenBuffers(1, &mut vbo) };

    unsafe {
        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(gl::ARRAY_BUFFER, (verts.len() * std::mem::size_of::<mapbox::LineVert>()) as _, verts.as_ptr().cast(), gl::STATIC_DRAW);

        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<mapbox::LineVert>() as i32, 0 as *const _);
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<mapbox::LineVert>() as i32, (2 * std::mem::size_of::<f32>()) as *const _);
        gl::EnableVertexAttribArray(1);
        gl::VertexAttribPointer(2, 1, gl::BYTE, gl::FALSE, std::mem::size_of::<mapbox::LineVert>() as i32, (4 * std::mem::size_of::<f32>()) as *const _);
        gl::EnableVertexAttribArray(2);

        gl::BindBuffer(gl::ARRAY_BUFFER, 0);
        gl::BindVertexArray(0);
    }

    return Tile{
        x,
        y,
        z,
        extent: 4096,
        vao,
        vbo,
        vertex_len: verts.len(),
    };
}

fn main() {
    // Load font
    // dbg!(x);

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
        // layout(location = 1) in vec2 a_bary;
        out vec3 v_bary;
        out float v_dist;


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

            vec2 gp = position + normal*tesselation_width;
            gl_Position = transform * vec4(gp, 0.0, 1.0);
        }
    ";

    const FRAG_SHADER: &str = "
        #version 330 core
        #extension GL_OES_standard_derivatives : enable
        out vec4 color;

        in vec3 v_bary;
        in float v_dist;

        uniform float tesselation_width;
        const vec4 border_color = vec4(.5, .5, .5, 1);

        uniform vec4 fill_color;

        const float feather = 1.5;
        const float line_width = 0.7;
        const float line_smooth = .5;
        const vec4 edge_color = vec4(0, 0, 0, 1);

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

        void main() {
            float dist = abs(v_dist);

            float border = 2/tesselation_width;

            color = fill_color;


            { // Border
                // float ffactor = linearstep(1-border-feather, 1-border, dist);
                // color = mix(color, border_color, ffactor);
            }

            float invdist = 1-dist;
            float grad = fwidth(invdist);
            float ffactor = clamp(invdist/(feather*grad), 0, 1);
            color.a *= ffactor;

            // color = mix(edge_color, color, edge_factor());
            // color = vec4(.9, .9, .9, 1.0);
        }
    ";

    let shader_program;
    {
        let program = compile_shader(VERT_SHADER, FRAG_SHADER);
        let transform_str = CString::new("transform").unwrap();
        let width_str = CString::new("tesselation_width").unwrap();
        let fill_color_str = CString::new("fill_color").unwrap();
        unsafe {
            shader_program = LineProg{
                program,
                transform: gl::GetUniformLocation(program, transform_str.as_ptr()),
                width: gl::GetUniformLocation(program, width_str.as_ptr()),
                fill_color: gl::GetUniformLocation(program, fill_color_str.as_ptr()),
            }
        }
    }

    let tris = triangulate(
        &[100.0, 100.0, 200.0, 200.0],
        &[100.0, 200.0, 200.0, 100.0]
    );

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

    /*
    unsafe {
        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(gl::ARRAY_BUFFER, (vertecies.len() * std::mem::size_of::<f32>()) as _, vertecies.as_ptr().cast(), gl::STATIC_DRAW);

        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 5 * std::mem::size_of::<f32>() as i32, 0 as *const _);
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, 5 * std::mem::size_of::<f32>() as i32, (3 * std::mem::size_of::<f32>()) as *const _);
        gl::EnableVertexAttribArray(1);

        gl::BindBuffer(gl::ARRAY_BUFFER, 0);
        gl::BindVertexArray(0);
    }
    */
    let file = std::fs::File::open("aalborg.mvt").unwrap();
    let tile = compile_tile(0, 0, 0, file).unwrap();
    let file = std::fs::File::open("aalborg.mvt").unwrap();
    let ctile = compile_tile(1, 1, 1, file).unwrap();
    let ptile = placeholder_tile(1, 0, 1);

    // -------------------------------------------

    let mut projection = ortho(0.0, display.width as _, display.height as _, 0.0, 0.0, 1.0);
    let mut hold: Option<Vector2> = None;
    let mut scale = 1.0;
    let mut viewport_pos = Vector2::new(0.0, 0.0);
    let mut mouse_world = Vector2::new(0.0, 0.0);
    const MAP_SIZE : u64 = 512;
    while !window.should_close() {
        glfw.poll_events();

        display.size_change = false;

        let mut zoom = 0.0;
        for (_, event) in glfw::flush_messages(&events) {
            glfw_handle_event(&mut window, event, &mut display, &mut zoom);
        }

        if zoom != 0.0 {
            let old_scale = scale;
            scale *= 1.0 + zoom;

            // Limit the scale to some sensible? values
            scale = scale.clamp(1.0, 1000.0);
            zoom = scale/old_scale - 1.0;

            viewport_pos.x += display.mouse_x as f32 / scale * zoom;
            viewport_pos.y += display.mouse_y as f32 / scale * zoom;
        }

        {
            // Calculate the mouse position in the world
            mouse_world.x = display.mouse_x as f32;
            mouse_world.y = display.mouse_y as f32;
            mouse_world.divf(scale);
            mouse_world.addv2(&viewport_pos);
        }

        if display.size_change {
            projection = ortho(0.0, display.width as _, display.height as _, 0.0, 0.0, 1.0);
        }

        if window.get_mouse_button(glfw::MouseButtonLeft) == glfw::Action::Press {
            if let Some(hold) = hold {
                let mut diff = hold;
                diff.subv2(&mouse_world);
                viewport_pos.addv2(&diff);
                mouse_world = hold;
            } else {
                hold = Some(mouse_world);
            }
        } else {
            hold = None;
        }

        let visible_tiles = vec![
            &tile,
            &ctile,
            &ptile,
        ];

        let camera_matrix = mat4_multply(&mat4_translate(-viewport_pos.x, -viewport_pos.y), &mat4_scale(scale));

        clear_color(Color(0.3, 0.4, 0.6, 1.0));

        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        let scale_level = 1;

        unsafe {
            gl::UseProgram(shader_program.program);
        }

        let vp = mat4_multply(&camera_matrix, &projection);
        for tile in visible_tiles {
            let grid_step = MAP_SIZE / (tile.z as u64+1);

            // Transform the tilelocal coordinates to global coordinates
            let tile_transform;
            {
                let xcoord = tile.x * scale_level * grid_step;
                let ycoord = tile.y * scale_level * grid_step;
                let tile_matrix = mat4_multply(&mat4_scale(grid_step as f32/ tile.extent as f32), &mat4_translate(xcoord as f32, ycoord as f32));
                tile_transform = mat4_multply(&tile_matrix, &vp);
            }


            // Calculate the screen position of the time and scissor that
            {
                let mut v1 = Vector2::new(0.0, tile.extent as f32);
                let mut v2 = Vector2::new(tile.extent as f32, 0.0);

                // The clipspace transform
                let mvp2d = mat4_to_mat3(&tile_transform);
                v1.apply_transform(&mvp2d);
                v2.apply_transform(&mvp2d);

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
                    gl::Scissor(v1.x as i32, v1.y as i32, v2.x as i32, v2.y as i32);
                }
            }

            unsafe {
                gl::Clear(gl::COLOR_BUFFER_BIT);
            }

            unsafe {
                gl::UniformMatrix4fv(shader_program.transform, 1, gl::TRUE, tile_transform.as_ptr());
                gl::BindVertexArray(tile.vao);

                gl::Uniform1f(shader_program.width, 3.0);
                gl::Uniform4f(shader_program.fill_color, 0.8, 0.8, 0.8, 1.0);
                gl::DrawArrays(gl::TRIANGLES, 0, tile.vertex_len as _);

                gl::Uniform1f(shader_program.width, 2.0);
                gl::Uniform4f(shader_program.fill_color, 1.0, 1.0, 1.0, 1.0);
                gl::DrawArrays(gl::TRIANGLES, 0, tile.vertex_len as _);

                gl::BindVertexArray(0);
                gl::Disable(gl::SCISSOR_TEST);
            }
        }

        draw_text(&projection, &font, "Hello world!".as_bytes(), &mouse_world);

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
