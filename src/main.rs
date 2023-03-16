mod triangulate;
mod mapbox;

use triangulate::triangulate;
use std::convert::TryInto;

use glfw;
use glfw::Context;
use gl;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;
const TITLE: &str = "Hello From OpenGL World!";

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
        1.0+factor,        0.0,        0.0, 0.0,
               0.0, 1.0+factor,        0.0, 0.0,
               0.0,        0.0, 1.0+factor, 0.0,
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

const IDENTITY: [f32; 16] = [
           1.0,    0.0,    0.0, 0.0,
           0.0,    1.0,    0.0, 0.0,
           0.0,    0.0,    1.0, 0.0,
           0.0,    0.0,    0.0, 1.0,
];

fn main() {
    let mut file = std::fs::File::open("aalborg.mvt").unwrap();
    let (start, verts) = mapbox::read_one_linestring(&mut mapbox::pbuf::Message::new(&mut file)).unwrap();
    // dbg!(x);

    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::Resizable(false));

    let mut display;
    let (mut window, events) = glfw.create_window(WIDTH, HEIGHT, TITLE, glfw::WindowMode::Windowed).unwrap();
    {
        let (screen_width, screen_height) = window.get_framebuffer_size();
        display = DisplayState{
            width: screen_width,
            height: screen_height,
            size_change: true,
            mouse_x: 0.0,
            mouse_y: 0.0,
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
    // -------------------------------------------

    const VERT_SHADER: &str = "
        #version 330 core
        #extension GL_ARB_explicit_uniform_location : enable
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 normal;
        layout(location = 2) in float sign;
        // layout(location = 1) in vec2 a_bary;
        varying vec3 v_bary;
        varying float v_dist;


        layout(location = 0) uniform mat4 transform;
        layout(location = 1) uniform float tesselation_width;

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
        #extension GL_ARB_explicit_uniform_location : enable
        #extension GL_OES_standard_derivatives : enable
        out vec4 color;
        varying vec3 v_bary;
        varying float v_dist;

        layout(location = 1) uniform float tesselation_width;
        const vec4 border_color = vec4(.5, .5, .5, 1);

        const float line_width = 0.7;
        const float line_smooth = .5;
        const vec4 edge_color = vec4(0, 0, 0, 1);

        layout(location = 2) uniform vec4 fill_color;

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
            float feather = .8/tesselation_width;
            float border = 2/tesselation_width;

            color = fill_color;

            float dist = abs(v_dist);

            { // Border
                // float ffactor = linearstep(1-border-feather, 1-border, dist);
                // color = mix(color, border_color, ffactor);
            }

            float ffactor = 1-smoothstep(1-feather, 1, dist);
            color.a *= ffactor;

            // color = mix(edge_color, color, edge_factor());
            // color = vec4(.9, .9, .9, 1.0);
        }
    ";

    let vertex_shader = unsafe { gl::CreateShader(gl::VERTEX_SHADER) };
    unsafe {
        gl::ShaderSource(vertex_shader, 1, &VERT_SHADER.as_bytes().as_ptr().cast(), &VERT_SHADER.len().try_into().unwrap());
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
        gl::ShaderSource(fragment_shader, 1, &FRAG_SHADER.as_bytes().as_ptr().cast(), &FRAG_SHADER.len().try_into().unwrap());
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

    let shader_program = unsafe { gl::CreateProgram() };
    unsafe {
        gl::AttachShader(shader_program, vertex_shader);
        gl::AttachShader(shader_program, fragment_shader);
        gl::LinkProgram(shader_program);

        let mut success = 0;
        gl::GetProgramiv(shader_program, gl::LINK_STATUS, &mut success);
        if success == 0 {
            let mut v: Vec<u8> = Vec::with_capacity(1024);
            let mut log_len = 0_i32;
            gl::GetProgramInfoLog(shader_program, 1024, &mut log_len, v.as_mut_ptr().cast());
            v.set_len(log_len.try_into().unwrap());
            panic!("Program Link Error: {}", String::from_utf8_lossy(&v));
        }

        gl::DetachShader(shader_program, vertex_shader);
        gl::DetachShader(shader_program, fragment_shader);
        gl::DeleteShader(vertex_shader);
        gl::DeleteShader(fragment_shader);
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

    let mut vao = 0;
    unsafe { gl::GenVertexArrays(1, &mut vao) };

    let mut vbo = 0;
    unsafe { gl::GenBuffers(1, &mut vbo) };

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

        gl::Enable(gl::BLEND);
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
    }

    // -------------------------------------------
    println!("OpenGL version: {}", gl_get_string(gl::VERSION));
    println!("GLSL version: {}", gl_get_string(gl::SHADING_LANGUAGE_VERSION));

    let mut projection = ortho(0.0, display.width as _, display.height as _, 0.0, 0.0, 1.0);
    let mut camera_matrix = IDENTITY;
    let mut hold: Option<(f64, f64)> = None;
    while !window.should_close() {
        glfw.poll_events();

        display.size_change = false;

        for (_, event) in glfw::flush_messages(&events) {
            glfw_handle_event(&mut window, event, &mut display, &mut camera_matrix);
        }
        if display.size_change {
            projection = ortho(0.0, display.width as _, display.height as _, 0.0, 0.0, 1.0);
        }

        clear_color(Color(0.3, 0.4, 0.6, 1.0));

        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        // zoom = (display.mouse_x / 700.0) as f32;

        if window.get_mouse_button(glfw::MouseButtonLeft) == glfw::Action::Press {
            if let Some((lastx, lasty)) = hold {
                camera_matrix = mat4_multply(&camera_matrix, &mat4_translate((display.mouse_x-lastx) as _, (display.mouse_y-lasty) as _));
            }
            hold = Some((display.mouse_x, display.mouse_y));
        } else {
            hold = None;
        }

        unsafe {
            gl::UseProgram(shader_program);
            let mvp = mat4_multply(&camera_matrix, &projection);

            gl::UniformMatrix4fv(0, 1, gl::TRUE, mvp.as_ptr());
            gl::BindVertexArray(vao);

            gl::Uniform1f(1, 3.0);
            gl::Uniform4f(2, 0.8, 0.8, 0.8, 1.0);
            gl::DrawArrays(gl::TRIANGLES, 0, verts.len() as _);

            gl::Uniform1f(1, 2.0);
            gl::Uniform4f(2, 1.0, 1.0, 1.0, 1.0);
            gl::DrawArrays(gl::TRIANGLES, 0, verts.len() as _);

            gl::BindVertexArray(0);
        }

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

fn glfw_handle_event(window: &mut glfw::Window, event: glfw::WindowEvent, state: &mut DisplayState, camera: &mut [f32; 16]) {
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
            state.mouse_x = x;
            state.mouse_y = y;
        },
        Event::Scroll(_, y) => {
            let mut transform = mat4_translate(state.mouse_x as f32, state.mouse_y as f32);
            transform = mat4_multply(&mat4_scale(y as f32 * 0.1), &transform, );
            transform = mat4_multply(&mat4_translate(-state.mouse_x as _, -state.mouse_y as _), &transform, );
            *camera = mat4_multply( camera, &transform, );
        },
        _ => {},
    }
}
