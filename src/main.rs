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

fn main() {
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
        }
    }

    window.make_current();
    window.set_key_polling(true);
    window.set_framebuffer_size_polling(true);
    gl::load_with(|ptr| window.get_proc_address(ptr) as *const _);

    unsafe {
        gl::Viewport(0, 0, display.width, display.height);
        clear_color(Color(0.4, 0.4, 0.4, 1.0));
    }
    // -------------------------------------------

    const VERT_SHADER: &str = "
        #version 330 core
        #extension GL_ARB_explicit_uniform_location : enable
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec2 a_bary;
        varying vec3 v_bary;

        layout(location = 0) uniform mat4 transform;

        void main() {
            v_bary = vec3(a_bary, 1 - a_bary.x - a_bary.y);
            gl_Position = transform * vec4(position, 1.0);
        }
    ";

    const FRAG_SHADER: &str = "
        #version 330 core
        #extension GL_OES_standard_derivatives : enable
        out vec4 color;
        varying vec3 v_bary;

        const float line_width = 1.0;
        const float line_smooth = 1.5;
        const vec4 edge_color = vec4(0, 0, 0, 1);

        const vec4 fill_color = vec4(1, 1, 1, 1);

        float edge_factor() {
            vec3 d = fwidth(v_bary);
            vec3 smth = line_smooth * d;
            d *= line_width;
            vec3 f = smoothstep(d, d+smth, v_bary);
            return min(min(f.x, f.y), f.z);
        }

        void main() {
            color = mix(edge_color, fill_color, edge_factor());
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

    // -------------------------------------------
    println!("OpenGL version: {}", gl_get_string(gl::VERSION));
    println!("GLSL version: {}", gl_get_string(gl::SHADING_LANGUAGE_VERSION));

    let mut transform = ortho(0.0, display.width as _, display.height as _, 0.0, 0.0, 1.0);
    while !window.should_close() {
        glfw.poll_events();

        display.size_change = false;

        for (_, event) in glfw::flush_messages(&events) {
            glfw_handle_event(&mut window, event, &mut display);
        }
        if display.size_change {
            transform = ortho(0.0, display.width as _, display.height as _, 0.0, 0.0, 1.0);
        }

        clear_color(Color(0.3, 0.4, 0.6, 1.0));

        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        unsafe {
            gl::UseProgram(shader_program);
            gl::UniformMatrix4fv(0, 1, gl::TRUE, transform.as_ptr());
            gl::BindVertexArray(vao);

            gl::DrawArrays(gl::TRIANGLES, 0, (vertecies.len()  / 5) as _);

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
}

pub fn gl_get_string<'a>(name: gl::types::GLenum) -> &'a str {
    let v = unsafe { gl::GetString(name) };
    let v: &std::ffi::CStr = unsafe { std::ffi::CStr::from_ptr(v as *const i8) };
    v.to_str().unwrap()
}

fn glfw_handle_event(window: &mut glfw::Window, event: glfw::WindowEvent, state: &mut DisplayState) {
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
        _ => {},
    }
}
