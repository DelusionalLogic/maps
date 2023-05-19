pub mod pbuf {
    use std::io::Read;

    #[derive(Debug)]
    pub enum Error {
        EOF(),
        IOError(std::io::Error),
        EncodingError(std::string::FromUtf8Error),
    }

    pub type Result<T> = std::result::Result<T, Error>;

    impl From<std::io::Error> for Error {
        fn from(value: std::io::Error) -> Self {
            return Error::IOError(value);
        }
    }

    impl From<std::string::FromUtf8Error> for Error {
        fn from(value: std::string::FromUtf8Error) -> Self {
            return Error::EncodingError(value);
        }
    }

    #[derive(Debug, PartialEq)]
    pub enum WireType {
        VarInt,
        I64,
        Len,
        SGroup,
        EGroup,
        I32,

        // Internal stuff not in spec
        EOM,
    }

    impl From<u64> for WireType {
        fn from(x: u64) -> Self {
            return match x {
                0 => WireType::VarInt,
                1 => WireType::I64,
                2 => WireType::Len,
                3 => WireType::SGroup,
                4 => WireType::EGroup,
                5 => WireType::I32,
                _ => panic!("WireType out of range"),
            };
        }
    }

    #[derive(Debug, PartialEq)]
    pub struct TypeAndTag {
        pub wtype: WireType,
        pub tag: u64,
    }

    impl TypeAndTag {
        pub fn parse(x: u64) -> Self {
            let wtype: WireType = (x & 7).into();
            let tag = x >> 3;

            return TypeAndTag{
                wtype,
                tag
            };
        }

        pub fn eom() -> Self {
            return TypeAndTag {
                wtype: WireType::EOM,
                tag: 0,
            };
        }
    }

    struct BoundedRead<'a, T: Read> {
        reader : &'a mut T,
        pos: usize,
    }

    impl<'a, T: Read> BoundedRead<'a, T> {
        pub fn new(reader: &'a mut T) -> Self {
            return BoundedRead {
                reader,
                pos: 0,
            }
        }
    }

    impl <'a, T: Read> Read for BoundedRead<'a, T> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.reader.read(buf)?;
            self.pos += n;

            return Ok(n);
        }
    }

    pub struct Message<'a, T: Read> {
        reader: BoundedRead<'a, T>,
        stack: Vec<usize>,
    }

    pub struct PackedField<'a, 'b, X: Read, T> {
        message: &'a mut Message<'b, X>,
        fun: fn(&mut Message<'b, X>) -> T,

        end: usize,
    }

    impl<'a, 'b, X: Read, T> Iterator for PackedField<'a, 'b, X, T> {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            if self.message.reader.pos == self.end {
                return None;
            }
            debug_assert!(self.message.reader.pos < self.end);

            return Some((self.fun)(self.message));
        }
    }

    impl <'a, T: Read> Message<'a, T> {
        pub fn new(reader: &'a mut T) -> Self {
            return Self {
                reader: BoundedRead::new(reader),
                stack: vec![],
            }
        }

        pub fn next(&mut self) -> Result<TypeAndTag> {
            if let Some(x) = self.stack.last() {
                if self.reader.pos == *x {
                    self.stack.pop();
                    return Ok(TypeAndTag::eom());
                }
                debug_assert!(self.reader.pos < *x);
            }

            let tat = TypeAndTag::parse(self.read_var_int()?);
            return Ok(tat);
        }

        pub fn read_string(&mut self) -> Result<String> {
            let len = self.read_var_int()? as usize;
            let mut res = vec![0; len];

            self.reader.read_exact(&mut res)?;

            return Ok(String::from_utf8(res)?);
        }

        pub fn read_var_int(&mut self) -> Result<u64> {
            let mut value: u64 = 0;

            let mut byte: [u8; 1] = [0];
            let mut i = 0;
            loop {
                let n = self.reader.read(&mut byte)?;
                if n == 0 {
                    return Err(Error::EOF());
                }

                value |= (byte[0] as u64 & 0x7F) << i;

                i += 7;
                if byte[0] & 0x80 == 0 {
                    break;
                }
            }

            return Ok(value);
        }

        pub fn read_packed_var_int<'b>(&'b mut self) -> Result<PackedField<'b, 'a, T, Result<u64>>> {
            let len = self.read_var_int()?;
            let pos = self.reader.pos;
            return Ok(PackedField {
                message: self,
                fun: Self::read_var_int,
                end: pos + len as usize,
            });
        }

        pub fn read_zig(&mut self) -> Result<i64> {
            let value = self.read_var_int()?;
            return Ok(decode_zig(value));
        }

        pub fn enter_message(&mut self) -> Result<()> {
            let len = self.read_var_int()? as usize;
            if let Some(x) = self.stack.last() {
                if len > *x {
                    panic!("Submessage length is longer than container");
                }
            }
            self.stack.push(len + self.reader.pos);
            return Ok(());
        }

        pub fn exit_message(&mut self) -> Result<()> {
            if let Some(x) = self.stack.pop() {
                self.fastforward(x - self.reader.pos)?;
            } else {
                panic!("Not in message");
            }

            return Ok(());
        }

        fn fastforward(&mut self, bytes: usize) -> Result<()> {
            let mut buf = vec![0; bytes];
            match self.reader.read_exact(&mut buf) {
                Ok(_) => return Ok(()),
                Err(x) if x.kind() == std::io::ErrorKind::UnexpectedEof => {
                    return Err(Error::EOF());
                },
                Err(x) => return Err(x.into()),
            };
        }

        pub fn skip(&mut self, wtype: WireType) -> Result<()> {
            match wtype {
                WireType::VarInt => {
                    self.read_var_int()?;
                },
                WireType::I64 => {
                    self.fastforward(8)?;
                },
                WireType::Len => {
                    let len = self.read_var_int()? as usize;
                    self.fastforward(len)?;
                },
                WireType::I32 => {
                    self.fastforward(4)?;
                },
                _ => { panic!(); },
            };
            return Ok(());
        }
    }

    pub fn read_var_int<T: Read>(reader: &mut T) -> Result<u64> {
        let mut value: u64 = 0;

        let mut byte: [u8; 1] = [0];
        let mut i = 0;
        loop {
            let n = reader.read(&mut byte)?;
            if n == 0 {
                return Err(Error::EOF());
            }

            value |= (byte[0] as u64 & 0x7F) << i;

            i += 7;
            if byte[0] & 0x80 == 0 {
                break;
            }
        }

        return Ok(value);
    }

    pub fn decode_zig(value: u64) -> i64 {
        // If the bottom bit is set flip will be all 1's, if it's unset it will be all 0's
        let flip = -((value & 1) as i64) as u64;
        let signed = ((value >> 1) ^ flip) as i64;
        return signed;
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn bounded_reader_limit() {
            let mut data = [0b00000001, 0b00000001].as_ref();
            let mut reader = BoundedRead::new(&mut data);

            let mut buf = [0; 1];
            reader.read(&mut buf).unwrap();
            assert_eq!(reader.pos, 1);

            reader.read(&mut buf).unwrap();
            assert_eq!(reader.pos, 2);
        }

        #[test]
        fn var_int_1() {
            let mut data = [0b00000001].as_ref();
            let mut msg = Message::new(&mut data);
            let value = msg.read_var_int().unwrap();

            assert_eq!(value, 1);
        }

        #[test]
        fn var_int_150() {
            let mut data = [0b10010110, 0b00000001].as_ref();
            let mut msg = Message::new(&mut data);
            let value = msg.read_var_int().unwrap();

            assert_eq!(value, 150);
        }

        #[test]
        fn zig_0() {
            let mut data = [0b00000000].as_ref();
            let mut msg = Message::new(&mut data);
            let value = msg.read_zig().unwrap();

            assert_eq!(value, 0);
        }

        #[test]
        fn zig_1() {
            let mut data = [0b00000010].as_ref();
            let mut msg = Message::new(&mut data);
            let value = msg.read_zig().unwrap();

            assert_eq!(value, 1);
        }

        #[test]
        fn zig_neg1() {
            let mut data = [0b00000001].as_ref();
            let mut msg = Message::new(&mut data);
            let value = msg.read_zig().unwrap();

            assert_eq!(value, -1);
        }

        #[test]
        fn zig_max() {
            let mut data = [0b11111110, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b00000001].as_ref();
            let mut msg = Message::new(&mut data);
            let value = msg.read_zig().unwrap();

            assert_eq!(value, 0x7FFFFFFFFFFFFFFF);
        }

        #[test]
        fn zig_neg_max() {
            let mut data = [0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b00000001].as_ref();
            let mut msg = Message::new(&mut data);
            let value = msg.read_zig().unwrap();

            assert_eq!(value, -0x8000000000000000);
        }

        #[test]
        fn skip_i64() {
            let mut data = [0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000001].as_ref();
            let mut msg = Message::new(&mut data);
            msg.skip(WireType::I64).unwrap();
            let value = msg.read_var_int().unwrap();

            assert_eq!(value, 1);
        }

        #[test]
        #[ignore]
        fn skip_i64_eof() {
            let mut data = [0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000001].as_ref();
            let mut msg = Message::new(&mut data);
            msg.skip(WireType::I64).unwrap();
            let value = msg.read_var_int().unwrap();

            assert_eq!(value, 1);
        }

        #[test]
        fn skip_i32() {
            let mut data = [0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000001].as_ref();
            let mut msg = Message::new(&mut data);
            msg.skip(WireType::I32).unwrap();
            let value = msg.read_var_int().unwrap();

            assert_eq!(value, 1);
        }

        #[test]
        fn skip_len() {
            let mut data = [0b00000001, 0b00000000, 0b00000001].as_ref();
            let mut msg = Message::new(&mut data);
            msg.skip(WireType::Len).unwrap();
            let value = msg.read_var_int().unwrap();

            assert_eq!(value, 1);
        }

        #[test]
        fn field_1_varint() {
            let mut data = [0b00001000, 0b00000000].as_ref();
            let mut msg = Message::new(&mut data);
            let field = msg.next().unwrap();

            assert_eq!(field.wtype, WireType::VarInt);
            assert_eq!(field.tag, 1);
        }

        #[test]
        fn field_2_varint() {
            let mut data = [0b00001000, 0b00000000, 0b00010000, 0b00000000].as_ref();
            let mut msg = Message::new(&mut data);

            // skip first field
            let field = msg.next().unwrap();
            msg.skip(field.wtype).unwrap();

            let field = msg.next().unwrap();

            assert_eq!(field.wtype, WireType::VarInt);
            assert_eq!(field.tag, 2);
        }

        #[test]
        fn field_eof() {
            let mut data = [].as_ref();
            let mut msg = Message::new(&mut data);

            let err = msg.next();
            assert!(matches!(err.unwrap_err(), Error::EOF()))
        }

        #[test]
        fn submessage() {
            let mut data = [0b00001010, 0b00000010, 0b00001000, 0b00000011].as_ref();
            let mut msg = Message::new(&mut data);

            let field = msg.next().unwrap();
            assert_eq!(field.wtype, WireType::Len);
            assert_eq!(field.tag, 1);

            msg.enter_message().unwrap();
            let field = msg.next().unwrap();
            assert_eq!(field.wtype, WireType::VarInt);
            assert_eq!(field.tag, 1);
            let value = msg.read_var_int().unwrap();
            assert_eq!(value, 3);

            let field = msg.next().unwrap();
            assert_eq!(field.wtype, WireType::EOM);

            let err = msg.next();
            assert!(matches!(err.unwrap_err(), Error::EOF()));
        }

        #[test]
        fn skip_submessage() {
            let mut data = [0b00001010, 0b00000010, 0b00001000, 0b00000011, 0b00010000, 0b00000000].as_ref();
            let mut msg = Message::new(&mut data);

            let field = msg.next().unwrap();
            assert_eq!(field.wtype, WireType::Len);
            assert_eq!(field.tag, 1);

            msg.enter_message().unwrap();
            msg.exit_message().unwrap();

            let field = msg.next().unwrap();
            assert_eq!(field.wtype, WireType::VarInt);
            assert_eq!(field.tag, 2);
            let value = msg.read_var_int().unwrap();
            assert_eq!(value, 0);

            let err = msg.next();
            assert!(matches!(err.unwrap_err(), Error::EOF()));
        }
    }
}

use std::io::Read;

#[derive(Clone,Copy,Debug)]
pub struct Vector {
    pub x: f32,
    pub y: f32,
}

impl Vector {
    pub fn new(x: f32, y: f32) -> Self {
        return Vector { x, y };
    }

    pub fn unit(&mut self) {
        let len = f32::sqrt(f32::powi(self.x, 2) + f32::powi(self.y, 2));
        self.x /= len;
        self.y /= len;
    }
}

#[derive(Clone,Copy,Debug)]
#[repr(packed)]
pub struct LineVert {
    pub x: f32,
    pub y: f32,
    pub norm_x: f32,
    pub norm_y: f32,
    pub sign: i8,
}

#[derive(Debug)]
pub struct LineStart {
    pos: usize,
}

pub fn read_one_linestring<T: Read>(reader: &mut pbuf::Message<T>) -> pbuf::Result<(Vec<LineStart>, Vec<LineVert>, Vec<LineVert>)> {
    let mut start = vec![];
    let mut vert : Vec<LineVert> = vec![];
    let mut polys : Vec<LineVert> = vec![];

    println!("Reading!");
    while let Ok(field) = reader.next() {
        match field {
            pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 3} => {
                let mut seen_name = false;
                reader.enter_message()?;
                'layer: while let Ok(field) = reader.next() {
                    match field {
                        pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                        pbuf::TypeAndTag{wtype: pbuf::WireType::VarInt, tag: 15} => {
                            let version = reader.read_var_int()?;
                            assert_eq!(version, 2);
                        }
                        pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 1} => {
                            assert!(!seen_name);
                            let name = reader.read_string()?;
                            println!("Layer named {}", name);
                            if name != "roads" && name != "earth" {
                                reader.exit_message()?;
                                break 'layer;
                            }

                            seen_name = true;
                        }
                        pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 2} => {
                            assert!(seen_name);
                            reader.enter_message()?;
                            let mut seen_type = None;
                            'feature: while let Ok(field) = reader.next() {
                                match field {
                                    pbuf::TypeAndTag{wtype: pbuf::WireType::VarInt, tag: 3} => {
                                        let geo_type = reader.read_var_int()?;
                                        // if  geo_type != 2 {
                                        //     reader.exit_message()?;
                                        //     break 'feature;
                                        // }

                                        seen_type = Some(geo_type);
                                    },
                                    pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 4} => {
                                        assert!(seen_type.is_some());

                                        if seen_type.unwrap() == 2 {
                                            let mut cx = 0.0;
                                            let mut cy = 0.0;
                                            let mut it = reader.read_packed_var_int()?;
                                            while let Some(cmd) = it.next() {
                                                let cmd = cmd?;
                                                if (cmd & 7) == 1 { // MoveTo
                                                    let x = pbuf::decode_zig(it.next().unwrap().unwrap());
                                                    cx += x as f32;
                                                    let y = pbuf::decode_zig(it.next().unwrap().unwrap());
                                                    cy += y as f32;

                                                    start.push(LineStart{pos: vert.len()});
                                                    // Don't push any verts until the line is drawn
                                                } else if (cmd & 7) == 2 { // LineTo
                                                    for i in 0..(cmd >> 3) {
                                                        let x = pbuf::decode_zig(it.next().unwrap().unwrap()) as f32;
                                                        let px = cx + x;
                                                        let y = pbuf::decode_zig(it.next().unwrap().unwrap()) as f32;
                                                        let py = cy + y;

                                                        pmtile::add_point(&mut vert, crate::math::Vector2 { x: cx, y: cy }, crate::math::Vector2 { x: px, y: py }, i != 0);

                                                        cx = px;
                                                        cy = py;
                                                    }
                                                } else if (cmd & 7) == 7 { // ClosePath
                                                    // Do nothing, the path is assumed closed
                                                }
                                            }
                                        } else if seen_type.unwrap() == 3 {
                                            println!("POLY");
                                            reader.skip(pbuf::WireType::Len)?;
                                        }
                                    }
                                    pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                                    x => { reader.skip(x.wtype)?; },
                                }
                            }
                        }
                        x => { reader.skip(x.wtype)?; },
                    }
                }
            },
            x => { reader.skip(x.wtype)?; },
        }
    }

    return Ok((start, vert, polys));
}

pub mod pmtile {
    use crate::math::Vector2;
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::io::Read;
    use std::io::Seek;
    use flate2::bufread::GzDecoder;
    use super::pbuf;
    use core::ptr::NonNull;

    struct BinarySlice {
        data: Vec<u8>,
        cursor: usize,
    }

    impl Read for BinarySlice {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let end = (self.cursor + buf.len()).min(self.data.len());
            let len = end - self.cursor;
            buf[0..len].copy_from_slice(&self.data[self.cursor..end]);

            self.cursor = end;
            return Ok(len);
        }
    }

    impl BinarySlice {
        pub fn from_read<T: Read + Seek>(read: &mut T, start: u64, len: usize) -> Self {

            let mut slice = BinarySlice {
                data: Vec::with_capacity(len),
                cursor: 0,
            };

            read.seek(std::io::SeekFrom::Start(start)).unwrap();
            slice.data.resize(len, 0);
            read.read_exact(&mut slice.data).unwrap();

            return slice;
        }

        pub fn from_read_decompress<T: Read + Seek>(read: &mut T, compression: &Compression, start: u64, len: usize) -> Self {
            let mut buf = Vec::with_capacity(len);
            read.seek(std::io::SeekFrom::Start(start)).unwrap();
            buf.resize(len, 0);
            read.read_exact(&mut buf).unwrap();

            let mut slice = BinarySlice {
                data: Vec::new(),
                cursor: 0,
            };

            match compression {
                Compression::GZip => {
                    let mut reader = GzDecoder::new(buf.as_slice());
                    reader.read_to_end(&mut slice.data).unwrap();
                }
                Compression::None => panic!(),
                Compression::Brotli => panic!(),
                Compression::Zstd => panic!(),
            }

            return slice;
        }

        pub fn skip(&mut self, size: usize) {
            self.cursor += size;
        }

        pub fn u64(&mut self) -> u64 {
            let end = self.cursor+std::mem::size_of::<u64>();
            let val = u64::from_le_bytes(self.data[self.cursor..end].try_into().unwrap());
            self.cursor = end;
            return val;
        }

        pub fn u16(&mut self) -> u16 {
            let end = self.cursor+std::mem::size_of::<u16>();
            let val = u16::from_le_bytes(self.data[self.cursor..end].try_into().unwrap());
            self.cursor = end;
            return val;
        }

        pub fn u8(&mut self) -> u8 {
            let end = self.cursor+std::mem::size_of::<u8>();
            let val = u8::from_le_bytes(self.data[self.cursor..end].try_into().unwrap());
            self.cursor = end;
            return val;
        }
    }

    pub fn coords_to_id(x: u64, y: u64, level: u8) -> u64 {
        fn rotate(n: i64, xy: &mut [i64; 2], rx: bool, ry: bool) {
            if !ry {
                if rx {
                    xy[0] = n - 1 - xy[0];
                    xy[1] = n - 1 - xy[1];
                }
                let t = xy[0];
                xy[0] = xy[1];
                xy[1] = t;
            }
        }


        let mut acc = 0;
        let mut tz = 0;
        while tz < level {
            acc += (0x1 << tz) * (0x1 << tz);
            tz += 1;
        }
        let n: i64 = 1 << level;
        let mut d = 0;
        let mut xy = [x as i64, y as i64];
        let mut s: i64 = n >> 1;
        while s > 0 {
            let rx = xy[0] & s > 0;
            let ry = xy[1] & s > 0;
            d += s * s * ((3 * rx as i64) ^ ry as i64);
            rotate(s, &mut xy, rx, ry);
            s = s >> 1;
        }
        return (acc + d) as u64;
    }

    pub struct Tile {
        pub tid: u64,
        pub x: u64,
        pub y: u64,
        pub z: u8,
        pub extent: u16,
        pub vao: u32,
        pub vbo: u32,
        pub vertex_len: usize,
    }

    impl Drop for Tile {
        fn drop(&mut self) {
            unsafe {
                gl::DeleteVertexArrays(1, &self.vao);
                gl::DeleteBuffers(1, &self.vbo);
            }
        }
    }

    pub fn add_point(verts: &mut Vec<super::LineVert>, lv: Vector2<f32>, v1: Vector2<f32>, connect_previous: bool) {
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
        verts.push(super::LineVert { x:   cx, y:   cy, norm_x:  bend_norm_x, norm_y:  bend_norm_y, sign: 1 });
        verts.push(super::LineVert { x:   cx, y:   cy, norm_x: -bend_norm_x, norm_y: -bend_norm_y, sign: -1 });
        verts.push(super::LineVert { x: v1.x, y: v1.y, norm_x:  normal.x, norm_y:  normal.y, sign: 1 });

        verts.push(super::LineVert { x: v1.x, y: v1.y, norm_x: -normal.x, norm_y: -normal.y, sign: -1 });
        verts.push(super::LineVert { x: v1.x, y: v1.y, norm_x:  normal.x, norm_y:  normal.y, sign: 1 });
        verts.push(super::LineVert { x:   cx, y:   cy, norm_x: -bend_norm_x, norm_y: -bend_norm_y, sign: -1 });
    }

    fn compile_tile<T: Read>(x: u64, y: u64, z: u8, reader: &mut T) -> Result<Tile, String> {
        let (start, verts, polys) = super::read_one_linestring(&mut pbuf::Message::new(reader)).unwrap();

        let mut vao = 0;
        unsafe { gl::GenVertexArrays(1, &mut vao) };

        let mut vbo = 0;
        unsafe { gl::GenBuffers(1, &mut vbo) };

        unsafe {
            gl::BindVertexArray(vao);

            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(gl::ARRAY_BUFFER, (verts.len() * std::mem::size_of::<super::LineVert>()) as _, verts.as_ptr().cast(), gl::STATIC_DRAW);

            gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<super::LineVert>() as i32, 0 as *const _);
            gl::EnableVertexAttribArray(0);
            gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<super::LineVert>() as i32, (2 * std::mem::size_of::<f32>()) as *const _);
            gl::EnableVertexAttribArray(1);
            gl::VertexAttribPointer(2, 1, gl::BYTE, gl::FALSE, std::mem::size_of::<super::LineVert>() as i32, (4 * std::mem::size_of::<f32>()) as *const _);
            gl::EnableVertexAttribArray(2);

            gl::BindBuffer(gl::ARRAY_BUFFER, 0);
            gl::BindVertexArray(0);
        }

        // @INCOMPLETE @CLEANUP: The extent here should be read from the file
        let tid = unsafe { TILE_NUM +=1; TILE_NUM };
        return Ok(Tile{
            tid,
            x,
            y,
            z,
            extent: 4096,
            vao,
            vbo,
            vertex_len: verts.len(),
        });
    }

    static mut TILE_NUM  : u64 = 0;
    pub fn placeholder_tile(x: u64, y: u64, z: u8) -> Tile {
        let mut verts: Vec<super::LineVert> = vec![];

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
            gl::BufferData(gl::ARRAY_BUFFER, (verts.len() * std::mem::size_of::<super::LineVert>()) as _, verts.as_ptr().cast(), gl::STATIC_DRAW);

            gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<super::LineVert>() as i32, 0 as *const _);
            gl::EnableVertexAttribArray(0);
            gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<super::LineVert>() as i32, (2 * std::mem::size_of::<f32>()) as *const _);
            gl::EnableVertexAttribArray(1);
            gl::VertexAttribPointer(2, 1, gl::BYTE, gl::FALSE, std::mem::size_of::<super::LineVert>() as i32, (4 * std::mem::size_of::<f32>()) as *const _);
            gl::EnableVertexAttribArray(2);

            gl::BindBuffer(gl::ARRAY_BUFFER, 0);
            gl::BindVertexArray(0);
        }

        let tid = unsafe { TILE_NUM +=1; TILE_NUM };
        return Tile{
            tid,
            x,
            y,
            z,
            extent: 4096,
            vao,
            vbo,
            vertex_len: verts.len(),
        };
    }

    pub enum DEntry {
        Leaf(u64, usize),
        Tile(u64, u64, u64, usize),
    }

    pub struct Directory {
        id: Vec<u64>,
        runlength: Vec<u64>,
        len: Vec<usize>,
        offset: Vec<u64>,

        next: Option<NonNull<Directory>>,
        prev: Option<NonNull<Directory>>,
    }

    impl Directory {
        fn parse(slice: &mut BinarySlice) -> Self {
            fn read_varint(slice: &mut BinarySlice) -> u64 {
                let mut value: u64 = 0;

                let mut i = 0;
                loop {
                    let n = slice.u8();

                    value |= (n as u64 & 0x7F) << i;

                    i += 7;
                    if n & 0x80 == 0 {
                        break;
                    }
                }

                return value;
            }

            let num_entries = read_varint(slice) as usize;
            assert!(num_entries > 0);

            let mut id = Vec::with_capacity(num_entries);
            let mut runlength = Vec::with_capacity(num_entries);
            let mut len = Vec::with_capacity(num_entries);
            let mut offset = Vec::with_capacity(num_entries);

            let mut pval = 0;
            for _ in 0..num_entries {
                let val = read_varint(slice);
                pval += val;
                id.push(pval);
            }

            for _ in 0..num_entries {
                let val = read_varint(slice);
                runlength.push(val);
            }

            for _ in 0..num_entries {
                let val = read_varint(slice) as usize;
                len.push(val);
            }

            let mut eoffset = 0;
            for i in 0..num_entries {
                let val = read_varint(slice);
                if val == 0 {
                    eoffset += len[i-1] as u64;
                } else {
                    eoffset = val-1;
                }
                offset.push(eoffset);
            }

            return Directory {
                id,
                runlength,
                len,
                offset,

                next: None,
                prev: None,
            };
        }

        fn find_entry(&self, id: u64) -> DEntry {
            let index = match self.id.binary_search(&id) {
                Ok(x) => x,
                Err(x) => x-1,
            };

            if self.runlength[index] == 0 {
                return DEntry::Leaf(self.offset[index], self.len[index]);
            }

            return DEntry::Tile(self.id[index], self.runlength[index], self.offset[index], self.len[index]);
        }

    }

    #[derive(Debug)]
    enum Compression {
        None,
        GZip,
        Brotli,
        Zstd,
    }

    impl From<u8> for Compression {
        fn from(i: u8) -> Self {
            match i {
                1 => Compression::None,
                2 => Compression::GZip,
                3 => Compression::Brotli,
                4 => Compression::Zstd,
                _ => panic!("Unknown compression"),
            }
        }

    }

    pub struct File {
        file: std::fs::File,
        root_offset: u64,
        root_len: usize,

        leaf_offset: u64,
        leaf_len: usize,
        tile_offset: u64,
        tile_len: usize,

        internal_compression: Compression,
        tile_compression: Compression,

        min_zoom: u8,
        pub max_zoom: u8,

        fdir: Option<NonNull<Directory>>,
        ldir: Option<NonNull<Directory>>,
        dcache: HashMap<u64, Directory>,
    }

    impl File {
        pub fn new<P: AsRef<std::path::Path>>(path: P) -> Self {
            let mut file = std::fs::File::open(path).unwrap();

            let mut slice: BinarySlice = BinarySlice::from_read(&mut file, 0, 127);
            assert!(slice.u16() == 0x4d50); // Check for PM header
            slice.skip(5);
            assert!(slice.u8() == 3); // Version

            let root_offset = slice.u64();
            let root_len = slice.u64() as usize;

            slice.skip(16);

            let leaf_offset = slice.u64();
            let leaf_len = slice.u64() as usize;
            let tile_offset = slice.u64();
            let tile_len = slice.u64() as usize;

            slice.skip(25);

            let internal_compression = slice.u8().into();
            let tile_compression = slice.u8().into();

            assert!(slice.u8() == 1); // Tile type (MVT)

            let min_zoom = slice.u8();
            let max_zoom = slice.u8();

            return File {
                file,
                root_offset,
                root_len,
                leaf_offset,
                leaf_len,
                tile_offset,
                tile_len,
                internal_compression,
                tile_compression,
                min_zoom,
                max_zoom,

                fdir: None,
                ldir: None,
                dcache: HashMap::new(),
            };
        }

        fn read_directory(&mut self, offset: u64, len: usize) -> &Directory {
            match self.dcache.entry(offset) {
                std::collections::hash_map::Entry::Occupied(e) => {
                    let mut dir = e.into_mut();

                    // Unlink the item
                    if let Some(mut x) = dir.prev {
                        // Not the first item, relink
                        unsafe{x.as_mut().next = dir.next};
                    } else {
                        // The first item already
                        return dir;
                    }

                    if let Some(mut x) = dir.next {
                        // Not the last item, relink
                        unsafe{x.as_mut().prev = dir.prev};
                    } else {
                        // We are the last item and must fixup the tail pointer
                        self.ldir = dir.prev;
                    }

                    // Attach to the start
                    if let Some(mut x) = self.fdir {
                        dir.next = self.fdir;
                        unsafe {x.as_mut().prev = Some(dir.into())};
                    }
                    self.fdir = Some(dir.into());

                    return dir;
                },
                std::collections::hash_map::Entry::Vacant(e) => {
                    let mut slice = BinarySlice::from_read_decompress(&mut self.file, &self.internal_compression, offset, len);
                    let dir = Directory::parse(&mut slice);

                    let mut dir = e.insert(dir);

                    // Attach the new directory to the start of the lru
                    if let Some(mut x) = self.fdir {
                        dir.next = self.fdir;
                        unsafe {x.as_mut().prev = Some(dir.into())};
                    }
                    self.fdir = Some(dir.into());

                    return dir;
                },
            };
        }

        fn load_tile(&mut self, x: u64, y: u64, level: u8) -> Option<Tile> {
            if level > self.max_zoom || level < self.min_zoom {
                return None;
            }

            let tid = coords_to_id(x, y, level);

            let mut offset = self.root_offset;
            let mut len = self.root_len;
            loop {
                let dir = self.read_directory(offset, len);

                match dir.find_entry(tid) {
                    DEntry::Tile(pval, runlength, eoffset, elen) => {
                        if pval + runlength <= tid {
                            return None;
                        }
                        assert!(eoffset + (elen as u64) < self.tile_len as u64);

                        let mut tile_slice = BinarySlice::from_read_decompress(&mut self.file, &self.tile_compression, eoffset + self.tile_offset, elen);
                        return Some(compile_tile(x, y, level, &mut tile_slice).unwrap());
                    },
                    DEntry::Leaf(eoffset, elen) => {
                        assert!(eoffset + (elen as u64) < self.leaf_len as u64);
                        offset = eoffset + self.leaf_offset;
                        len = elen;
                    },
                }
            }
        }
    }

    pub struct LiveTiles {
        pub source: File,

        pub active: HashMap<u64, Tile>,
        pub visible: Vec<u64>,
    }

    impl LiveTiles {
        pub fn new(source: File) -> Self {
            return LiveTiles {
                source,
                active: HashMap::new(),
                visible: vec![],
            };
        }

        pub fn retrieve_visible_tiles(&mut self, left: u64, top: u64, right: u64, bottom: u64, level: u8) {
            let mut keys: HashSet<u64> = self.active.keys().cloned().collect();

            for x in left.max(0)..right.max(0) {
                for y in top.max(0)..bottom.max(0) {
                    let id = coords_to_id(x, y, level);
                    keys.remove(&id);
                    if !self.active.contains_key(&id) {
                        if let Some(ptile) = self.source.load_tile(x, y, level) {
                            self.active.insert(id, ptile);
                        }
                    }

                }
            }

            for k in keys {
                self.active.remove(&k);
            }

            self.visible.clear();
            for x in left.max(0)..right.max(0) {
                for y in top.max(0)..bottom.max(0) {
                    let id = coords_to_id(x, y, level);
                    if self.active.contains_key(&id) {
                        self.visible.push(id);
                    }
                }
            }

        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn tileid_outermost_layer() {
            let id = coords_to_id(0, 0, 0);
            assert_eq!(id, 0);
        }

        #[test]
        fn tileid_first_tile_under_outermost() {
            let id = coords_to_id(0, 0, 1);
            assert_eq!(id, 1);
        }

        #[test]
        fn tileid_third_tile_under_outermost() {
            let id = coords_to_id(1, 1, 1);
            assert_eq!(id, 3);
        }
    }
}
