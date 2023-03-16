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

    #[derive(PartialEq)]
    enum State {
        Field,
        Tag,
        Value,
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

    pub fn read_zig<T: Read>(reader: &mut T) -> Result<i64> {
        let value = read_var_int(reader)?;
        // If the bottom bit is set flip will be all 1's, if it's unset it will be all 0's
        let flip = -((value & 1) as i64) as u64;
        let signed = ((value >> 1) ^ flip) as i64;
        dbg!(value);
        return Ok(signed);
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
    x: f32,
    y: f32,
    norm_x: f32,
    norm_y: f32,
    sign: i8,
}

#[derive(Debug)]
pub struct LineStart {
    pos: usize,
}

pub fn read_one_linestring<T: Read>(reader: &mut pbuf::Message<T>) -> pbuf::Result<(Vec<LineStart>, Vec<LineVert>)> {
    let mut start = vec![];
    let mut vert : Vec<LineVert> = vec![];

    println!("Reading!");
    while let Ok(field) = reader.next() {
        match field {
            pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 3} => {
                println!("Layer!");
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
                            let name = reader.read_string()?;
                            if name != "roads" {
                                reader.exit_message()?;
                                break 'layer;
                            }

                            seen_name = true;
                        }
                        pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 2} => { 
                            assert!(seen_name);
                            reader.enter_message()?;
                            let mut seen_type = false;
                            'feature: while let Ok(field) = reader.next() {
                                match field {
                                    pbuf::TypeAndTag{wtype: pbuf::WireType::VarInt, tag: 3} => { 
                                        if reader.read_var_int()? != 2 {
                                            reader.exit_message()?;
                                            break 'feature;
                                        }

                                        seen_type = true;
                                    },
                                    pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 4} => { 
                                        if !seen_type {
                                            reader.exit_message()?;
                                            break 'feature;
                                        }

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

                                                    let mut normal = Vector::new(y, -(x));
                                                    normal.unit();

                                                    let bend_norm_x;
                                                    let bend_norm_y;
                                                    if i != 0 {
                                                        let len = vert.len();

                                                        let last_normx = vert[len-2].norm_x;
                                                        let last_normy = vert[len-2].norm_y;

                                                        let mut join_x = last_normx + normal.x;
                                                        let mut join_y = last_normy + normal.y;
                                                        let join_len = f32::sqrt(f32::powi(join_x, 2) + f32::powi(join_y, 2));
                                                        join_x /= join_len;
                                                        join_y /= join_len;

                                                        let cos_angle = normal.x * join_x + normal.y * join_y;
                                                        let l = 1.0 / cos_angle;

                                                        bend_norm_x = join_x * l;
                                                        bend_norm_y = join_y * l;

                                                        vert[len-2].norm_x = bend_norm_x;
                                                        vert[len-2].norm_y = bend_norm_y;
                                                        vert[len-3].norm_x = -bend_norm_x;
                                                        vert[len-3].norm_y = -bend_norm_y;
                                                        vert[len-4].norm_x = bend_norm_x;
                                                        vert[len-4].norm_y = bend_norm_y;
                                                    } else {
                                                        bend_norm_x = normal.x;
                                                        bend_norm_y = normal.y;
                                                    }

                                                    // Now construct the tris
                                                    vert.push(LineVert { x: cx, y: cy, norm_x:  bend_norm_x, norm_y:  bend_norm_y, sign: 1 });
                                                    vert.push(LineVert { x: cx, y: cy, norm_x: -bend_norm_x, norm_y: -bend_norm_y, sign: -1 });
                                                    vert.push(LineVert { x: px, y: py, norm_x:  normal.x, norm_y:  normal.y, sign: 1 });

                                                    vert.push(LineVert { x: px, y: py, norm_x: -normal.x, norm_y: -normal.y, sign: -1 });
                                                    vert.push(LineVert { x: px, y: py, norm_x:  normal.x, norm_y:  normal.y, sign: 1 });
                                                    vert.push(LineVert { x: cx, y: cy, norm_x: -bend_norm_x, norm_y: -bend_norm_y, sign: -1 });

                                                    cx = px;
                                                    cy = py;
                                                }
                                            }
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

    return Ok((start, vert));
}
