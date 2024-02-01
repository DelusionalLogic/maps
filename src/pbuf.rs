use std::io::Read;

use crate::io::BinarySlice;

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

pub struct Message {
    reader: BinarySlice,
    stack: Vec<usize>,
}

impl std::fmt::Debug for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return f.debug_struct("Message")
            .finish_non_exhaustive();
    }
}

pub struct PackedField<'a, T> {
    message: &'a mut Message,
    fun: fn(&mut Message) -> T,

    end: usize,
}

impl<'a, T> Iterator for PackedField<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.message.reader.cursor == self.end {
            return None;
        }
        debug_assert!(self.message.reader.cursor < self.end);

        return Some((self.fun)(self.message));
    }
}

impl Clone for Message {
    fn clone(&self) -> Self {
        return Message {
            reader: self.reader.clone(),
            stack: self.stack.clone(),
        }
    }
}

impl Message {
    pub fn new(reader: BinarySlice) -> Self {
        return Self {
            reader,
            stack: vec![],
        }
    }

    pub fn next(&mut self) -> Result<TypeAndTag> {
        if let Some(x) = self.stack.last() {
            if self.reader.cursor == *x {
                self.stack.pop();
                return Ok(TypeAndTag::eom());
            }
            debug_assert!(self.reader.cursor < *x);
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

    pub fn read_packed_var_int<'b>(&'b mut self) -> Result<PackedField<'b, Result<u64>>> {
        let len = self.read_var_int()?;
        let pos = self.reader.cursor;
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
        self.stack.push(len + self.reader.cursor);
        return Ok(());
    }

    pub fn exit_message(&mut self) -> Result<()> {
        if let Some(x) = self.stack.pop() {
            self.fastforward(x - self.reader.cursor)?;
        } else {
            panic!("Not in message");
        }

        return Ok(());
    }

    fn fastforward(&mut self, bytes: usize) -> Result<()> {
        self.reader.cursor += bytes;
        return Ok(());
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
        let mut reader = BinarySlice::from_u8slice([0b00000001, 0b00000001], 0);

        let mut buf = [0; 1];
        reader.read(&mut buf).unwrap();
        assert_eq!(reader.cursor, 1);

        reader.read(&mut buf).unwrap();
        assert_eq!(reader.cursor, 2);
    }

    #[test]
    fn var_int_1() {
        let data = BinarySlice::from_u8slice([0b00000001], 0);
        let mut msg = Message::new(data);
        let value = msg.read_var_int().unwrap();

        assert_eq!(value, 1);
    }

    #[test]
    fn var_int_150() {
        let data = BinarySlice::from_u8slice([0b10010110, 0b00000001], 0);
        let mut msg = Message::new(data);
        let value = msg.read_var_int().unwrap();

        assert_eq!(value, 150);
    }

    #[test]
    fn zig_0() {
        let data = BinarySlice::from_u8slice([0b00000000], 0);
        let mut msg = Message::new(data);
        let value = msg.read_zig().unwrap();

        assert_eq!(value, 0);
    }

    #[test]
    fn zig_1() {
        let data = BinarySlice::from_u8slice([0b00000010], 0);
        let mut msg = Message::new(data);
        let value = msg.read_zig().unwrap();

        assert_eq!(value, 1);
    }

    #[test]
    fn zig_neg1() {
        let data = BinarySlice::from_u8slice([0b00000001], 0);
        let mut msg = Message::new(data);
        let value = msg.read_zig().unwrap();

        assert_eq!(value, -1);
    }

    #[test]
    fn zig_max() {
        let data = BinarySlice::from_u8slice(
            [0b11111110, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b00000001],
            0
        );
        let mut msg = Message::new(data);
        let value = msg.read_zig().unwrap();

        assert_eq!(value, 0x7FFFFFFFFFFFFFFF);
    }

    #[test]
    fn zig_neg_max() {
        let data = BinarySlice::from_u8slice(
            [0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b00000001],
            0
        );
        let mut msg = Message::new(data);
        let value = msg.read_zig().unwrap();

        assert_eq!(value, -0x8000000000000000);
    }

    #[test]
    fn skip_i64() {
        let data = BinarySlice::from_u8slice(
            [0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000001],
            0
        );
        let mut msg = Message::new(data);
        msg.skip(WireType::I64).unwrap();
        let value = msg.read_var_int().unwrap();

        assert_eq!(value, 1);
    }

    #[test]
    #[ignore]
    fn skip_i64_eof() {
        let data = BinarySlice::from_u8slice(
            [0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000001],
            0
        );
        let mut msg = Message::new(data);
        msg.skip(WireType::I64).unwrap();
        let value = msg.read_var_int().unwrap();

        assert_eq!(value, 1);
    }

    #[test]
    fn skip_i32() {
        let data = BinarySlice::from_u8slice(
            [0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000001],
            0
        );
        let mut msg = Message::new(data);
        msg.skip(WireType::I32).unwrap();
        let value = msg.read_var_int().unwrap();

        assert_eq!(value, 1);
    }

    #[test]
    fn skip_len() {
        let data = BinarySlice::from_u8slice(
            [0b00000001, 0b00000000, 0b00000001],
            0
        );
        let mut msg = Message::new(data);
        msg.skip(WireType::Len).unwrap();
        let value = msg.read_var_int().unwrap();

        assert_eq!(value, 1);
    }

    #[test]
    fn field_1_varint() {
        let data = BinarySlice::from_u8slice(
            [0b00001000, 0b00000000],
            0
        );
        let mut msg = Message::new(data);
        let field = msg.next().unwrap();

        assert_eq!(field.wtype, WireType::VarInt);
        assert_eq!(field.tag, 1);
    }

    #[test]
    fn field_2_varint() {
        let data = BinarySlice::from_u8slice(
            [0b00001000, 0b00000000, 0b00010000, 0b00000000],
            0
        );
        let mut msg = Message::new(data);

        // skip first field
        let field = msg.next().unwrap();
        msg.skip(field.wtype).unwrap();

        let field = msg.next().unwrap();

        assert_eq!(field.wtype, WireType::VarInt);
        assert_eq!(field.tag, 2);
    }

    #[test]
    fn field_eof() {
        let data = BinarySlice::from_u8slice(
            [],
            0
        );
        let mut msg = Message::new(data);

        let err = msg.next();
        assert!(matches!(err.unwrap_err(), Error::EOF()))
    }

    #[test]
    fn submessage() {
        let data = BinarySlice::from_u8slice(
            [0b00001010, 0b00000010, 0b00001000, 0b00000011],
            0
        );
        let mut msg = Message::new(data);

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
        let data = BinarySlice::from_u8slice(
            [0b00001010, 0b00000010, 0b00001000, 0b00000011, 0b00010000, 0b00000000],
            0
        );
        let mut msg = Message::new(data);

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
