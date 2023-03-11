use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;

fn skip<T: Read + Seek>(reader: &mut T, typeid: u64) -> std::io::Result<()> {
    match typeid {
        0 => { read_var_int(reader)?; },
        1 => { reader.seek(SeekFrom::Current(8))?; },
        2 => {
            let len = read_var_int(reader)?;
            reader.seek(SeekFrom::Current(len as i64))?;
        },
        5 => { reader.seek(SeekFrom::Current(4))?; },
        _ => { panic!(); },
    };

    return Ok(());
}

fn read_var_int<T: Read>(reader: &mut T) -> std::io::Result<u64> {
    let mut value: u64 = 0;

    let mut byte: [u8; 1] = [0];
    let mut i = 0;
    loop {
        reader.read_exact(&mut byte)?;

        value |= (byte[0] as u64 & 0x7F) << i;

        i += 7;
        if byte[0] & 0x80 == 0 {
            break;
        }
    }

    return Ok(value);
}

fn read_zig<T: Read>(reader: &mut T) -> std::io::Result<i64> {
    let value = read_var_int(reader)?;
    // If the bottom bit is set flip will be all 1's, if it's unset it will be all 0's
    let flip = -((value & 1) as i64) as u64;
    let signed = ((value >> 1) ^ flip) as i64;
    dbg!(value);
    return Ok(signed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn var_int_1() {
        let data = [0b00000001];
        let value = read_var_int(&mut data.as_ref()).unwrap();

        assert_eq!(value, 1);
    }

    #[test]
    fn var_int_150() {
        let data = [0b10010110, 0b00000001];
        let value = read_var_int(&mut data.as_ref()).unwrap();

        assert_eq!(value, 150);
    }

    #[test]
    fn zig_0() {
        let data = [0b00000000];
        let value = read_zig(&mut data.as_ref()).unwrap();

        assert_eq!(value, 0);
    }

    #[test]
    fn zig_1() {
        let data = [0b00000010];
        let value = read_zig(&mut data.as_ref()).unwrap();

        assert_eq!(value, 1);
    }

    #[test]
    fn zig_neg1() {
        let data = [0b00000001];
        let value = read_zig(&mut data.as_ref()).unwrap();

        assert_eq!(value, -1);
    }

    #[test]
    fn zig_max() {
        let data = [0b11111110, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b00000001];
        let value = read_zig(&mut data.as_ref()).unwrap();

        assert_eq!(value, 0x7FFFFFFFFFFFFFFF);
    }

    #[test]
    fn zig_neg_max() {
        let data = [0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b11111111, 0b00000001];
        let value = read_zig(&mut data.as_ref()).unwrap();

        assert_eq!(value, -0x8000000000000000);
    }
}
