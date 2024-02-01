use std::io::Read;
use std::io::Seek;
use std::rc::Rc;
use flate2::bufread::GzDecoder;

#[derive(Debug)]
pub enum Compression {
    None,
    GZip,
    Brotli,
    Zstd,
}

pub struct BinarySlice {
    data: Rc<Vec<u8>>,
    pub cursor: usize,
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

impl Clone for BinarySlice {
    fn clone(&self) -> Self {
        return BinarySlice {
            data: self.data.clone(),
            cursor: self.cursor,
        }
    }
}

impl BinarySlice {
    pub fn from_u8slice<const N: usize>(read: [u8; N], start: usize) -> Self {
        let data = read.to_vec();

        return BinarySlice {
            data: Rc::new(data),
            cursor: start,
        };
    }

    pub fn from_read<T: Read + Seek>(read: &mut T, start: u64, len: usize) -> Self {
        let mut data = Vec::with_capacity(len);

        read.seek(std::io::SeekFrom::Start(start)).unwrap();
        data.resize(len, 0);
        read.read_exact(&mut data).unwrap();

        return BinarySlice {
            data: Rc::new(data),
            cursor: 0,
        };
    }

    pub fn from_read_decompress<T: Read + Seek>(read: &mut T, compression: &Compression, start: u64, len: usize) -> Self {
        let mut buf = Vec::with_capacity(len);
        read.seek(std::io::SeekFrom::Start(start)).unwrap();
        buf.resize(len, 0);
        read.read_exact(&mut buf).unwrap();

        let mut data = Vec::new();

        match compression {
            Compression::GZip => {
                let mut reader = GzDecoder::new(buf.as_slice());
                reader.read_to_end(&mut data).unwrap();
            }
            Compression::None => panic!(),
            Compression::Brotli => panic!(),
            Compression::Zstd => panic!(),
        }

        return BinarySlice {
            data: Rc::new(data),
            cursor: 0,
        };
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

