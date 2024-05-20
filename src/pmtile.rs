use crate::io::{
    BinarySlice,
    Compression
};
use core::ptr::NonNull;
use std::collections::HashMap;

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

pub enum DEntry {
    Leaf(u64, usize),
    Tile(u64, u64, u64, usize),
}

pub struct Directory {
    id: Vec<u64>,
    runlength: Vec<u64>,
    len: Vec<usize>,
    offset: Vec<u64>,

    pub next: Option<NonNull<Directory>>,
    pub prev: Option<NonNull<Directory>>,
}

impl Directory {
    pub fn parse(slice: &mut BinarySlice) -> Self {
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

    pub fn find_entry(&self, id: u64) -> DEntry {
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

fn compression(i: u8) -> Compression {
    match i {
        1 => Compression::None,
        2 => Compression::GZip,
        3 => Compression::Brotli,
        4 => Compression::Zstd,
        _ => panic!("Unknown compression"),
    }
}

pub struct PMTile {
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

impl<'a> PMTile {
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

        let internal_compression = compression(slice.u8());
        let tile_compression = compression(slice.u8());

        assert!(slice.u8() == 1); // Tile type (MVT)

        let min_zoom = slice.u8();
        let max_zoom = slice.u8();

        return PMTile {
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
                let dir = e.into_mut();

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

    pub fn load_tile(&mut self, x: u64, y: u64, level: u8) -> Option<BinarySlice> {
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
                    assert!(eoffset + (elen as u64) <= self.tile_len as u64);

                    let tile_slice = BinarySlice::from_read_decompress(&mut self.file, &self.tile_compression, eoffset + self.tile_offset, elen);
                    return Some(tile_slice);
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


#[cfg(test)]
mod tests {
    use crate::{
        math::Vector2,
        map::LineBuilder
    };

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

    #[test]
    fn build_straight_line() {
        let mut line = LineBuilder::new();

        let v1 = Vector2::new(0.0, 0.0);
        let v2 = Vector2::new(0.0, 1.0);
        let v3 = Vector2::new(0.0, 2.0);

        line.add_point(v1, v2, false);
        line.add_point(v2, v3, true);

        assert_eq!(12, line.verts.len());
        {
            let vx = line.verts[0].x;
            let vy = line.verts[0].y;
            assert_eq!(0.0, vx);
            assert_eq!(0.0, vy);
            let nx = line.verts[0].norm_x;
            let ny = line.verts[0].norm_y;
            assert_eq!(1.0, nx);
            assert_eq!(0.0, ny);
        }
        {
            let vx = line.verts[1].x;
            let vy = line.verts[1].y;
            assert_eq!(0.0, vx);
            assert_eq!(0.0, vy);
            let nx = line.verts[1].norm_x;
            let ny = line.verts[1].norm_y;
            assert_eq!(-1.0, nx);
            assert_eq!(0.0, ny);
        }
        {
            let vx = line.verts[2].x;
            let vy = line.verts[2].y;
            assert_eq!(0.0, vx);
            assert_eq!(1.0, vy);
            let nx = line.verts[2].norm_x;
            let ny = line.verts[2].norm_y;
            assert_eq!(1.0, nx);
            assert_eq!(0.0, ny);
        }
        {
            let vx = line.verts[3].x;
            let vy = line.verts[3].y;
            assert_eq!(0.0, vx);
            assert_eq!(1.0, vy);
            let nx = line.verts[3].norm_x;
            let ny = line.verts[3].norm_y;
            assert_eq!(-1.0, nx);
            assert_eq!(0.0, ny);
        }
        {
            let vx = line.verts[4].x;
            let vy = line.verts[4].y;
            assert_eq!(0.0, vx);
            assert_eq!(1.0, vy);
            let nx = line.verts[4].norm_x;
            let ny = line.verts[4].norm_y;
            assert_eq!(1.0, nx);
            assert_eq!(0.0, ny);
        }
        {
            let vx = line.verts[5].x;
            let vy = line.verts[5].y;
            assert_eq!(0.0, vx);
            assert_eq!(0.0, vy);
            let nx = line.verts[5].norm_x;
            let ny = line.verts[5].norm_y;
            assert_eq!(-1.0, nx);
            assert_eq!(0.0, ny);
        }
        {
            let vx = line.verts[6].x;
            let vy = line.verts[6].y;
            assert_eq!(0.0, vx);
            assert_eq!(1.0, vy);
            let nx = line.verts[6].norm_x;
            let ny = line.verts[6].norm_y;
            assert_eq!(1.0, nx);
            assert_eq!(0.0, ny);
        }
        {
            let vx = line.verts[7].x;
            let vy = line.verts[7].y;
            assert_eq!(0.0, vx);
            assert_eq!(1.0, vy);
            let nx = line.verts[7].norm_x;
            let ny = line.verts[7].norm_y;
            assert_eq!(-1.0, nx);
            assert_eq!(0.0, ny);
        }
        {
            let vx = line.verts[8].x;
            let vy = line.verts[8].y;
            assert_eq!(0.0, vx);
            assert_eq!(2.0, vy);
            let nx = line.verts[8].norm_x;
            let ny = line.verts[8].norm_y;
            assert_eq!(1.0, nx);
            assert_eq!(0.0, ny);
        }
        {
            let vx = line.verts[9].x;
            let vy = line.verts[9].y;
            assert_eq!(0.0, vx);
            assert_eq!(2.0, vy);
            let nx = line.verts[9].norm_x;
            let ny = line.verts[9].norm_y;
            assert_eq!(-1.0, nx);
            assert_eq!(0.0, ny);
        }
        {
            let vx = line.verts[10].x;
            let vy = line.verts[10].y;
            assert_eq!(0.0, vx);
            assert_eq!(2.0, vy);
            let nx = line.verts[10].norm_x;
            let ny = line.verts[10].norm_y;
            assert_eq!(1.0, nx);
            assert_eq!(0.0, ny);
        }
        {
            let vx = line.verts[11].x;
            let vy = line.verts[11].y;
            assert_eq!(0.0, vx);
            assert_eq!(1.0, vy);
            let nx = line.verts[11].norm_x;
            let ny = line.verts[11].norm_y;
            assert_eq!(-1.0, nx);
            assert_eq!(0.0, ny);
        }
    }
}
