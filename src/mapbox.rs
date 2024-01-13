use crate::math::Vector2;

pub mod pbuf {
    use std::io::Read;

    use super::pmtile::BinarySlice;

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
}

#[derive(Clone,Copy,Debug)]
#[repr(packed)]
pub struct GlVert {
    pub x: f32,
    pub y: f32,
    pub norm_x: f32,
    pub norm_y: f32,
    pub sign: i8,
}

pub struct PointGeom {
    pub data: Vec<Vector2<f32>>,
    pub name: Vec<Option<usize>>,
}

#[derive(Debug,Clone)]
pub struct LineStart {
    pub pos: usize,
}

pub struct LineGeom {
    pub start: Vec<LineStart>,
    pub name: Vec<Option<usize>>,
    pub data: Vec<Vector2<f32>>,
}

impl LineGeom {
    pub fn new() -> Self {
        return LineGeom{
            start: Vec::new(),
            name: Vec::new(),
            data: Vec::new(),
        }
    }
}

impl PointGeom {
    pub fn new() -> Self {
        return Self {
            data: Vec::new(),
            name: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct PolyGeom {
    pub start: Vec<LineStart>,
    pub data: Vec<GlVert>,
}

pub struct RawTile {
    pub buildings: PolyGeom,
    pub earth: PolyGeom,
    pub water: PolyGeom,
    pub areas: PolyGeom,
    pub farmland: PolyGeom,

    pub roads: LineGeom,
    pub highways: LineGeom,
    pub major: LineGeom,
    pub medium: LineGeom,
    pub minor: LineGeom,

    pub points: PointGeom,

    pub strings: Vec<String>,
}

pub fn read_one_linestring(reader: &mut pbuf::Message) -> pbuf::Result<RawTile> {
    let layer_names = [
        "roads",
        "earth",
        "water",
        "landuse",
        "buildings",
        "places",
    ];

    fn scan_for_layers<const COUNT: usize>(reader: &mut pbuf::Message, layer_names: [&str; COUNT]) -> pbuf::Result<[Option<pbuf::Message>; COUNT]> {
        let mut layer_messages = Vec::with_capacity(COUNT);
        for _ in 0..COUNT {
            layer_messages.push(None);
        }

        while let Ok(field) = reader.next() {
            match field {
                pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 3} => {
                    reader.enter_message()?;
                    let mut name = None;
                    let layer_message = reader.clone();
                    while let Ok(field) = reader.next() {
                        match field {
                            pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                            pbuf::TypeAndTag{wtype: pbuf::WireType::VarInt, tag: 15} => {
                                let version = reader.read_var_int()?;
                                assert_eq!(version, 2);
                            }
                            pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 1} => {
                                name = Some(reader.read_string()?);
                            }
                            x => { reader.skip(x.wtype)?; },
                        }
                    }

                    if let Some(name) = name {
                        for (i, lname) in layer_names.iter().enumerate() {
                            if name.as_str() == *lname {
                                layer_messages[i] = Some(layer_message);
                                break;
                            }
                        }
                    }
                },
                x => { reader.skip(x.wtype)?; },
            }
        }

        return Ok(layer_messages.try_into().unwrap());
    }

    struct Filter<const N: usize> {
        keys: [Option<u64>; N],
        values: [Option<u64>; N],
        groups: [u64; N],
    }

    impl<const N: usize> Filter<N> {
        pub fn scan(reader: &mut pbuf::Message, keys: [&str; N], values: [&str; N], groups: [u64; N]) ->pbuf::Result<Self> {
            assert!(keys.len() == values.len());
            assert!(keys.len() == groups.len());

            let (keys, values) = scan_for_keys_and_values(reader, &keys.to_vec(), &values.to_vec())?;

            return Ok(Filter{
                keys: keys.try_into().expect("Array of scanned keys did not match input"),
                values: values.try_into().expect("Array of scanned values did not match input"),
                groups,
            });
        }

        pub fn sieve_features(&self, reader: &mut pbuf::Message) -> pbuf::Result<(Vec<pbuf::Message>, [Vec<pbuf::Message>; N])> {
            // @SPEED: There may be some way to read the buffer linearly here instead of fucking up the
            // ordering. This is fine for now.
            let mut other = vec![];
            let mut features = std::array::from_fn(|_| vec![]);

            {
                while let Ok(field) = reader.next() {
                    match field {
                        pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                        pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 2} => {
                            reader.enter_message()?;
                            let feature_ptr = reader.clone();
                            let mut bucket = &mut other;
                            while let Ok(field) = reader.next() {
                                match field {
                                    pbuf::TypeAndTag{wtype: pbuf::WireType::VarInt, tag: 3} => {
                                        let geo_type = reader.read_var_int()?;
                                        assert!(geo_type == 3);
                                    },
                                    pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 2} => {
                                        let mut it = reader.read_packed_var_int()?;
                                        while let Some(tag) = it.next() {
                                            let key = tag?;
                                            let value = it.next().unwrap()?;

                                            for i in 0..self.keys.len() {
                                                if self.keys[i] == Some(key) && self.values[i] == Some(value) {
                                                    let group = self.groups[i];
                                                    bucket = &mut features[group as usize];
                                                }
                                            }
                                        }
                                    },
                                    pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                                    x => { reader.skip(x.wtype)?; },
                                }
                            }
                            bucket.push(feature_ptr);
                        }
                        x => { reader.skip(x.wtype)?; },
                    }
                }

                return Ok((other, features));
            }
        }
    }

    fn lookup_values(reader: &mut pbuf::Message, values: &Vec<u64>, order: &Vec<usize>) -> pbuf::Result<(Vec<usize>, Vec<String>)> {
        let mut oit = order.iter().peekable();

        let mut index = Vec::with_capacity(values.len());
        let mut value_ids = Vec::with_capacity(values.len());
        for _ in 0..values.len() {
            index.push(0);
        }

        let mut cursor = 0;
        while let Some(i) = oit.next() {
            // We skipped all the Nones
            let value = values[*i];

            while cursor < value {
                let field = reader.next()?;
                match field {
                    pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { panic!(); }
                    pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 4} => {
                        reader.skip(pbuf::WireType::Len)?;
                        cursor += 1;
                    },
                    x => { reader.skip(x.wtype)?; }
                }
            }

            while let Ok(field) = reader.next() {
                match field {
                    pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { panic!(); }
                    pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 4} => {
                        reader.enter_message()?;
                        match reader.next().unwrap() {
                            pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 1} => {
                                index[*i] = value_ids.len();
                                value_ids.push(reader.read_string()?);
                            }
                            _ => { panic!("Incorrect name value type"); },
                        }

                        reader.exit_message()?;
                        break;
                    },
                    x => { reader.skip(x.wtype)?; },
                }
            }
            cursor += 1;

            // Remap anything pointing to the same index
            while let Some(i2) = oit.peek() {
                let n_value = values[**i2];
                if n_value == value {
                    index[**i2] = index[*i];
                    oit.next();
                } else {
                    break;
                }
            }
        }

        return Ok((index, value_ids));
    }

    fn scan_for_keys_and_values(reader: &mut pbuf::Message, keys: &Vec<&str>, values: &Vec<&str>) -> pbuf::Result<(Vec<Option<u64>>, Vec<Option<u64>>)> {
        let mut key_count = 0;
        let mut value_count = 0;

        let mut key_ids = Vec::with_capacity(keys.len());
        let mut value_ids = Vec::with_capacity(values.len());
        for _ in 0..keys.len() {
            key_ids.push(None);
        }
        for _ in 0..values.len() {
            value_ids.push(None);
        }

        while let Ok(field) = reader.next() {
            match field {
                pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 3} => {
                    let key = reader.read_string()?;
                    for (i, hey) in keys.iter().enumerate() {
                        if *hey == &key {
                            assert!(key_ids[i].is_none());
                            key_ids[i] = Some(key_count);
                        }
                    }
                    key_count += 1;
                }
                pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 4} => {
                    reader.enter_message()?;
                    while let Ok(field) = reader.next() {
                        match field {
                            pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                            pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 1} => {
                                let value = reader.read_string()?;
                                for (i, hey) in values.iter().enumerate() {
                                    if *hey == &value {
                                        assert!(value_ids[i].is_none());
                                        value_ids[i] = Some(value_count);
                                    }
                                }
                            }
                            x => { reader.skip(x.wtype)?; },
                        }
                    }
                    value_count += 1;
                },
                x => { reader.skip(x.wtype)?; },
            }
        }

        return Ok((key_ids, value_ids));
    }

    fn read_road_layer(reader: &mut pbuf::Message, tile: &mut RawTile) -> pbuf::Result<()> {
        // Since the ordering of the fields in the "messages" aren't well defined this end up being
        // way more difficult than it really should be. First we find the key and value indexes of
        // the keys/values we care about, then we have to do a double pass on all the features
        // first segment them according to the tags, and then read them out in the second pass.
        let key_id;
        let name_key_id;
        let highway_value_id;
        let major_value_id;
        let medium_value_id;
        let minor_value_id;

        {
            let (key_ids, value_ids) = scan_for_keys_and_values(
                &mut reader.clone(),
                &vec!["pmap:kind", "name"],
                &vec!["highway", "major_road", "medium_road", "minor_road"],
            )?;

            key_id = key_ids[0];
            name_key_id = key_ids[1];

            highway_value_id = value_ids[0];
            major_value_id = value_ids[1];
            medium_value_id = value_ids[2];
            minor_value_id = value_ids[3];
        }

        // @SPEED: There may be some way to read the buffer linearly here instead of fucking up the
        // ordering. This is fine for now.
        let mut highway_features = vec![];
        let mut major_features = vec![];
        let mut medium_features = vec![];
        let mut minor_features = vec![];
        let mut other_features = vec![];

        let mut name_reader = reader.clone();
        let mut name_index = Vec::new();

        {
            while let Ok(field) = reader.next() {
                match field {
                    pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                    pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 2} => {
                        reader.enter_message()?;
                        let feature_ptr = reader.clone();
                        let mut bucket = &mut other_features;
                        let mut name_value = None;
                        while let Ok(field) = reader.next() {
                            match field {
                                pbuf::TypeAndTag{wtype: pbuf::WireType::VarInt, tag: 3} => {
                                    let geo_type = reader.read_var_int()?;
                                    assert!(geo_type == 2);
                                },
                                pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 2} => {
                                    let mut it = reader.read_packed_var_int()?;
                                    while let Some(tag) = it.next() {
                                        let key = tag?;
                                        let value = it.next().unwrap()?;

                                        if Some(key) == key_id {
                                            if Some(value) == highway_value_id {
                                                bucket = &mut highway_features;
                                            } else if Some(value) == major_value_id {
                                                bucket = &mut major_features;
                                            } else if Some(value) == medium_value_id {
                                                bucket = &mut medium_features;
                                            } else if Some(value) == minor_value_id {
                                                bucket = &mut minor_features;
                                            }
                                        } else if Some(key) == name_key_id {
                                            name_value = Some(value);
                                        }
                                    }
                                },
                                pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                                x => { reader.skip(x.wtype)?; },
                            }
                        }
                        let slot = if name_value.is_some() {
                            name_index.push(name_value.unwrap());
                            Some(name_index.len()-1)
                        } else {
                            None
                        };
                        bucket.push((feature_ptr, slot));
                    }
                    x => { reader.skip(x.wtype)?; },
                }
            }
        }

        let mut index : Vec<usize> = (0..name_index.len()).collect();
        index.sort_by_key(|x| name_index[*x]);
        let (index, mut names) = lookup_values(&mut name_reader, &name_index, &index).unwrap();
        let offset = tile.strings.len();

        tile.strings.append(&mut names);

        fn read_geometry(features: Vec<(pbuf::Message, Option<usize>)>, name_index: &Vec<usize>, offset: usize, geom: &mut LineGeom) -> pbuf::Result<()> {

            for (mut reader, namei) in features {
                while let Ok(field) = reader.next() {
                    match field {
                        pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 4} => {
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

                                    geom.start.push(LineStart{pos: geom.data.len()});
                                    // @Speed: Maybe we can avoid this memcpy somehow.
                                    geom.name.push(namei.map(|x| offset + name_index[x]));
                                    geom.data.push(crate::math::Vector2 { x: cx, y: cy });
                                } else if (cmd & 7) == 2 { // LineTo
                                    for _ in 0..(cmd >> 3) {
                                        let x = pbuf::decode_zig(it.next().unwrap().unwrap()) as f32;
                                        let px = cx + x;
                                        let y = pbuf::decode_zig(it.next().unwrap().unwrap()) as f32;
                                        let py = cy + y;

                                        geom.data.push(crate::math::Vector2 { x: px, y: py });

                                        cx = px;
                                        cy = py;
                                    }
                                } else {
                                    panic!("Unknown command");
                                }
                            }
                        }
                        pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                        x => { reader.skip(x.wtype)?; },
                    }
                }
            }
            return Ok(());
        }

        read_geometry(other_features, &index, offset, &mut tile.roads)?;
        read_geometry(highway_features, &index, offset, &mut tile.highways)?;
        read_geometry(major_features, &index, offset, &mut tile.major)?;
        read_geometry(medium_features, &index, offset, &mut tile.medium)?;
        read_geometry(minor_features, &index, offset, &mut tile.minor)?;

        return Ok(());
    }


    fn read_landuse_layer(reader: &mut pbuf::Message) -> pbuf::Result<(PolyGeom, PolyGeom, PolyGeom)> {
        let filter = Filter::scan(
            &mut reader.clone(),
            ["landuse", "amenity", "place", "landuse"],
            ["farmland", "university", "neighbourhood", "residential"],
            [ 0, 1, 1, 1, ],
        )?;

        let (mut other_features, mut features) = filter.sieve_features(reader)?;

        fn read_geometry(features: &mut Vec<pbuf::Message>) -> pbuf::Result<PolyGeom> {
            let mut poly_start : Vec<LineStart> = vec![];
            let mut polys : Vec<GlVert> = vec![];

            for reader in features {
                while let Ok(field) = reader.next() {
                    match field {
                        pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 4} => {
                            let mut cx = 0.0;
                            let mut cy = 0.0;
                            let mut start_x = 0.0;
                            let mut start_y = 0.0;
                            let mut state = 0;
                            let mut it = reader.read_packed_var_int()?;
                            while let Some(cmd) = it.next() {
                                let cmd = cmd?;
                                if (cmd & 7) == 1 { // MoveTo
                                    assert!(state == 0);
                                    state = 1;
                                    let x = pbuf::decode_zig(it.next().unwrap().unwrap());
                                    cx += x as f32;
                                    let y = pbuf::decode_zig(it.next().unwrap().unwrap());
                                    cy += y as f32;

                                    poly_start.push(LineStart{pos: polys.len()});
                                    polys.push(GlVert { x: cx, y: cy, norm_x: 0.0, norm_y: 0.0, sign: 0 });
                                    start_x = cx;
                                    start_y = cy;
                                } else if (cmd & 7) == 2 { // LineTo
                                    assert!(state == 1);
                                    state = 2;
                                    for _ in 0..(cmd >> 3) {
                                        let x = pbuf::decode_zig(it.next().unwrap().unwrap()) as f32;
                                        let px = cx + x;
                                        let y = pbuf::decode_zig(it.next().unwrap().unwrap()) as f32;
                                        let py = cy + y;

                                        polys.push(GlVert { x: px, y: py, norm_x: 0.0, norm_y: 0.0, sign: 1 });

                                        cx = px;
                                        cy = py;
                                    }
                                } else if (cmd & 7) == 7 { // ClosePath
                                    assert!(state == 2);
                                    state = 0;
                                    if cx == start_x && cy == start_y {
                                        // println!("Starting vertex is repeated. This is non-conforming, but we will patch the geometry");
                                        polys.pop();
                                    }
                                } else {
                                    panic!("Unknown command");
                                }
                            }
                        }
                        pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                        x => { reader.skip(x.wtype)?; },
                    }
                }
            }
            return Ok(PolyGeom { start: poly_start, data: polys });
        }

        return Ok((
            read_geometry(&mut other_features)?,
            read_geometry(&mut features[0])?,
            read_geometry(&mut features[1])?,
        ));
    }

    fn read_poly_layer(reader: &mut pbuf::Message) -> pbuf::Result<PolyGeom> {
        let mut poly_start : Vec<LineStart> = vec![];
        let mut polys : Vec<GlVert> = vec![];

        while let Ok(field) = reader.next() {
            match field {
                pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 2} => {
                    reader.enter_message()?;
                    while let Ok(field) = reader.next() {
                        match field {
                            pbuf::TypeAndTag{wtype: pbuf::WireType::VarInt, tag: 3} => {
                                let geo_type = reader.read_var_int()?;
                                assert!(geo_type == 3);
                            },
                            pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 4} => {
                                let mut cx = 0.0;
                                let mut cy = 0.0;
                                let mut start_x = 0.0;
                                let mut start_y = 0.0;
                                let mut state = 0;
                                let mut it = reader.read_packed_var_int()?;
                                while let Some(cmd) = it.next() {
                                    let cmd = cmd?;
                                    if (cmd & 7) == 1 { // MoveTo
                                        assert!(state == 0);
                                        state = 1;
                                        let x = pbuf::decode_zig(it.next().unwrap().unwrap());
                                        cx += x as f32;
                                        let y = pbuf::decode_zig(it.next().unwrap().unwrap());
                                        cy += y as f32;

                                        poly_start.push(LineStart{pos: polys.len()});
                                        polys.push(GlVert { x: cx, y: cy, norm_x: 0.0, norm_y: 0.0, sign: 0 });
                                        start_x = cx;
                                        start_y = cy;
                                    } else if (cmd & 7) == 2 { // LineTo
                                        assert!(state == 1);
                                        state = 2;
                                        for _ in 0..(cmd >> 3) {
                                            let x = pbuf::decode_zig(it.next().unwrap().unwrap()) as f32;
                                            let px = cx + x;
                                            let y = pbuf::decode_zig(it.next().unwrap().unwrap()) as f32;
                                            let py = cy + y;

                                            polys.push(GlVert { x: px, y: py, norm_x: 0.0, norm_y: 0.0, sign: 1 });

                                            cx = px;
                                            cy = py;
                                        }
                                    } else if (cmd & 7) == 7 { // ClosePath
                                        assert!(state == 2);
                                        state = 0;
                                        if cx == start_x && cy == start_y {
                                            // println!("Starting vertex is repeated. This is non-conforming, but we will patch the geometry");
                                            polys.pop();
                                        }
                                    } else {
                                        panic!("Unknown command");
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

        return Ok(PolyGeom{
            start: poly_start,
            data: polys,
        })
    }

    fn read_point_layer(reader: &mut pbuf::Message, tile: &mut RawTile) -> pbuf::Result<()> {
        let geom = &mut tile.points;

        let name_key_id = {
            let (key_ids, _) = scan_for_keys_and_values(
                &mut reader.clone(),
                &vec!["name"],
                &vec![],
            )?;

            key_ids[0]
        };

        let mut name_reader = reader.clone();
        let mut names = Vec::new();

        while let Ok(field) = reader.next() {
            match field {
                pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 2} => {
                    reader.enter_message()?;
                    let mut skip = false;
                    let mut name_value = None;
                    while let Ok(field) = reader.next() {
                        match field {
                            pbuf::TypeAndTag{wtype: pbuf::WireType::VarInt, tag: 3} => {
                                let geo_type = reader.read_var_int()?;
                                assert!(geo_type == 1);
                            },
                            pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 2} => {
                                let mut it = reader.read_packed_var_int()?;
                                while let Some(tag) = it.next() {
                                    let key = tag?;
                                    let value = it.next().unwrap()?;

                                    if Some(key) == name_key_id {
                                        name_value = Some(value);
                                    }
                                }
                            },
                            pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 4} => {
                                let mut state = 0;
                                let mut it = reader.read_packed_var_int()?;
                                while let Some(cmd) = it.next() {
                                    let cmd = cmd?;
                                    if (cmd & 7) == 1 { // MoveTo
                                        assert!(cmd >> 3 == 1);
                                        assert!(state == 0);
                                        state = 1;
                                        let x = pbuf::decode_zig(it.next().unwrap().unwrap()) as f32;
                                        let y = pbuf::decode_zig(it.next().unwrap().unwrap()) as f32;

                                        // Discard any point feature that doesn't lie directly in
                                        // our tile. We don't want points in the overlap region.
                                        // 4096 is the extent of the tile
                                        if x >= 0.0 && x < 4096.0 && y >= 0.0 && y < 4096.0 {
                                            geom.data.push(Vector2::new(x, y));
                                        } else {
                                            skip = true;
                                        }
                                    } else {
                                        panic!("Unknown command");
                                    }
                                }
                            }
                            pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                            x => { reader.skip(x.wtype)?; },
                        }
                    }

                    if !skip {
                        if let Some(x) = name_value {
                            names.push(x);
                        }
                    }

                }
                x => { reader.skip(x.wtype)?; },
            }
        }

        let mut index : Vec<usize> = (0..names.len()).collect();
        index.sort_by_key(|x| names[*x]);
        let (index, mut names) = lookup_values(&mut name_reader, &names, &index).unwrap();
        let offset = tile.strings.len();

        tile.strings.append(&mut names);

        for i in 0..index.len() {
            geom.name.push(Some(offset + index[i]));
        }

        assert!(geom.data.len() == geom.name.len());
        return Ok(());
    }

    let mut layer_messages = scan_for_layers(reader, layer_names)?;

    let earth = if let Some(layer) = &mut layer_messages[1] {
        read_poly_layer(layer)?
    } else {
        PolyGeom {
            start: vec![],
            data: vec![],
        }
    };

    let buildings = if let Some(layer) = &mut layer_messages[4] {
        read_poly_layer(layer)?
    } else {
        PolyGeom {
            start: vec![],
            data: vec![],
        }
    };

    let water = if let Some(layer) = &mut layer_messages[2] {
        read_poly_layer(layer)?
    } else {
        PolyGeom {
            start: vec![],
            data: vec![],
        }
    };

    let (_landuse, farmland, areas) = if let Some(layer) = &mut layer_messages[3] {
        read_landuse_layer(layer)?
    } else {
        (
            PolyGeom { start: vec![], data: vec![] },
            PolyGeom { start: vec![], data: vec![] },
            PolyGeom { start: vec![], data: vec![] },
        )
    };

    let mut tile = RawTile {
        roads: LineGeom::new(),
        highways: LineGeom::new(),
        major: LineGeom::new(),
        medium: LineGeom::new(),
        minor: LineGeom::new(),
        earth,
        buildings,
        water,
        farmland,
        areas,

        points: PointGeom::new(),

        strings: Vec::new(),
    };

    if let Some(layer) = &mut layer_messages[0] {
        read_road_layer(layer, &mut tile)?;
    };

    if let Some(layer) = &mut layer_messages[5] {
        read_point_layer(layer, &mut tile)?;
    }

    return Ok(tile);
}

pub mod pmtile {
    use crate::font::FontMetric;
    use crate::map::compile_tile;
    use crate::math::Transform;
    use crate::math::Vector2;
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::io::Read;
    use std::io::Seek;
    use std::rc::Rc;
    use flate2::bufread::GzDecoder;
    use super::pbuf;
    use core::ptr::NonNull;

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

    pub trait Renderer {
        type Layer;

        fn upload_layer(data: &Vec<crate::mapbox::GlVert>) -> Self::Layer;
        fn upload_multi_layer(vertex_data: &Vec<crate::mapbox::GlVert>, cmd: Vec<RenderCommand>, unlabeled: usize, labels: Vec<Label>) -> Self::Layer;
    }

    pub struct GL { }

    impl Renderer for GL {
        type Layer = GlLayer;

        fn upload_layer(data: &Vec<crate::mapbox::GlVert>) -> GlLayer {
            return Self::upload_multi_layer(data, vec![RenderCommand::Simple(data.len())], 1, Vec::new());
        }

        fn upload_multi_layer(vertex_data: &Vec<crate::mapbox::GlVert>, cmd: Vec<RenderCommand>, unlabeled: usize, labels: Vec<Label>) -> Self::Layer {
            let mut vao = 0;
            unsafe { gl::GenVertexArrays(1, &mut vao) };

            let mut vbo = 0;
            unsafe { gl::GenBuffers(1, &mut vbo) };

            unsafe {
                gl::BindVertexArray(vao);

                gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
                gl::BufferData(gl::ARRAY_BUFFER, (vertex_data.len() * std::mem::size_of::<super::GlVert>()) as _, vertex_data.as_ptr().cast(), gl::STATIC_DRAW);

                gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<super::GlVert>() as i32, 0 as *const _);
                gl::EnableVertexAttribArray(0);
                gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<super::GlVert>() as i32, (2 * std::mem::size_of::<f32>()) as *const _);
                gl::EnableVertexAttribArray(1);
                gl::VertexAttribPointer(2, 1, gl::BYTE, gl::FALSE, std::mem::size_of::<super::GlVert>() as i32, (4 * std::mem::size_of::<f32>()) as *const _);
                gl::EnableVertexAttribArray(2);

                gl::BindBuffer(gl::ARRAY_BUFFER, 0);
                gl::BindVertexArray(0);
            }
            return GlLayer{ vao, vbo, unlabeled, commands: cmd, labels };
        }
    }

    pub struct Label {
        pub rank: u8,

        pub pos: Vector2<f32>,

        pub min: Vector2<f32>,
        pub max: Vector2<f32>,

        pub cmds: usize,

        pub not_before: f32,
    }

    pub enum RenderCommand {
        Simple(usize),
        Target(Vector2<f32>, usize),
        PositionedLetter(char, Vector2<f32>, usize),
    }

    pub struct GlLayer {
        pub vao: u32,
        pub vbo: u32,
        pub unlabeled: usize,
        pub commands: Vec<RenderCommand>,
        pub labels: Vec<Label>,
    }

    pub const LAYERTYPE_MAX: usize = 11;
    #[derive(Clone, Copy)]
    pub enum LayerType {
        EARTH,
        ROADS,
        HIGHWAYS,
        MAJOR,
        MEDIUM,
        MINOR,
        BUILDINGS,
        WATER,
        FARMLAND,
        AREAS,
        POINTS,
    }

    pub struct Tile<R: Renderer> {
        pub tid: u64,
        pub x: u64,
        pub y: u64,
        pub z: u8,
        pub extent: u16,

        pub layers: [Option<R::Layer>; LAYERTYPE_MAX],
    }

    impl Drop for GlLayer {
        fn drop(&mut self) {
            unsafe {
                gl::DeleteVertexArrays(1, &self.vao);
                gl::DeleteBuffers(1, &self.vbo);
            }
        }
    }

    pub enum Ownership<'a, T> {
        Owned(T),
        Unowned(&'a mut T),
    }

    impl <'a, T> std::ops::Deref for Ownership<'a, T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            return match self {
                Ownership::Owned(x) => x,
                Ownership::Unowned(x) => x,
            }
        }
    }

    impl <'a, T> std::ops::DerefMut for Ownership<'a, T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            return match self {
                Ownership::Owned(x) => x,
                Ownership::Unowned(x) => x,
            }
        }
    }

    pub struct LineBuilder<'a> {
        pub verts: Ownership<'a, Vec<super::GlVert>>,
    }

    impl <'a> LineBuilder<'a> {
        pub fn new() -> Self {
            return LineBuilder{
                verts: Ownership::Owned(Vec::new()),
            };
        }

        pub fn new_under(under: &'a mut Vec<super::GlVert>) -> Self {
            return LineBuilder{
                verts: Ownership::Unowned(under),
            };
        }

        pub fn add_point(&mut self, lv: Vector2<f32>, v1: Vector2<f32>, connect_previous: bool, add_point: bool) {
            let cx = lv.x;
            let cy = lv.y;

            let mut ltov = v1.clone();
            ltov -= lv;

            let mut normal = ltov.clone();
            normal.normal();
            normal.unit();

            let bend_norm_x;
            let bend_norm_y;

            if connect_previous {
                let len = self.verts.len();

                let last_normx = self.verts[len-2].norm_x;
                let last_normy = self.verts[len-2].norm_y;

                let mut join_x = last_normx + normal.x;
                let mut join_y = last_normy + normal.y;
                let join_len = f32::sqrt(f32::powi(join_x, 2) + f32::powi(join_y, 2));
                join_x /= join_len;
                join_y /= join_len;

                let cos_angle = normal.x * join_x + normal.y * join_y;
                let l = 1.0 / cos_angle;
                // Don't do a miter for very sharp corners
                if l < 2.0 {
                    bend_norm_x = join_x * l;
                    bend_norm_y = join_y * l;

                    self.verts[len-4].norm_x = bend_norm_x;
                    self.verts[len-4].norm_y = bend_norm_y;
                    self.verts[len-3].norm_x = -bend_norm_x;
                    self.verts[len-3].norm_y = -bend_norm_y;
                    self.verts[len-2].norm_x = bend_norm_x;
                    self.verts[len-2].norm_y = bend_norm_y;
                } else {
                    // @HACK @COMPLETE: Do another type  of join here. Right now it's just
                    // disconnected
                    bend_norm_x = normal.x;
                    bend_norm_y = normal.y;
                }
            } else {
                bend_norm_x = normal.x;
                bend_norm_y = normal.y;
            }

            if add_point {
                // Now construct the tris
                self.verts.push(super::GlVert { x:   cx, y:   cy, norm_x:  bend_norm_x, norm_y:  bend_norm_y, sign: 1 });
                self.verts.push(super::GlVert { x:   cx, y:   cy, norm_x: -bend_norm_x, norm_y: -bend_norm_y, sign: -1 });
                self.verts.push(super::GlVert { x: v1.x, y: v1.y, norm_x:  normal.x, norm_y:  normal.y, sign: 1 });

                self.verts.push(super::GlVert { x: v1.x, y: v1.y, norm_x: -normal.x, norm_y: -normal.y, sign: -1 });
                self.verts.push(super::GlVert { x: v1.x, y: v1.y, norm_x:  normal.x, norm_y:  normal.y, sign: 1 });
                self.verts.push(super::GlVert { x:   cx, y:   cy, norm_x: -bend_norm_x, norm_y: -bend_norm_y, sign: -1 });
            }
        }
    }

    static mut TILE_NUM  : u64 = 0;
    pub fn placeholder_tile(x: u64, y: u64, z: u8) -> Tile<GL> {
        let mut line = LineBuilder::new();

        let mut lv = Vector2::new(0.0, 0.0);

        {
            let nv = Vector2::new(0.0, 4096.0);
            line.add_point(lv, nv, false, true);
            lv = nv;
        }

        {
            let nv = Vector2::new(4096.0, 4096.0);
            line.add_point(lv, nv, true, true);
            lv = nv;
        }

        {
            let nv = Vector2::new(4096.0, 0.0);
            line.add_point(lv, nv, true, true);
            lv = nv;
        }

        {
            let nv = Vector2::new(0.0, 0.0);
            line.add_point(lv, nv, true, true);
        }

        let mut vao = 0;
        unsafe { gl::GenVertexArrays(1, &mut vao) };

        let mut vbo = 0;
        unsafe { gl::GenBuffers(1, &mut vbo) };

        unsafe {
            gl::BindVertexArray(vao);

            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(gl::ARRAY_BUFFER, (line.verts.len() * std::mem::size_of::<super::GlVert>()) as _, line.verts.as_ptr().cast(), gl::STATIC_DRAW);

            gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<super::GlVert>() as i32, 0 as *const _);
            gl::EnableVertexAttribArray(0);
            gl::VertexAttribPointer(1, 2, gl::FLOAT, gl::FALSE, std::mem::size_of::<super::GlVert>() as i32, (2 * std::mem::size_of::<f32>()) as *const _);
            gl::EnableVertexAttribArray(1);
            gl::VertexAttribPointer(2, 1, gl::BYTE, gl::FALSE, std::mem::size_of::<super::GlVert>() as i32, (4 * std::mem::size_of::<f32>()) as *const _);
            gl::EnableVertexAttribArray(2);

            gl::BindBuffer(gl::ARRAY_BUFFER, 0);
            gl::BindVertexArray(0);
        }

        let mut layers: [Option<GlLayer>; 11] = Default::default();
        layers[LayerType::ROADS as usize] = Some(GlLayer{ vao, vbo, unlabeled: 1, commands: vec![RenderCommand::Simple(line.verts.len())], labels: Vec::new() });

        let tid = unsafe { TILE_NUM +=1; TILE_NUM };
        return Tile{
            tid,
            x,
            y,
            z,
            extent: 4096,
            layers,
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
    pub enum Compression {
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

    pub struct File<R: Renderer> {
        pub font: FontMetric,
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

        r: std::marker::PhantomData<R>,
    }

    impl<'a, R: Renderer> File<R> {
        pub fn new<P: AsRef<std::path::Path>>(path: P, font: FontMetric) -> Self {
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
                font,
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

                r: std::marker::PhantomData,
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

        pub fn load_tile(&mut self, x: u64, y: u64, level: u8) -> Option<Tile<R>> {
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
                        return Some(compile_tile(&mut self.font, x, y, level, tile_slice).unwrap());
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
        pub source: File<GL>,

        pub active: HashMap<u64, Tile<GL>>,
        pub visible: Vec<u64>,
    }

    impl<'a> LiveTiles {
        pub fn new(source: File<GL>) -> Self {
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

            if level == 0 {
                // The coords_to_id function returns 0 for all tiles on level 0. The visbility
                // calculation in the else branch therefore doesn't work for level 0. Just hardcode
                // that result.
                self.visible = vec![0];
            } else {
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

        #[test]
        fn build_straight_line() {
            let mut line = LineBuilder::new();

            let v1 = Vector2::new(0.0, 0.0);
            let v2 = Vector2::new(0.0, 1.0);
            let v3 = Vector2::new(0.0, 2.0);

            line.add_point(v1, v2, false, true);
            line.add_point(v2, v3, true, true);

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
}
