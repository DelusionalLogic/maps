use crate::{math::Vector2, pbuf};

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

pub fn parse_tile(reader: &mut pbuf::Message) -> pbuf::Result<RawTile> {
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

    struct Filter<const N: usize, const K: usize> {
        keys: [Option<u64>; N],
        values: [Option<u64>; N],
        str_keys: [Option<u64>; K],
        groups: [u64; N],
    }

    impl<const N: usize, const K: usize> Filter<N, K> {
        pub fn scan(reader: &mut pbuf::Message, keys: [&str; N], values: [&str; N], groups: [u64; N], str_keys: [&str; K]) ->pbuf::Result<Self> {
            assert!(keys.len() == values.len());
            assert!(keys.len() == groups.len());

            let (keys, values) = scan_for_keys_and_values(&mut reader.clone(), &keys.to_vec(), &values.to_vec())?;
            let (str_keys, _) = scan_for_keys_and_values(reader, &str_keys.to_vec(), &vec![])?;

            return Ok(Filter{
                keys: keys.try_into().expect("Array of scanned keys did not match input"),
                values: values.try_into().expect("Array of scanned values did not match input"),
                str_keys: str_keys.try_into().expect("static"),
                groups,
            });
        }

        pub fn sieve_features(&self, reader: &mut pbuf::Message) -> pbuf::Result<(Vec<(pbuf::Message, [Option<usize>; K])>, [Vec<(pbuf::Message, [Option<usize>; K])>; N], [Vec<u64>; K])> {
            // @SPEED: There may be some way to read the buffer linearly here instead of fucking up the
            // ordering. This is fine for now.
            let mut other = vec![];
            let mut features = std::array::from_fn(|_| vec![]);
            let mut str_values : [Vec<u64>; K] = vec![Vec::new(); K].try_into().expect("static");

            {
                while let Ok(field) = reader.next() {
                    match field {
                        pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                        pbuf::TypeAndTag{wtype: pbuf::WireType::Len, tag: 2} => {
                            reader.enter_message()?;
                            let feature_ptr = reader.clone();
                            let mut bucket = &mut other;
                            let mut name_index = [None; K];
                            while let Ok(field) = reader.next() {
                                match field {
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

                                            for i in 0..self.str_keys.len() {
                                                if self.str_keys[i] == Some(key) {
                                                    str_values[i].push(value);
                                                    name_index[i] = Some(str_values[i].len()-1);
                                                }
                                            }
                                        }
                                    },
                                    pbuf::TypeAndTag{wtype: pbuf::WireType::EOM, ..} => { break; }
                                    x => { reader.skip(x.wtype)?; },
                                }
                            }
                            bucket.push((feature_ptr, name_index));
                        }
                        x => { reader.skip(x.wtype)?; },
                    }
                }

                return Ok((other, features, str_values));
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
        let filter = Filter::scan(
            &mut reader.clone(),
            ["pmap:kind", "pmap:kind", "pmap:kind", "pmap:kind"],
            ["highway", "major_road", "medium_road", "minor_road"],
            [ 0, 1, 2, 3, ],
            ["name"]
        )?;

        let mut name_reader = reader.clone();
        let (mut other_features, mut features, name_indexes) = filter.sieve_features(reader)?;

        let name_index = &name_indexes[0];

        dbg!(&name_index);
        let mut index : Vec<usize> = (0..name_index.len()).collect();
        index.sort_by_key(|x| name_index[*x]);
        let (index, mut names) = lookup_values(&mut name_reader, &name_index, &index).unwrap();
        let offset = tile.strings.len();

        tile.strings.append(&mut names);

        fn read_geometry(features: &mut Vec<(pbuf::Message, [Option<usize>; 1])>, name_index: &Vec<usize>, offset: usize, geom: &mut LineGeom) -> pbuf::Result<()> {

            for (reader, [namei]) in features {
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

        read_geometry(&mut other_features, &index, offset, &mut tile.roads)?;
        read_geometry(&mut features[0], &index, offset, &mut tile.highways)?;
        read_geometry(&mut features[1], &index, offset, &mut tile.major)?;
        read_geometry(&mut features[2], &index, offset, &mut tile.medium)?;
        read_geometry(&mut features[3], &index, offset, &mut tile.minor)?;

        return Ok(());
    }


    fn read_landuse_layer(reader: &mut pbuf::Message) -> pbuf::Result<(PolyGeom, PolyGeom, PolyGeom)> {
        let filter = Filter::scan(
            &mut reader.clone(),
            ["landuse", "amenity", "place", "landuse", "pmap:kind"],
            ["farmland", "university", "neighbourhood", "residential", "pedestrian"],
            [ 0, 1, 1, 1, 0, ],
            []
        )?;

        let (mut other_features, mut features, _) = filter.sieve_features(reader)?;

        fn read_geometry(features: &mut Vec<(pbuf::Message, [Option<usize>; 0])>) -> pbuf::Result<PolyGeom> {
            let mut poly_start : Vec<LineStart> = vec![];
            let mut polys : Vec<GlVert> = vec![];

            for (reader, _) in features {
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
                                        let cnt = cmd >> 3;
                                        assert!(cnt >= 1);

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

                                        // @COMPLETE @HACK Right now we just discard the other
                                        // points in a multipoint
                                        for _ in 0..cnt-1 {
                                            pbuf::decode_zig(it.next().unwrap().unwrap());
                                            pbuf::decode_zig(it.next().unwrap().unwrap());
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
