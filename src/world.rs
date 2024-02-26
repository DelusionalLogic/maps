use std::collections::{
    HashMap,
    HashSet
};

use crate::{
    pmtile::{
        coords_to_id,
        PMTile,
    },
    pbuf::Message,
    mapbox::parse_tile,
    font::FontMetric,
    map::{
        compile_tile,
        Tile,
        GL
    }
};

#[derive(Clone, Copy, Debug)]
pub struct TileHandle { 
    value: u64
}

impl TileHandle {
    fn new(value: u64) -> Self {
        return TileHandle {
            value
        };
    }
}

pub struct World {
    pub source: PMTile,

    active: HashMap<u64, Tile<GL>>,
    pub visible: Vec<TileHandle>,
}
impl World {
    pub fn new(source: PMTile) -> Self {
        return World {
            source,
            active: HashMap::new(),
            visible: vec![],
        };
    }

    pub fn update_visible_tiles(&mut self, left: u64, top: u64, right: u64, bottom: u64, level: u8, font: &mut FontMetric) -> (Vec<TileHandle>, Vec<TileHandle>) {
        let right = (right+1).min(2_u64.pow(level.into()));
        let bottom = (bottom+1).min(2_u64.pow(level.into()));
        let top = top.min(bottom);
        let left = left.min(right);

        // @HACK we can do this without a map, but i'm just too lazy right now
        let mut id_to_coord = HashMap::new();
        let mut desired = HashSet::new();

        // Try to load the current screenful of tiles
        for x in left..right {
            for y in top..bottom {
                let id = coords_to_id(x, y, level);
                desired.insert(id);
                id_to_coord.insert(id, (x, y, level));
            }
        }

        let loaded: HashSet<u64> = self.active.keys().cloned().collect();

        let to_add = &desired - &loaded;
        let to_remove = &loaded - &desired;

        for id in &to_add {
            let (x, y, level) = id_to_coord.get(&id).unwrap();
            if !self.active.contains_key(&id) {
                if let Some(ptile) = self.source.load_tile(*x, *y, *level) {
                    let raw_tile = parse_tile(&mut Message::new(ptile)).unwrap();
                    let gl_tile = compile_tile(*id, font, *x, *y, *level, raw_tile).unwrap();
                    self.active.insert(*id, gl_tile);
                }
            }
        }

        for id in &to_remove {
            self.active.remove(id);
        }


        // The visible set is only the tiles that successfully loaded
        self.visible.clear();
        for id in desired {
            if self.active.contains_key(&id) {
                self.visible.push(TileHandle::new(id));
            }
        }
        self.visible.sort_by_key(|x| x.value);

        let to_add = to_add.iter()
            .filter(|x| self.active.contains_key(x))
            .map(|x| TileHandle::new(*x))
            .collect();

        let to_remove = to_remove.iter()
            .map(|x| TileHandle::new(*x))
            .collect();

        return (to_add, to_remove);
    }

    pub fn get(&self, key: TileHandle) -> Option<&Tile<GL>> {
        return self.active.get(&key.value);
    }

    pub fn get_mut(&mut self, key: TileHandle) -> Option<&mut Tile<GL>> {
        return self.active.get_mut(&key.value);
    }
}
