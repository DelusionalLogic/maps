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

pub struct Viewport {
    active: HashSet<u64>,
}

impl Viewport {

    pub fn new() -> Self {
        return Viewport {
            active: HashSet::new(),
        };
    }

    pub fn update(&mut self, left: u64, top: u64, right: u64, bottom: u64, level: u8) -> (Vec<TileHandle>, Vec<TileHandle>, Vec<(u64, u64, u8)>) {
        let right = (right+1).min(2_u64.pow(level.into()));
        let bottom = (bottom+1).min(2_u64.pow(level.into()));
        let top = top.min(bottom);
        let left = left.min(right);

        // @HACK we can do this without a map, but i'm just too lazy right now
        let mut desired = HashSet::new();
        let mut to_add = Vec::new();
        let mut id_to_coord = Vec::new();

        for x in left..right {
            for y in top..bottom {
                let id = coords_to_id(x, y, level);
                desired.insert(id);
                if !self.active.contains(&id) {
                    to_add.push(id);
                    id_to_coord.push((x, y, level));
                }
            }
        }

        let loaded: HashSet<u64> = self.active.clone();

        let to_remove = &loaded - &desired;
        self.active = desired;

        let to_add = to_add.iter()
            .map(|x| TileHandle::new(*x))
            .collect();

        let to_remove = to_remove.iter()
            .map(|x| TileHandle::new(*x))
            .collect();

        return (to_add, to_remove, id_to_coord);
    }
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct TileHandle { 
    value: u64,
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

    // All the tiles that are able to be looked up
    active: HashMap<u64, Tile<GL>>,
    pub visible: Vec<TileHandle>,
    pub viewport: Viewport,
}
impl World {
    pub fn new(source: PMTile) -> Self {
        return World {
            source,
            active: HashMap::new(),
            visible: vec![],
            viewport: Viewport::new(),
        };
    }

    pub fn update_visible_tiles(&mut self, left: u64, top: u64, right: u64, bottom: u64, level: u8, font: &mut FontMetric) -> (Vec<TileHandle>, Vec<TileHandle>) {
        let (to_add, to_remove, id_to_coord) = self.viewport.update(left, top, right, bottom, level);

        // @HACK This is a bad way to handle intersection, but lets move on
        for id in &to_remove {
            self.visible.retain(|x| x.value != id.value);
        }

        for (id, (x, y, level)) in to_add.iter().zip(id_to_coord.iter()) {
            if !self.active.contains_key(&id.value) {
                if let Some(ptile) = self.source.load_tile(*x, *y, *level) {
                    let raw_tile = parse_tile(&mut Message::new(ptile)).unwrap();
                    let gl_tile = compile_tile(id.value, font, *x, *y, *level, raw_tile).unwrap();
                    self.active.insert(id.value, gl_tile);
                    self.visible.push(*id);
                }
            } else {
                self.visible.push(*id);
            }
        }

        // The visible set is only the tiles that successfully loaded
        self.visible.sort_by_key(|x| x.value);

        let to_add = to_add.into_iter()
            .filter(|x| self.active.contains_key(&x.value))
            .collect();

        let to_remove = to_remove.into_iter()
            .filter(|x| self.active.contains_key(&x.value))
            .collect();

        return (to_add, to_remove);
    }

    pub fn free(&mut self, to_remove: &Vec<TileHandle>) {
        for id in to_remove {
            self.active.remove(&id.value);
        }
    }

    pub fn get(&self, key: TileHandle) -> Option<&Tile<GL>> {
        return self.active.get(&key.value);
    }

    pub fn get_mut(&mut self, key: TileHandle) -> Option<&mut Tile<GL>> {
        return self.active.get_mut(&key.value);
    }
}
