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

pub struct World {
    pub font: FontMetric,
    pub source: PMTile,

    pub active: HashMap<u64, Tile<GL>>,
    pub visible: Vec<u64>,
}

impl<'a> World {
    pub fn new(source: PMTile, font: FontMetric) -> Self {
        return World {
            font,
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
                        let raw_tile = parse_tile(&mut Message::new(ptile)).unwrap();
                        let gl_tile = compile_tile(id, &mut self.font, x, y, level, raw_tile).unwrap();
                        self.active.insert(id, gl_tile);
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
