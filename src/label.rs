use crate::math::Vector2;

pub struct Box {
    pub min: Vector2<f32>,
    pub max: Vector2<f32>,
}

pub fn select_nooverlap(boxes: &Vec<Box>, to_draw: &mut Vec<usize>) {
    let mut res : Vec<usize> = Vec::new();

    to_draw.retain(|i| {
        let b = &boxes[*i];
        let mut overlap = false;
        for r in &res {
            if b.min.x >= boxes[*r].max.x { continue; }
            if b.max.x <= boxes[*r].min.x { continue; }

            if b.min.y >= boxes[*r].max.y { continue; }
            if b.max.y <= boxes[*r].min.y { continue; }

            overlap = true;
            break;
        }

        if !overlap {
            res.push(*i);
            return true;
        }

        return false;
    });
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_overlap() {
        let boxes = vec![
            Box {
                min: Vector2::new(0.0, 0.0),
                max: Vector2::new(1.0, 1.0),
            },
            Box {
                min: Vector2::new(1.0, 1.0),
                max: Vector2::new(2.0, 2.0),
            },
        ];

        let mut res = vec![0, 1];
        select_nooverlap(&boxes, &mut res);

        assert_eq!(res.len(), 2);
        assert_eq!(res[0], 0);
        assert_eq!(res[1], 1);
    }

    #[test]
    fn second_overlap_first() {
        let boxes = vec![
            Box {
                min: Vector2::new(0.0, 0.0),
                max: Vector2::new(1.0, 1.0),
            },
            Box {
                min: Vector2::new(0.5, 0.5),
                max: Vector2::new(1.0, 1.0),
            },
        ];

        let mut res = vec![0, 1];
        select_nooverlap(&boxes, &mut res);

        assert_eq!(res.len(), 1);
        assert_eq!(res[0], 0);
    }

    #[test]
    fn overlap_on_one_axis() {
        let boxes = vec![
            Box {
                min: Vector2::new(0.0, 0.0),
                max: Vector2::new(1.0, 1.0),
            },
            Box {
                min: Vector2::new(0.0, 1.0),
                max: Vector2::new(1.0, 2.0),
            },
        ];

        let mut res = vec![0, 1];
        select_nooverlap(&boxes, &mut res);

        assert_eq!(res.len(), 2);
        assert_eq!(res[0], 0);
        assert_eq!(res[1], 1);
    }
}
