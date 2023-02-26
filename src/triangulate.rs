
#[derive(Clone,Copy,Debug)]
pub struct Vector {
    x: f64,
    y: f64,
}

impl Vector {
    pub fn sub(&mut self, other: &Vector) {
        self.x -= other.x;
        self.y -= other.y;
    }

    pub fn cross(&self, other: &Vector) -> f64 {
        return self.x * other.y - self.y * other.x;
    }
}

pub struct BBox {
    min: Vector,
    max: Vector,
}

impl BBox {
    fn from_points(xs: Vec<f64>, ys: Vec<f64>) -> Self {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for x in xs {
            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
        }

        for y in ys {
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }
        }

        return BBox {
            min: Vector {
                x: min_x,
                y: min_y,
            },
            max: Vector {
                x: max_x,
                y: max_y,
            },
        };
    }
}

#[derive(Clone,Copy,Debug)]
struct TriSide {
    tri: usize,
    side: u8,
}

#[derive(Debug)]
pub struct Tris {
    verts: Vec<Vector>,
    tris: Vec<[usize; 3]>,
    edges: Vec<[usize; 2]>,
    adj: Vec<[Option<TriSide>; 3]>,
    inverse_tri: Vec<TriSide>,
}

impl Tris {
    pub fn super_tri_of_bbox(bbox: &BBox, num_verts: u64) -> Self {
        let mut verts = Vec::with_capacity(num_verts as usize + 3);
        let mut tris = Vec::with_capacity((num_verts as usize * 2) + 1);
        let edges = Vec::with_capacity(num_verts as usize * 2);
        let mut adj = Vec::with_capacity(tris.capacity());
        let mut inverse_tri = Vec::with_capacity(verts.capacity());

        let center_x = (bbox.min.x + bbox.max.x) / 2.0;
        let center_y = (bbox.min.y + bbox.max.y) / 2.0;

        let size_x = bbox.max.x - bbox.min.x;
        let size_y = bbox.max.y - bbox.min.y;

        let inrad = f64::sqrt(f64::powi(size_x, 2) + f64::powi(size_y, 2));
        let stride = inrad * 2.0 * f64::sqrt(3.0/2.0);

        verts.push(Vector{
            x: center_x - stride,
            y: center_y - inrad,
        });
        verts.push(Vector{
            x: center_x,
            y: center_y + inrad * 2.0,
        });
        verts.push(Vector{
            x: center_x + stride,
            y: center_y - inrad,
        });

        tris.push([
            0, 1, 2
        ]);

        adj.push([
            None, None, None
        ]);

        inverse_tri.push(TriSide{
            tri: 0, side: 0
        });
        inverse_tri.push(TriSide{
            tri: 0, side: 1
        });
        inverse_tri.push(TriSide{
            tri: 0, side: 2
        });

        return Tris {
            verts,
            tris,
            edges,
            adj,
            inverse_tri,
        };
    }

    fn find_containg_tri(&self, p: &Vector, start_tri: Option<usize>) -> Option<TriSide> {
        // @INCOMPLETE: We should have some fallback for when we don't have a first guess
        let mut cursor = TriSide{
            tri: start_tri.unwrap(),
            side: 0,
        };

        loop {
            let mut found = true;
            for _ in 0..3 {
                let root = self.tris[cursor.tri][cursor.side as usize];
                let next = self.tris[cursor.tri][anticlockwise(cursor.side) as usize];

                let mut to_p = p.clone();
                to_p.sub(&self.verts[root]);

                let mut to_next = self.verts[next].clone();
                to_next.sub(&self.verts[root]);

                if to_next.cross(&to_p).is_sign_positive() {
                    found = false;
                    break;
                }
                cursor.side = anticlockwise(cursor.side);
            }

            if found {
                return Some(cursor);
            }

            cursor.side = clockwise(cursor.side);
            cursor = match self.opposing_tri(&cursor) {
                None => return None,
                Some(x) => x.clone(),
            };
        }
    }

    fn opposing_tri(&self, side: &TriSide) -> Option<TriSide> {
        return self.adj[side.tri][side.side as usize];
    }

    fn split_tri(&mut self, contain: &TriSide, p: &Vector) -> (usize, usize, usize) {
        let tri1 = contain.tri;
        let tri1o = self.tris[tri1];

        let v0 = tri1o[0];
        let v1 = tri1o[1];
        let v2 = tri1o[2];

        // Fetch the neigbouring tris as their adjencency information will be used later on and
        // updated as well.
        let o0 = self.opposing_tri(&TriSide{tri: contain.tri, side: 0});
        let o1 = self.opposing_tri(&TriSide{tri: contain.tri, side: 1});
        let o2 = self.opposing_tri(&TriSide{tri: contain.tri, side: 2});

        let v3 = self.verts.len();
        self.verts.push(*p);
        // Original tri becomes v0 v1 v3
        self.tris[tri1][2] = v3;
        let tri2o = [v1, v2, v3];
        let tri3o = [v2, v0, v3];

        // Insert the new tris while keeping tabs on their ids
        let tri2 = self.tris.len();
        self.tris.push(tri2o);
        let tri3 = self.tris.len();
        self.tris.push(tri3o);

        // Update the foreign side of the adjecency information if applicable
        if let Some(o0) = o0 {
            self.adj[o0.tri][o0.side as usize] = Some(TriSide{tri: tri2, side: 2});
        }
        if let Some(o1) = o1 {
            self.adj[o1.tri][o1.side as usize] = Some(TriSide{tri: tri3, side: 2});
        }
        if let Some(o2) = o2 {
            self.adj[o2.tri][o2.side as usize] = Some(TriSide{tri: tri1, side: 2});
        }

        // Set the adjecency information for this triangle
        // We have to be careful to push these in for same order we pushed the tris
        self.adj.push([
            Some(TriSide { tri: tri3, side: 1 }),
            Some(TriSide { tri: tri1, side: 0 }),
            o0,
        ]);
        self.adj.push([
            Some(TriSide { tri: tri1, side: 1 }),
            Some(TriSide { tri: tri2, side: 0 }),
            o1,
        ]);
        self.adj[tri1] = [
            Some(TriSide { tri: tri2, side: 1 }),
            Some(TriSide { tri: tri3, side: 0 }),
            o2,
        ];

        self.inverse_tri[v0] = TriSide{ tri: tri1, side: 0};
        self.inverse_tri[v1] = TriSide{ tri: tri1, side: 1};
        self.inverse_tri[v2] = TriSide{ tri: tri2, side: 1};
        self.inverse_tri.push(TriSide{ tri: tri1, side: 2});

        return (v3, tri2, tri1);
    }

    fn resolve_with_swaps(&mut self, tri: &TriSide, mut stack: Vec<TriSide>) {
        while let Some(curs) = stack.pop() {
            let t_opt = self.opposing_tri(&curs);
            if t_opt.is_none() {
                continue;
            }
            let t_opt = t_opt.unwrap();
            let t_op = self.tris[t_opt.tri];
            let oppo_clock = clockwise(t_opt.side);
            let me_clock = clockwise(curs.side);
            let t_op_swap = anticlockwise(t_opt.side);
            if incircle(self.verts[t_op[0]], self.verts[t_op[1]], self.verts[t_op[2]], &self.verts[t_op[tri.side as usize]]) {
                let t_me_swap = anticlockwise(curs.side);

                let t3 = self.opposing_tri(&TriSide { tri: curs.tri, side: me_clock });
                let t2 = self.opposing_tri(&TriSide { tri: t_opt.tri, side: oppo_clock });

                self.adj[curs.tri][curs.side as usize] = t2;
                if let Some(t2) = t2 {
                    self.adj[t2.tri][t2.side as usize] = Some(curs);
                }

                self.adj[t_opt.tri][t_opt.side as usize] = t3;
                if let Some(t3) = t3 {
                    self.adj[t3.tri][t3.side as usize] = Some(t_opt);
                }


                self.adj[curs.tri][me_clock as usize] = Some(TriSide{tri: t_opt.tri, side:oppo_clock});
                self.adj[t_opt.tri][oppo_clock as usize] = Some(TriSide{tri: curs.tri, side:me_clock});

                self.inverse_tri[self.tris[curs.tri][t_me_swap as usize]] = TriSide{tri: t_opt.tri, side: oppo_clock};
                self.inverse_tri[self.tris[t_opt.tri][t_op_swap as usize]] = TriSide{tri: curs.tri, side: me_clock};

                self.tris[t_opt.tri][t_op_swap as usize] = self.tris[curs.tri][curs.side as usize];
                self.tris[curs.tri][t_me_swap as usize] = self.tris[t_opt.tri][t_opt.side as usize];

                stack.push(curs);
                stack.push(TriSide {tri: t_opt.tri, side: t_op_swap});
            }
        }
    }

    pub fn add_point(&mut self, p: &Vector) -> Result<usize, String> {
        let contain = self.find_containg_tri(p, Some(0))
            .ok_or("The point is outside the mesh")?;

        let (new_vert, tri2, tri3) = self.split_tri(&contain, p);

        let stack = vec![
            TriSide{tri: contain.tri, side: 2},
            TriSide{tri: tri2, side: 2},
            TriSide{tri: tri3, side: 2},
        ];
        self.resolve_with_swaps(&TriSide{tri: contain.tri, side: 2}, stack);

        return Ok(new_vert);
    }

    fn find_tri_with_vert(&self, needle: usize) -> &TriSide {
        return &self.inverse_tri[needle];
    }

    pub fn add_edge(&mut self, v1i: usize, v2i: usize) {
        assert!(v1i != v2i);

        // Find a triangle with one of the vertecies as a corner
        let start_tri = self.find_tri_with_vert(v1i).clone();

        let v1 = self.verts[v1i];
        let v2 = self.verts[v2i];

        let mut to = v2.clone();
        to.sub(&v1);

        let mut tri_cursor = start_tri;

        // Find the triangle that straddles the line between v1 and v2
        loop {
            let tri = self.tris[tri_cursor.tri];

            let mut ab = self.verts[tri[clockwise(tri_cursor.side) as usize]].clone();
            let mut ac = self.verts[tri[anticlockwise(tri_cursor.side) as usize]].clone();

            ab.sub(&v1);
            ac.sub(&v1);

            // Check if the vector v1 -> v2 is inside the span of the vectors v1 -> b and v1 -> c
            let inside = (ab.cross(&ac) * ab.cross(&to)).is_sign_positive() &&
                (ac.cross(&ab) * ac.cross(&to)).is_sign_positive();

            // If we aren't inside then we much continue the search by finding the next triangle
            // that shares this vertex
            if !inside {
                let op_tri = self.opposing_tri(&TriSide{tri: tri_cursor.tri, side: clockwise(tri_cursor.side)}).unwrap();

                // If we arrive back at the start, that's an error. It should NEVER be possible for
                // no triangle to straddle the edge
                assert!(op_tri.tri != start_tri.tri);

                tri_cursor.tri = op_tri.tri;
                tri_cursor.side = clockwise(op_tri.side);
            } else {
                break;
            }
        }

        // We now have the triangle that straddles the edge in tri_cursor poiting at the side of v1

        // We now have to destroy all the triangles cut by the new edge to reconstruct them. While
        // doing this we want to keep track of the nodes that fall _below_ and _above_ the new
        // edge. We will also want to keep track of their adjecencies to make reassignment easier
        // when we get to that.

        // If a single tri contains both v1 and v2 we aren't cutting any edge and trivially return.
        // The edge is already part of the mesh
        let mut upper : Vec<usize> = vec![];
        let mut upper_adj : Vec<Option<TriSide>> = vec![];
        if self.tris[tri_cursor.tri][clockwise(tri_cursor.side) as usize] != v2i {
            upper.push(self.tris[tri_cursor.tri][clockwise(tri_cursor.side) as usize]);
            upper_adj.push(self.opposing_tri(&TriSide {tri: tri_cursor.tri, side: anticlockwise(tri_cursor.side)}));
        } else {
            return
        }

        let mut lower : Vec<usize> = vec![];
        let mut lower_adj : Vec<Option<TriSide>> = vec![];
        if self.tris[tri_cursor.tri][anticlockwise(tri_cursor.side) as usize] != v2i {
            lower.push(self.tris[tri_cursor.tri][anticlockwise(tri_cursor.side) as usize]);
            lower_adj.push(self.opposing_tri(&TriSide {tri: tri_cursor.tri, side: clockwise(tri_cursor.side)}));
        } else {
            return
        }
    }
}

fn incircle(mut a: Vector, mut b: Vector, mut c: Vector, d: &Vector) -> bool {
    a.sub(&d);
    b.sub(&d);
    c.sub(&d);

    let bcdet = b.cross(&c);
    let cadet = c.cross(&a);
    let abdet = a.cross(&b);

    let alift = a.x * a.x + a.y * a.y;
    let blift = b.x * b.x + b.y * b.y;
    let clift = c.x * c.x + c.y * c.y;

    let deter = alift * bcdet + blift * cadet + clift * abdet;
    return deter.is_sign_negative();
}

fn anticlockwise(side: u8) -> u8 {
    return (side + 1) % 3;
}

fn clockwise(side: u8) -> u8 {
    return if side == 0 { 2 } else { side - 1 };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_bbox_from_points() {
        let xs = vec![-1.0, 2.0];
        let ys = vec![1.0, 3.0];

        let bbox = BBox::from_points(xs, ys);

        assert_eq!(bbox.min.x, -1.0);
        assert_eq!(bbox.min.y, 1.0);
        assert_eq!(bbox.max.x, 2.0);
        assert_eq!(bbox.max.y, 3.0);
    }

    fn left_of(p1: &Vector, p2: &Vector, p3: &Vector) -> bool {
        return f64::is_sign_negative((p2.x-p1.x)*(p3.y-p1.y) - (p2.y-p1.y)*(p3.x-p1.x));
    }
    #[test]
    fn left_of_test() {
        let p1 = Vector{ x:1.0, y:  1.0 };
        let p2 = Vector{ x:1.0, y:101.0 };
        let p3 = Vector{ x:1.1, y: 51.0 };

        assert!(left_of(&p1, &p2, &p3));
    }
    #[test]
    fn point_falls_right() {
        let p1 = Vector{ x:1.0, y:  1.0 };
        let p2 = Vector{ x:1.0, y:101.0 };
        let p3 = Vector{ x:0.9, y: 51.0 };

        assert!(!left_of(&p1, &p2, &p3));
    }

    fn point_inside_tri(p1: &Vector, p2: &Vector, p3: &Vector, p4: &Vector) -> bool {
        if !left_of(p1, p2, p4) {
            return false;
        }

        if !left_of(p2, p3, p4) {
            return false;
        }

        if !left_of(p3, p1, p4) {
            return false;
        }

        return true;
    }

    fn box_inside_tri(tri: &Tris, bbox: &BBox) -> bool {
        let first_tri = tri.tris[0];

        let check = |x: f64, y: f64| -> bool {
            return !point_inside_tri(&tri.verts[first_tri[0]], &tri.verts[first_tri[1]], &tri.verts[first_tri[2]], &Vector{x, y});
        };

        if check(bbox.min.x, bbox.min.y) {
            return false;
        }
        if check(bbox.min.x, bbox.max.y) {
            return false;
        }
        if check(bbox.max.x, bbox.max.y) {
            return false;
        }
        if check(bbox.max.x, bbox.min.y) {
            return false;
        }

        return true;
    }

    #[test]
    fn create_super_tri_from_bbox() {
        let bbox = BBox{
            min: Vector{x: -1.0, y: -1.0},
            max: Vector{x:  1.0, y:  1.0},
        };

        let tri = Tris::super_tri_of_bbox(&bbox, 0);

        // The only verticies should be the initial triangle
        assert_eq!(tri.verts.len(), 3);
        assert_eq!(tri.inverse_tri.len(), 3);
        // The only triangle should be the initial triangle
        assert_eq!(tri.tris.len(), 1);
        assert_eq!(tri.adj.len(), 1);
        // We haven't inserted any edges
        assert!(tri.edges.is_empty());

        // No adjecent triangles
        assert!(tri.adj[0][0].is_none());
        assert!(tri.adj[0][1].is_none());
        assert!(tri.adj[0][2].is_none());

        validate_mesh(&tri);

        assert!(box_inside_tri(&tri, &bbox));
    }

    #[test]
    fn find_point_inside_start_tri() {
        let bbox = BBox{
            min: Vector{x: -1.0, y: -1.0},
            max: Vector{x:  1.0, y:  1.0},
        };

        let tris = Tris::super_tri_of_bbox(&bbox, 0);
        let tri = tris.find_containg_tri(&Vector{x: 0.0, y: 0.0}, Some(0));

        assert!(tri.is_some());
        assert_eq!(tri.unwrap().tri, 0);
    }

    #[test]
    fn find_no_tri_point_outside_tris() {
        let bbox = BBox{
            min: Vector{x: -1.0, y: -1.0},
            max: Vector{x:  1.0, y:  1.0},
        };

        let tris = Tris::super_tri_of_bbox(&bbox, 0);
        // @FRAGILE: the (10, 10) here is arbitrary. It''s outside the bbox, but since the triangle
        // is allowed to extend outside of that it's not guaranteed to be outide the triangle. We
        // should do some sort of triangle bounding box to find a point that's for sure outside the
        // triangle
        let tri = tris.find_containg_tri(&Vector{x: 10.0, y: 10.0}, Some(0));

        assert!(tri.is_none());
    }

    fn validate_mesh(tris: &Tris) {
        assert_eq!(tris.verts.len(), tris.inverse_tri.len());
        assert_eq!( tris.tris.len(),         tris.adj.len());

        // All verts in the vert -> tri lookup array point to themselves
        {
            for (i, &x) in tris.inverse_tri.iter().enumerate() {
                assert_eq!(tris.tris[x.tri][x.side as usize], i);
            }
        };

        // All verts in all triangles opposes some other triangle with a differing opposing vert
        // and 2 other shared verts
        {
            for (trii, tri) in tris.tris.iter().enumerate() {
                for (side, verti) in tri.iter().enumerate() {
                    let adj = tris.adj[trii][side as usize];
                    if let Some(adj) = adj {
                        // The opposing triangle cannot be the same triangle
                        assert_ne!(adj.tri, trii);
                        // The vertex opposing a vertex cannot be the same vertex
                        assert_ne!(*verti, tris.tris[adj.tri][adj.side as usize]);

                        // The opposing triangle must share two vertecies with our triangle
                        assert_eq!(tri[clockwise(side as u8) as usize], tris.tris[adj.tri][anticlockwise(adj.side) as usize]);
                        assert_eq!(tri[anticlockwise(side as u8) as usize], tris.tris[adj.tri][clockwise(adj.side) as usize]);
                    }
                }
            }
        };

        // Check the winding on the triangles
        {
            for t in &tris.tris {
                let a = tris.verts[t[0]];
                let mut b = tris.verts[t[1]];
                let mut c = tris.verts[t[2]];

                b.sub(&a);
                c.sub(&a);

                assert!(!b.cross(&c).is_sign_positive());
            }
        };
    }

    #[test]
    fn split_a_single_triangle() {
        let bbox = BBox{
            min: Vector{x: -1.0, y: -1.0},
            max: Vector{x:  1.0, y:  1.0},
        };
        let mut tris = Tris::super_tri_of_bbox(&bbox, 0);
        let p = Vector{x: 0.0, y: 0.0};

        tris.split_tri(&TriSide { tri: 0, side: 0 }, &p);

        assert_eq!(tris.tris.len(), 3);
        validate_mesh(&tris);
    }

    #[test]
    fn add_first_point() {
        let bbox = BBox{
            min: Vector{x: -1.0, y: -1.0},
            max: Vector{x:  1.0, y:  1.0},
        };
        let mut tris = Tris::super_tri_of_bbox(&bbox, 0);
        let p = Vector{x: 0.0, y: 0.0};

        let vert = tris.add_point(&p).unwrap();

        assert_eq!(tris.tris.len(), 3);
        validate_mesh(&tris);

        // Count the number of triangles include the new vertex. Since we split one triangle into
        // 3, all three triangles must include the new vertex.
        let num_including = tris.tris.iter().flatten().fold(0, |a, x|{
            if *x == vert { a + 1 } else { a }
        });
        assert_eq!(num_including, 3);
    }
}
