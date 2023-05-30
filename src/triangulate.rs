use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;


#[derive(Debug)]
pub enum Error {
    InvalidPoly,
    DuplicatedEdge,
    InternalAlgo,
    PointOutsideMesh,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return match self {
            Error::InvalidPoly => write!(f, "invalid input polygon"),
            Error::DuplicatedEdge => write!(f, "duplicated Edge"),
            Error::InternalAlgo => write!(f, "internal algoritm error"),
            Error::PointOutsideMesh => write!(f, "point outside mesh"),
        };
    }
}

type Vector = crate::math::Vector2<f64>;

pub struct BBox {
    min: Vector,
    max: Vector,
}

impl BBox {
    fn from_points<T: Iterator<Item=(f64, f64)>>(points: T) -> Self {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for (x, y) in points {
            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
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

#[derive(Clone,Copy,Debug,PartialEq)]
struct TriSide {
    tri: usize,
    side: u8,
}

pub struct ActiveTriangulation {
    pub verts: Vec<Vector>,
    pub tris: Vec<[usize; 3]>,
    edges: HashSet<[usize; 2]>,
    adj: Vec<[Option<TriSide>; 3]>,
    inverse_tri: Vec<TriSide>,
}

pub struct Triangulation {
    pub verts: Vec<Vector>,
    pub tris: Vec<[usize; 3]>,
}

fn build_adjecency(tris: &[[usize; 3]]) -> Vec<[Option<TriSide>; 3]> {
    let mut adj : Vec<[Option<TriSide>; 3]> = vec![[None, None, None]; tris.len()];

    // Calculate the adjecencies
    {
        let mut seen = HashMap::new();
        let mut set_adj = |edge: TriSide| {
            let e = (tris[edge.tri][anticlockwise(edge.side) as usize], tris[edge.tri][clockwise(edge.side) as usize]);
            if let Some(opposite) = seen.remove(&e) {
                assert!(adj[edge.tri][edge.side as usize].is_none());
                adj[edge.tri][edge.side as usize] = Some(opposite);
                assert!(adj[opposite.tri][opposite.side as usize].is_none());
                adj[opposite.tri][opposite.side as usize] = Some(edge);
            } else {
                // Swap the edge direction to allow for the opposite side to do the lookup
                seen.insert((e.1, e.0), edge);
            }
        };

        for i in 0..tris.len() {
            set_adj(TriSide{tri: i, side: 0});
            set_adj(TriSide{tri: i, side: 1});
            set_adj(TriSide{tri: i, side: 2});
        }
    }

    return adj;
}

pub fn triangulate<S, P>(mut polys: S, points: P, expected_points: usize) -> Result<ActiveTriangulation, Error>
where P: Iterator<Item=(f64, f64)> + Clone,
      S: Iterator<Item=usize> {

    let bbox = BBox::from_points(points.clone());

    let mut tris = ActiveTriangulation::super_tri_of_bbox(&bbox, expected_points as _);

    let mut vid = Vec::with_capacity(expected_points);

    fn flush_tri(tris: &mut ActiveTriangulation, vid: &mut Vec<usize>) -> Result<(), Error> {
        if vid.len() < 3 { return Err(Error::InvalidPoly); }
        // Add all the edges
        for i in 0..vid.len()-1 {
            let next = i + 1;
            tris.add_edge(vid[i], vid[next])?;
        if let Some(x) = validate_mesh(&tris) {
            panic!("{:?}", x);
        }
        }
        tris.add_edge(*vid.last().unwrap(), *vid.first().unwrap())?;
        if let Some(x) = validate_mesh(&tris) {
            panic!("{:?}", x);
        }

        // Begin new poly
        vid.clear();

        return Ok(());
    }

    // Skip the first value since that's implicit
    assert!(polys.next().unwrap() == 0);

    let mut next_poly = polys.next();
    for (i, p) in points.enumerate() {
        if next_poly.is_some() && next_poly.unwrap() == i {
            // Flush the current poly to start a new one
            flush_tri(&mut tris, &mut vid)?;
            next_poly = polys.next();
        }
        vid.push(
            tris.add_point(&Vector{x: p.0, y: p.1})?
        );
        if let Some(x) = validate_mesh(&tris) {
            panic!("{:?}", x);
        }
    }
    // Flush the final triangle
    flush_tri(&mut tris, &mut vid)?;

    return Ok(tris);
}

impl ActiveTriangulation {
    pub fn super_tri_of_bbox(bbox: &BBox, num_verts: u64) -> Self {
        let mut verts = Vec::with_capacity(num_verts as usize + 3);
        let mut tris = Vec::with_capacity((num_verts as usize * 2) + 1);
        let edges = HashSet::with_capacity(num_verts as usize * 2);
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

        return ActiveTriangulation {
            verts,
            tris,
            edges,
            adj,
            inverse_tri,
        };
    }

    fn explicit(verts: &[Vector], tris: &[[usize; 3]]) -> Self {
        let adj = build_adjecency(tris);
        let mut inverse_tri = vec![TriSide{tri: 0, side: 0}; verts.len()];

        for i in 0..tris.len() {
            inverse_tri[tris[i][0]] = TriSide{tri: i, side: 0};
            inverse_tri[tris[i][1]] = TriSide{tri: i, side: 1};
            inverse_tri[tris[i][2]] = TriSide{tri: i, side: 2};
        }


        return ActiveTriangulation {
            verts: verts.to_vec(),
            tris: tris.to_vec(),
            edges: HashSet::new(),
            adj,
            inverse_tri,
        }
    }

    pub fn trim(self) -> Result<Triangulation, Error> {
        // Trim away the unused triangles left from the triangulation phase. The idea is to walk
        // the triangle mesh and save if it's inside or outside, flipping the state whenever we hit
        // a fixed edge. Every triangle is given it's state on the first visit, and future visit
        // just check if the state is consistent.
        let mut outside = vec![None; self.tris.len()];

        // All the triangles that touch point 0 (one of the start points) are guaranteed to be
        // outside
        let start_tri = self.inverse_tri[0];
        outside[start_tri.tri] = Some(true);

        let mut stack = vec![start_tri.tri];
        while let Some(x) = stack.pop() {
            // Update all out neighbours and schedule them if they haven't been visited before
            for i in 0..3 {
                let corner1 = anticlockwise(i);
                let corner2 = anticlockwise(corner1);

                let t_opt = self.opposing_tri(&TriSide { tri: x, side: i });
                if t_opt.is_none() {
                    continue;
                }

                // If the joining edge is fixed we flip the state bit
                let fixed = self.is_fixed(self.tris[x][corner1 as usize], self.tris[x][corner2 as usize]);
                let their_state = outside[x].unwrap() ^ fixed;

                if let Some(my_state) = outside[t_opt.unwrap().tri] {
                    // The triangle was already visited, just check for consistency
                    if my_state != their_state {
                        // Inconsistency, there was a hole in the polygon
                        return Err(Error::InternalAlgo);
                    }
                } else {
                    // Assign the current state to the triangle and schedule a visit to its
                    // neighbours
                    outside[t_opt.unwrap().tri] = Some(their_state);
                    stack.push(t_opt.unwrap().tri);
                }
            }
        }

        // @SPEED We could just compact the original triangle list instead of creating a new one
        let mut tris = Vec::new();
        for i in 0..self.tris.len() {
            if outside[i].unwrap() { continue; }

            tris.push(self.tris[i]);
        }

        return Ok(Triangulation {
            // The only verts we'd be able to discard are the first 3 ones, since all the others
            // are required to represent the geometry. That doesn't seem worth the hassle.
            verts: self.verts,
            tris,
        })
    }

    fn find_containg_tri(&self, p: &Vector, start_tri: Option<usize>) -> Result<(usize, TriangleLocation), Error> {
        let mut cursor = start_tri.unwrap_or(0);

        let mut iter = 0;
        loop {
            assert!(iter < 1000);
            iter += 1;

            let tri = self.tris[cursor];
            let loc = locate_point_in_tri(
                &self.verts[tri[0]],
                &self.verts[tri[1]],
                &self.verts[tri[2]],
                p,
            );

            // If the triangle is outside any edge we walk in that direction, otherwise we've found
            // it.
            match loc {
                TriangleLocation::Outside(x) => {
                    cursor = match self.opposing_tri(&TriSide{tri: cursor, side: x}) {
                        None => return Err(Error::PointOutsideMesh),
                        Some(x) => x.tri,
                    };
                }
                loc => return Ok((cursor, loc)),
            }
        }
    }

    fn opposing_tri(&self, side: &TriSide) -> Option<TriSide> {
        return self.adj[side.tri][side.side as usize];
    }

    fn split_tri(&mut self, tri: usize, p: &Vector) -> (usize, usize, usize) {
        let tri1 = tri;
        let tri1o = self.tris[tri1];

        let v0 = tri1o[0];
        let v1 = tri1o[1];
        let v2 = tri1o[2];

        // Fetch the neigbouring tris as their adjencency information will be used later on and
        // updated as well.
        let o0 = self.opposing_tri(&TriSide{tri, side: 0});
        let o1 = self.opposing_tri(&TriSide{tri, side: 1});
        let o2 = self.opposing_tri(&TriSide{tri, side: 2});

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

    pub fn is_fixed(&self, v1i: usize, v2i: usize) -> bool {
        return self.edges.contains(&[v1i, v2i]) || self.edges.contains(&[v2i, v1i]);
    }

    fn resolve_with_swaps(&mut self, v: usize, mut stack: Vec<TriSide>) {
        let mut iter = 0;
        while let Some(curs) = stack.pop() {
            iter += 1;
            assert!(iter < 100);

            let t_opt = self.opposing_tri(&curs);
            if t_opt.is_none() {
                continue;
            }
            let t_opt = t_opt.unwrap();
            let t_op = self.tris[t_opt.tri];
            let oppo_clock = clockwise(t_opt.side);
            let me_clock = clockwise(curs.side);
            let t_op_swap = anticlockwise(t_opt.side);

            if self.is_fixed(t_op[oppo_clock as usize], t_op[t_op_swap as usize]) {
                continue;
            }

            if incircle(self.verts[t_op[0]], self.verts[t_op[1]], self.verts[t_op[2]], &self.verts[v]) {
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

    pub fn add_point(&mut self, p: &Vector) -> Result<usize, Error> {
        // @SPEED: We should save some start_tri that's likely to be close to the point
        let (tri, loc) = self.find_containg_tri(p, None)?;

        assert!(loc.is_inside());

        if let TriangleLocation::IsCorner(x) = loc {
            return Ok(self.tris[tri][x as usize]);
        }

        if let TriangleLocation::OnEdge(side) = loc {
            // Split the 2 bordering tris into 4

            let original_tri1 = tri;
            let original_tri2 = self.opposing_tri(&TriSide{tri, side}).unwrap();

            // Fetch the outside neighbours
            // o1 and o3 aren't fetched as they are left untouched
            let o0 = self.opposing_tri(&TriSide{tri, side: anticlockwise(side)});
            let o2 = self.opposing_tri(&TriSide{tri: original_tri2.tri, side: anticlockwise(original_tri2.side)});

            let v0 = self.tris[original_tri1][side as usize];
            let v1 = self.tris[original_tri1][anticlockwise(side) as usize];
            let v2 = self.tris[original_tri2.tri][original_tri2.side as usize];
            let v3 = self.tris[original_tri1][clockwise(side) as usize];

            let new_v = self.verts.len();
            self.verts.push(*p);

            self.tris[original_tri1][clockwise(side) as usize] = new_v;
            self.tris[original_tri2.tri][clockwise(original_tri2.side) as usize] = new_v;

            let new_tri1o = [v3, v0, new_v];
            let new_tri1 = self.tris.len();
            self.tris.push(new_tri1o);
            let new_tri2o = [v1, v2, new_v];
            let new_tri2 = self.tris.len();
            self.tris.push(new_tri2o);

            self.adj[original_tri1][side as usize] = Some(TriSide{tri: new_tri2, side: 1});
            self.adj[original_tri1][anticlockwise(side) as usize] = Some(TriSide{tri: new_tri1, side: 0});
            self.adj[original_tri2.tri][original_tri2.side as usize] = Some(TriSide{tri: new_tri1, side: 1});
            self.adj[original_tri2.tri][anticlockwise(original_tri2.side) as usize] = Some(TriSide{tri: new_tri2, side: 0});

            if let Some(o0) = o0 {
                self.adj[o0.tri][o0.side as usize] = Some(TriSide{tri: new_tri1, side: 2});
            }
            if let Some(o2) = o2 {
                self.adj[o2.tri][o2.side as usize] = Some(TriSide{tri: new_tri2, side: 2});
            }

            // We have to push these in the same order as the triangles
            // new_tri1
            self.adj.push([
                Some(TriSide { tri: original_tri1, side: anticlockwise(side)}),
                Some(TriSide { tri: original_tri2.tri, side: original_tri2.side}),
                o0,
            ]);
            // new_tri2
            self.adj.push([
                Some(TriSide { tri: original_tri2.tri, side: anticlockwise(original_tri2.side)}),
                Some(TriSide { tri: original_tri1, side}),
                o2,
            ]);

            self.inverse_tri[v0] = TriSide{tri: new_tri1, side: 1};
            self.inverse_tri[v1] = TriSide{tri: new_tri2, side: 0};
            self.inverse_tri[v2] = TriSide{tri: new_tri2, side: 1};
            self.inverse_tri[v3] = TriSide{tri: new_tri1, side: 0};
            self.inverse_tri.push(TriSide{tri: new_tri1, side: 2});

            if self.edges.remove(&[v1, v3]) || self.edges.remove(&[v3, v1]) {
                self.edges.insert([v1, new_v]);
                self.edges.insert([new_v, v3]);
            }

            let stack = vec![
                TriSide{tri: original_tri1, side: clockwise(side)},
                TriSide{tri: original_tri2.tri, side: clockwise(original_tri2.side)},
                TriSide{tri: new_tri1, side: 2},
                TriSide{tri: new_tri2, side: 2},
            ];
            self.resolve_with_swaps(new_v, stack);

            return Ok(new_v);
        } else {
            let (new_vert, tri2, tri3) = self.split_tri(tri, p);

            let stack = vec![
                TriSide{tri, side: 2},
                TriSide{tri: tri2, side: 2},
                TriSide{tri: tri3, side: 2},
            ];
            self.resolve_with_swaps(new_vert, stack);

            return Ok(new_vert);
        }
    }

    fn find_tri_with_vert(&self, needle: usize) -> &TriSide {
        return &self.inverse_tri[needle];
    }

    pub fn add_edge(&mut self, v1i: usize, v2i: usize) -> Result<(), Error> {
        if v1i == v2i {
            return Err(Error::InvalidPoly);
        }

        if self.is_fixed(v1i, v2i) { return Err(Error::DuplicatedEdge); }

        // Find a triangle with one of the vertecies as a corner
        let start_tri = self.find_tri_with_vert(v1i).clone();

        let v1 = self.verts[v1i];
        let v2 = self.verts[v2i];

        let mut tri_cursor = start_tri;

        // Find the triangle that straddles the line between v1 and v2
        let mut iter = 0;
        loop {
            assert!(iter < 100);
            iter += 1;

            let tri = self.tris[tri_cursor.tri];

            let b = self.verts[tri[clockwise(tri_cursor.side) as usize]].clone();
            let c = self.verts[tri[anticlockwise(tri_cursor.side) as usize]].clone();

            // Check if the vector v1 -> v2 is inside the span of the vectors v1 -> b and v1 -> c
            let orient_ab = orient2d(&v1, &b, &v2);
            let orient_ac = orient2d(&v1, &c, &v2);

            let right_of_ab =  orient_ab >= 0.0;
            let left_of_ac =  orient_ac <= 0.0;
            let inside = right_of_ab && left_of_ac;

            // If we aren't inside then we much continue the search by finding the next triangle
            // that shares this vertex
            if inside {
                break;
            }

            let op_tri = self.opposing_tri(&TriSide{tri: tri_cursor.tri, side: clockwise(tri_cursor.side)}).unwrap();

            // If we arrive back at the start, that's an error. It should NEVER be possible for
            // no triangle to straddle the edge
            if op_tri.tri == start_tri.tri {
                return Err(Error::InternalAlgo);
            }

            tri_cursor.tri = op_tri.tri;
            tri_cursor.side = clockwise(op_tri.side);
        }

        // We now have the triangle that straddles the edge in tri_cursor poiting at the side of v1

        // We now have to destroy all the triangles cut by the new edge to reconstruct them. While
        // doing this we want to keep track of the nodes that fall _below_ and _above_ the new
        // edge. We will also want to keep track of their adjecencies to make reassignment easier
        // when we get to that.

        let mut upper : Vec<usize> = vec![];
        let mut upper_adj : Vec<Option<TriSide>> = vec![];
        if self.tris[tri_cursor.tri][clockwise(tri_cursor.side) as usize] != v2i {
            upper.push(self.tris[tri_cursor.tri][clockwise(tri_cursor.side) as usize]);
            upper_adj.push(self.opposing_tri(&TriSide {tri: tri_cursor.tri, side: anticlockwise(tri_cursor.side)}));
        } else {
            // If a single tri contains both v1 and v2 we aren't cutting any edge and trivially return.
            // The edge is already part of the mesh
            self.edges.insert([v1i, v2i]);
            return Ok(());
        }

        let mut lower : Vec<usize> = vec![];
        let mut lower_adj : Vec<Option<TriSide>> = vec![];
        if self.tris[tri_cursor.tri][anticlockwise(tri_cursor.side) as usize] != v2i {
            lower.push(self.tris[tri_cursor.tri][anticlockwise(tri_cursor.side) as usize]);
            lower_adj.push(self.opposing_tri(&TriSide {tri: tri_cursor.tri, side: clockwise(tri_cursor.side)}));
        } else {
            // If a single tri contains both v1 and v2 we aren't cutting any edge and trivially return.
            // The edge is already part of the mesh
            self.edges.insert([v1i, v2i]);
            return Ok(());
        }

        if self.is_fixed(
            self.tris[tri_cursor.tri][clockwise(tri_cursor.side) as usize],
            self.tris[tri_cursor.tri][anticlockwise(tri_cursor.side) as usize],
        ) {
            // Edge crosses other fixed edge
            return Err(Error::InvalidPoly);
        }

        // We'll tear our exactly as many triangles as we'll put back afterwards so we need to save
        // the triangle ids we tear out to reuse them when rebuilding
        // Kill the inital tri
        let mut dead_tris: Vec<usize> = vec![tri_cursor.tri];

        // Walk the strip of triangles that are cut by the new edge until we arrive at v2i
        let mut iter = 0;
        loop {
            assert!(iter < 100000);
            iter += 1;
            // Get the next tri
            let op_tri = self.opposing_tri(&tri_cursor).unwrap();
            // Kill that tri
            dead_tris.push(op_tri.tri);

            // The new tri shared two vertecies with the old one, those have already been
            // categorized. We need to categorize the new one opposite of the shared side

            // If the vertex is v2, we are done walking the strip and can break the loop
            if self.tris[op_tri.tri][op_tri.side as usize] == v2i {
                // Add the final adjecencies
                upper_adj.push(self.opposing_tri(&TriSide {tri: op_tri.tri, side: clockwise(op_tri.side)}));
                lower_adj.push(self.opposing_tri(&TriSide {tri: op_tri.tri, side: anticlockwise(op_tri.side)}));
                break;
            }

            // We need to figure out if the vertex is above or below the new edge.
            let vi = self.tris[op_tri.tri][op_tri.side as usize];
            let vo = self.verts[vi];

            let orient = orient2d(&v1, &vo, &v2);
            let is_above =  orient > 0.0;

            if is_above {
                // Vertex is above edge
                upper.push(self.tris[op_tri.tri][op_tri.side as usize]);
                upper_adj.push(self.opposing_tri(&TriSide{tri: op_tri.tri, side: clockwise(op_tri.side)}));
                // Next step is the side cut by the edge
                tri_cursor = TriSide{tri: op_tri.tri, side: anticlockwise(op_tri.side)};
            } else {
                // Vertex is below edge
                lower.push(self.tris[op_tri.tri][op_tri.side as usize]);
                lower_adj.push(self.opposing_tri(&TriSide{tri: op_tri.tri, side: anticlockwise(op_tri.side)}));
                // Next step is the side cut by the edge
                tri_cursor = TriSide{tri: op_tri.tri, side: clockwise(op_tri.side)};
            }
        }

        // @SPEED: Maybe this isn't necessary
        let upper_end = upper.len();
        // The pseudo poly will have one for adjecency than the number of verticies. We add a dummy
        // vertex value to make the array indecies match
        upper.push(0xDEADBEEF);
        upper.append(&mut lower);
        upper_adj.append(&mut lower_adj);
        // New tri will contain the new triangles that match corresponding opposites of the
        // upper_adj
        let mut new_tri = vec![TriSide{tri: 0, side: 0}; upper_adj.len()];

        let uv = self.rebuild_tris(&mut dead_tris, &upper, &upper_adj, &mut new_tri, v1i, v2i, false, 0, upper_end-1, None);
        let lv = self.rebuild_tris(&mut dead_tris, &upper, &upper_adj, &mut new_tri, v1i, v2i, true, upper_end+1, upper.len()-1, None);

        // @CLEANUP: I don't think we can ever NOT create a triangle at both top and bottom since
        // the initial cut will always add vertex to both upper and lower
        if let Some(x) = uv {
            if let Some(y) = lv {
                self.adj[y][2] = Some(TriSide{tri: x, side: 2});
            }

            self.inverse_tri[v1i] = TriSide{tri: x, side: 0};
            self.inverse_tri[v2i] = TriSide{tri: x, side: 1};
        }
        if let Some(x) = lv {
            if let Some(y) = uv {
                self.adj[y][2] = Some(TriSide{tri: x, side: 2});
            }

            self.inverse_tri[v1i] = TriSide{tri: x, side: 1};
            self.inverse_tri[v2i] = TriSide{tri: x, side: 0};
        }

        // Adjust the adjecencies of the tris touching the new ones
        for (l, r) in upper_adj.iter().zip(&new_tri) {
            if let Some(l) = l {
                self.adj[l.tri][l.side as usize] = Some(*r);
            }
        }


        dbg!(&upper);
        // Repair the adjecency information if the pseudo-poly contaied repeated verticies
        // (figure 7 in the paper)
        for i in 2..upper.len() {
            if i == upper_end || i-2 == upper_end {
                continue;
            }

            // If the node is present in the array twice the two new triangles will share and edge
            // and will therefore be each other adjecents. They'll have recoded the old triangle
            // though, so we go into them here and patch them directly
            if upper[i] == upper[i-2] {
                println!("{} and {} share an edge", i, i-2);
                let t1 = new_tri[i-1];
                let t2 = new_tri[i];

                self.adj[t1.tri][t1.side as usize] = Some(t2);
                self.adj[t2.tri][t2.side as usize] = Some(t1);
            }
        }

        // Fix the edge
        self.edges.insert([v1i, v2i]);

        return Ok(());
    }


    // @CLEANUP: Maybe we can rework this to use slices instead of vectors for verts and verts_adj
    fn rebuild_tris(&mut self, dead_tris: &mut Vec<usize>, verts: &Vec<usize>, verts_adj: &Vec<Option<TriSide>>, new_tri: &mut [TriSide], v1: usize, v2: usize, swap: bool, start: usize, end: usize, link: Option<TriSide>) -> Option<usize> {
        // Rebuild a triangle strip after tearing it out. We have to be careful to adjust the
        // adjecency relationship between triangles correctly

        if end < start {
            return None;
        }

        let mut ci = start;

        let newt = dead_tris.pop().unwrap();

        let mut v1_link = TriSide{tri: newt, side: 0};
        let mut v2_link = TriSide{tri: newt, side: 1};
        if swap {
            v1_link.side = 1;
            v2_link.side = 0;
        }

        let mut v1_t: Option<TriSide> = None;
        let mut v2_t: Option<TriSide> = None;
        if end-start > 0 {
            for i in start+1..end+1 {
                let v = verts[i];
                let a;
                let b;
                if swap {
                    a = self.verts[v2];
                    b = self.verts[v1];
                } else {
                    a = self.verts[v1];
                    b = self.verts[v2];
                }

                if incircle(a, b, self.verts[verts[ci]], &self.verts[v]) {
                    ci = i;
                }
            }

            if ci >= 1 {
                v2_t = self.rebuild_tris(dead_tris, verts, verts_adj, new_tri, v1, verts[ci], swap, start, ci-1, Some(v2_link))
                .map(|x| TriSide{tri: x, side: 2});
            }
            v1_t = self.rebuild_tris(dead_tris, verts, verts_adj, new_tri, verts[ci], v2, swap, ci+1, end, Some(v1_link))
                .map(|x| TriSide{tri: x, side: 2});
        }

        if ci == end {
            v1_t = verts_adj[ci+1];
            new_tri[ci+1] = v1_link;
        }
        if ci == start {
            v2_t = verts_adj[ci];
            new_tri[ci] = v2_link;
        }

        self.tris[newt][2] = verts[ci];
        self.adj[newt][2] = link;
        if swap {
            self.tris[newt][0] = v2;
            self.tris[newt][1] = v1;

            self.adj[newt][0] = v2_t;
            self.adj[newt][1] = v1_t;
        } else {
            self.tris[newt][0] = v1;
            self.tris[newt][1] = v2;

            self.adj[newt][0] = v1_t;
            self.adj[newt][1] = v2_t;
        }

        self.inverse_tri[verts[ci]].tri = newt;
        self.inverse_tri[verts[ci]].side = 2;

        return Some(newt);
    }
}

#[derive(Debug)]
enum ValidationError {
    InvalidStructure,
    InvalidInverse,
    InvalidAdjecency,
    BadWinding,
    MissingEdges,
}

fn is_winding_correct(tris: &ActiveTriangulation, tri: usize) -> bool {
    let t = tris.tris[tri];
    let a = tris.verts[t[0]];
    let mut b = tris.verts[t[1]];
    let mut c = tris.verts[t[2]];

    b.subv2(&a);
    c.subv2(&a);

    return !b.cross(&c).is_sign_positive();
}

fn validate_mesh(tris: &ActiveTriangulation) -> Option<ValidationError> {
    if tris.verts.len() != tris.inverse_tri.len() { return Some(ValidationError::InvalidStructure); };
    if  tris.tris.len() !=         tris.adj.len() { return Some(ValidationError::InvalidStructure) ; }

    // All verts in the vert -> tri lookup array point to themselves
    {
        for (i, &x) in tris.inverse_tri.iter().enumerate() {
            if x.side > 2 { return Some(ValidationError::InvalidStructure); }
            if tris.tris[x.tri][x.side as usize] != i { return Some(ValidationError::InvalidInverse); }
        }
    };

    // Check the adjecency by just building it again and checking that it's the same. There's
    // only one valid adjencency array for a given topology, so this should be robust
    {
        let adj = build_adjecency(&tris.tris);
        if tris.adj != adj { return Some(ValidationError::InvalidAdjecency); }
    };

    // Check the winding on the triangles
    {
        for i in 0..tris.tris.len() {
            if !is_winding_correct(tris, i) { return Some(ValidationError::BadWinding); }
        }
    }

    // Check that the edges are honored
    {
        let mut edges = tris.edges.clone();
        for i in 0..tris.tris.len() {
            let tri = tris.tris[i];
            for j in 0..3 {
                let next = anticlockwise(j);
                edges.remove(&[tri[j as usize], tri[next as usize]]);
            }
        }
        if edges.len() > 0 {
            return Some(ValidationError::MissingEdges);
        }
    }

    return None;
}


fn incircle(a: Vector, b: Vector, c: Vector, d: &Vector) -> bool {
    let det = robust::incircle(
        robust::Coord{x: a.x, y: a.y},
        robust::Coord{x: b.x, y: b.y},
        robust::Coord{x: c.x, y: c.y},
        robust::Coord{x: d.x, y: d.y},
    );

    return det < 0.0;
}

fn orient2d(a: &Vector, b: &Vector, c: &Vector) -> f64 {
    return robust::orient2d(
        robust::Coord{x: a.x, y: a.y},
        robust::Coord{x: b.x, y: b.y},
        robust::Coord{x: c.x, y: c.y},
    );
}

fn anticlockwise(side: u8) -> u8 {
    return (side + 1) % 3;
}

fn clockwise(side: u8) -> u8 {
    return (side + 2) % 3;
}

#[derive(Debug)]
enum TriangleLocation {
    Inside,
    OnEdge(u8),
    IsCorner(u8),
    Outside(u8),
}

impl TriangleLocation {
    fn is_inside(&self) -> bool {
        return match self {
            TriangleLocation::Inside => true,
            TriangleLocation::OnEdge(_) => true,
            TriangleLocation::IsCorner(_) => true,
            TriangleLocation::Outside(_) => false,
        }
    }
}

fn locate_point_in_tri(p0: &Vector, p1: &Vector, p2: &Vector, p: &Vector) -> TriangleLocation {
    let mut on_edge = None;

    if p0.x == p.x && p0.y == p.y {
        return TriangleLocation::IsCorner(0);
    }

    let orient = orient2d(p0, p1, p);
    if orient == 0.0 {
        on_edge = Some(2);
    } else if orient > 0.0 {
        return TriangleLocation::Outside(2);
    }

    if p1.x == p.x && p1.y == p.y {
        return TriangleLocation::IsCorner(1);
    }

    let orient = orient2d(p1, p2, p);
    if orient == 0.0 {
        on_edge = Some(0);
    } else if orient > 0.0 {
        return TriangleLocation::Outside(0);
    }

    if p2.x == p.x && p2.y == p.y {
        return TriangleLocation::IsCorner(2);
    }

    let orient = orient2d(p2, p0, p);
    if orient == 0.0 {
        on_edge = Some(1);
    } else if orient > 0.0 {
        return TriangleLocation::Outside(1);
    }

    if let Some(x) = on_edge {
        return TriangleLocation::OnEdge(x);
    }

    return TriangleLocation::Inside;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incircle_fuzz() {
        let a = Vector{ x: 1828.0, y: 4071.0 };
        let b = Vector{ x: 1829.0, y: 4080.0 };
        let c = Vector{ x: 1836.0, y: 4088.0 };
        let v = Vector{ x: 1828.0, y: 4096.0 };

        let res = incircle(a, b, c, &v);

        assert!(!res);
    }

    #[test]
    fn create_bbox_from_points() {
        let p = vec![(-1.0, 1.0), (2.0, 3.0)];

        let bbox = BBox::from_points(p.iter().copied());

        assert_eq!(bbox.min.x, -1.0);
        assert_eq!(bbox.min.y, 1.0);
        assert_eq!(bbox.max.x, 2.0);
        assert_eq!(bbox.max.y, 3.0);
    }

    fn box_inside_tri(tri: &ActiveTriangulation, bbox: &BBox) -> bool {
        let first_tri = tri.tris[0];

        let check = |x: f64, y: f64| -> bool {
            return !locate_point_in_tri(&tri.verts[first_tri[0]], &tri.verts[first_tri[1]], &tri.verts[first_tri[2]], &Vector{x, y}).is_inside();
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

        let tri = ActiveTriangulation::super_tri_of_bbox(&bbox, 0);

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

        assert!(validate_mesh(&tri).is_none());

        assert!(box_inside_tri(&tri, &bbox));
    }

    #[test]
    fn create_tris_from_array() {
        let verts = [
            Vector{x:   0.0, y:   0.0},
            Vector{x:   0.0, y: 100.0},
            Vector{x: 100.0, y: 100.0},
            Vector{x: 100.0, y:   0.0},
        ];
        let tris = [
            [0, 1, 2],
            [0, 2, 3],
        ];

        let tri = ActiveTriangulation::explicit(&verts, &tris);

        assert!(validate_mesh(&tri).is_none());

        assert_eq!(tri.verts.len(), 4);
        assert_eq!(tri.inverse_tri.len(), 4);

        assert_eq!(tri.tris.len(), 2);
        assert_eq!(tri.adj.len(), 2);

        // We haven't inserted any edges
        assert!(tri.edges.is_empty());

        // Check if adjecencies were computed for the right edges. The adjecencies themselves were
        // checked as part validation
        assert!(tri.adj[0][0].is_none());
        assert!(tri.adj[0][1].is_some());
        assert!(tri.adj[0][2].is_none());
        assert!(tri.adj[1][0].is_none());
        assert!(tri.adj[1][1].is_none());
        assert!(tri.adj[1][2].is_some());
    }

    #[test]
    fn add_edge_known_tri() {
        // Does not generate a valid mesh!
        let verts = [
            Vector{x:   0.0, y:   0.0},
            Vector{x:   0.0, y: 100.0},
            Vector{x: 100.0, y: 100.0},
            Vector{x: 100.0, y:   0.0},
        ];
        let tris = [
            [1, 2, 0],
            [0, 2, 3],
        ];

        let mut tri = ActiveTriangulation::explicit(&verts, &tris);

        tri.add_edge(1, 3);

        assert!(validate_mesh(&tri).is_none());
    }

    #[test]
    fn add_edge_little_bit_advanced() {
        // Does not generate a valid mesh!
        let verts = [
            Vector{x:   0.0, y:  50.0},
            Vector{x:  50.0, y: 100.0},
            Vector{x: 100.0, y: 100.0},
            Vector{x: 150.0, y:  50.0},
            Vector{x: 100.0, y:   0.0},
            Vector{x:  50.0, y:   0.0},

            // For the external tris
            Vector{x:   0.0, y: 100.0},
            Vector{x: 125.0, y: 125.0},
            Vector{x: 150.0, y: 100.0},
            Vector{x: 150.0, y:   0.0},
            Vector{x: 125.0, y: -25.0},
            Vector{x:   0.0, y:   0.0},
        ];
        let tris = [
            [0, 1, 5],
            [1, 2, 5],
            [5, 2, 4],
            [3, 4, 2],

            // External tris
            [0, 6, 1],
            [1, 7, 2],
            [2, 8, 3],
            [3, 9, 4],
            [4, 10, 5],
            [5, 11, 0],
        ];

        let mut tri = ActiveTriangulation::explicit(&verts, &tris);

        tri.add_edge(0, 3).unwrap();

        if let Some(err) = validate_mesh(&tri) {
            panic!("{:?}", err);
        }
    }

    #[ignore] // Currently broken due to incomplete implementation
    #[test]
    fn add_edge_figure_7() {
        // Does not generate a valid mesh!
        let verts = [
            Vector{x:   0.0, y:  50.0},
            Vector{x:  50.0, y: 100.0},
            Vector{x: 100.0, y: 100.0},
            Vector{x: 150.0, y:  50.0},
            Vector{x: 100.0, y:   0.0},
            Vector{x:  50.0, y:   0.0},
            Vector{x:  50.0, y:  25.0},
        ];
        let tris = [
            [0, 1, 5],
            [1, 6, 5],
            [1, 2, 6],
            [6, 2, 5],
            [5, 2, 4],
            [3, 4, 2],
        ];

        let mut tri = ActiveTriangulation::explicit(&verts, &tris);

        tri.add_edge(0, 3).unwrap();

        assert!(validate_mesh(&tri).is_none());
    }

    #[test]
    fn rebuild_tri_upper() {
        // Does not generate a valid mesh!
        let verts = [
            Vector{x:   0.0, y:   0.0},
            Vector{x:   0.0, y: 100.0},
            Vector{x: 100.0, y: 100.0},
            Vector{x: 100.0, y:   0.0},
        ];
        let tris = [
            [1, 2, 0],
            [0, 2, 3],
        ];

        let mut tri = ActiveTriangulation::explicit(&verts, &tris);

        // Compute the upper node
        {
            let mut dead_tris = [0]
                .to_vec();
            let mut verts = [
                0
            ].to_vec();
            let mut verts_adj = [
                None, None
            ].to_vec();
            let mut new_tri = vec![TriSide{tri:0, side:0}; verts_adj.len()];

            let uv = tri.rebuild_tris(&mut dead_tris, &mut verts, &mut verts_adj, &mut new_tri, 1, 3, false, 0, 0, None);

            assert!(uv.is_some());
            assert_eq!(uv.unwrap(), 0);
            // This HAS to be the node opposite the "new" edge. It's an ugly hardcoded hack
            assert_eq!(tri.tris[uv.unwrap()][2], 0);

            {
                let adj = new_tri[0];
                assert_eq!(tri.tris[adj.tri][adj.side as usize], 3);
            }
            {
                let adj = new_tri[1];
                assert_eq!(tri.tris[adj.tri][adj.side as usize], 1);
            }
        }

        // Then the lower node
        {
            let mut dead_tris = [1]
                .to_vec();
            let mut verts = [
                2
            ].to_vec();
            let mut verts_adj = [
                None, None
            ].to_vec();
            let mut new_tri = vec![TriSide{tri:0, side:0}; verts_adj.len()];

            let lv = tri.rebuild_tris(&mut dead_tris, &mut verts, &mut verts_adj, &mut new_tri, 1, 3, true, 0, 0, None);

            assert!(lv.is_some());
            assert_eq!(lv.unwrap(), 1);
            // This HAS to be the node opposite the "new" edge. It's an ugly hardcoded hack
            assert_eq!(tri.tris[lv.unwrap()][2], 2);

            {
                let adj = new_tri[0];
                assert_eq!(tri.tris[adj.tri][adj.side as usize], 3);
            }
            {
                let adj = new_tri[1];
                assert_eq!(tri.tris[adj.tri][adj.side as usize], 1);
            }
        }
    }

    #[test]
    fn find_point_inside_start_tri() {
        let bbox = BBox{
            min: Vector{x: -1.0, y: -1.0},
            max: Vector{x:  1.0, y:  1.0},
        };

        let tris = ActiveTriangulation::super_tri_of_bbox(&bbox, 0);
        let (tri, _) = tris.find_containg_tri(&Vector{x: 0.0, y: 0.0}, None).unwrap();

        assert_eq!(tri, 0);
    }

    #[test]
    fn find_no_tri_point_outside_tris() {
        let bbox = BBox{
            min: Vector{x: -1.0, y: -1.0},
            max: Vector{x:  1.0, y:  1.0},
        };

        let tris = ActiveTriangulation::super_tri_of_bbox(&bbox, 0);
        // @FRAGILE: the (10, 10) here is arbitrary. It''s outside the bbox, but since the triangle
        // is allowed to extend outside of that it's not guaranteed to be outide the triangle. We
        // should do some sort of triangle bounding box to find a point that's for sure outside the
        // triangle
        let res = tris.find_containg_tri(&Vector{x: 10.0, y: 10.0}, None);

        assert!(res.is_err());
    }

    #[test]
    fn split_a_single_triangle() {
        let bbox = BBox{
            min: Vector{x: -1.0, y: -1.0},
            max: Vector{x:  1.0, y:  1.0},
        };
        let mut tris = ActiveTriangulation::super_tri_of_bbox(&bbox, 0);
        let p = Vector{x: 0.0, y: 0.0};

        tris.split_tri(0, &p);

        assert_eq!(tris.tris.len(), 3);
        assert!(validate_mesh(&tris).is_none());
    }

    #[test]
    fn add_first_point() {
        let bbox = BBox{
            min: Vector{x: -1.0, y: -1.0},
            max: Vector{x:  1.0, y:  1.0},
        };
        let mut tris = ActiveTriangulation::super_tri_of_bbox(&bbox, 0);
        let p = Vector{x: 0.0, y: 0.0};

        let vert = tris.add_point(&p).unwrap();

        assert_eq!(tris.tris.len(), 3);
        assert!(validate_mesh(&tris).is_none());

        // Count the number of triangles include the new vertex. Since we split one triangle into
        // 3, all three triangles must include the new vertex.
        let num_including = tris.tris.iter().flatten().fold(0, |a, x|{
            if *x == vert { a + 1 } else { a }
        });
        assert_eq!(num_including, 3);
    }

    #[test]
    fn add_point_on_side() {
        let bbox = BBox{
            min: Vector{x: -1.0, y: -1.0},
            max: Vector{x:  1.0, y:  1.0},
        };
        let mut tris = ActiveTriangulation::super_tri_of_bbox(&bbox, 0);

        let v1 = tris.add_point(&Vector{x: 1.0, y: 0.0}).unwrap();
        let v2 = tris.add_point(&Vector{x:-1.0, y: 0.0}).unwrap();

        {
            // There has to be an edge between the v1 and v2
            let mut found = false;
            for tri in &tris.tris {
                if tri.contains(&v1) && tri.contains(&v2) {
                    found = true;
                    break;
                }
            }
            assert!(found);
        }

        // Now add a vertex right between them. This will be ON the previous asserted edge
        tris.add_point(&Vector{x: 0.0, y: 0.0}).unwrap();

        assert!(validate_mesh(&tris).is_none());
    }
}
