use criterion::{criterion_group, criterion_main, Criterion};
use maps::mapbox::pmtile::File;
use maps::mapbox::pmtile::Renderer;

pub struct StubRenderer { }

impl Renderer for StubRenderer {
    type Layer = ();

    fn upload_layer(_data: &Vec<maps::mapbox::GlVert>, labels: Vec<maps::mapbox::pmtile::Label>) -> Self::Layer {
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut tiles = File::<StubRenderer>::new("aalborg.pmtiles");
    c.bench_function("load_tile", |b| b.iter(|| {
        tiles.load_tile(8644, 5015, 14);
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
