use crabby::{ModelData, RealMatrix};
use rand::Rng;

fn generate_large_dataset(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();
    for _ in 0..n {
        data.push(Rng::gen_range(&mut rng, 0.0, 1.0));
    }
    data
}

fn main() {
    let x = RealMatrix::from_vec(generate_large_dataset(1000000), 1000000, 1);
    let y = RealMatrix::from_vec(&x.values.iter().map(|x| x * 2.0).collect::<Vec<f64>>());
    let model_data = ModelData::new(&x, &y);
}
