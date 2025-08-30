

// brute-force approximate nearest neighbor search using cosine
// similarity

// store all inserted vectors in memory
// computes cosine similraty between the query vector 
// and every stored vector
// returns the top k most similar vectors for some vector

use ndarray::array;
use ndarray::Array1;

fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot = a.dot(b);
    let norm_product = a.dot(a).sqrt() * b.dot(b).sqrt();
    if norm_product == 0.0 {
        0.0
    } else {
        dot / norm_product
    }
}

pub struct ANNIndex {
    data: Vec<Array1<f32>>,
    sz: usize,
}

impl ANNIndex {
    pub fn new() -> Self {
        Self { data: Vec::new(), sz: 0 }
    }

    pub fn insert(&mut self, v: Array1<f32>) {
        if self.data.is_empty() {
            self.sz = v.len();
        } else {
            assert_eq!(v.len(), self.sz, "Vectors in ANN index need to be same size!");
        }
        self.data.push(v);
    }

    pub fn get_top_k(&self, query: Array1<f32>, k: usize) -> Vec<(Array1<f32>, f32)> {
        if !self.data.is_empty() {
            assert_eq!(query.len(), self.sz, "Can only query ANN index with same size vector!");
        }
        // returns vector of tuples
        // each tuple contains the vector and the cosine similarity between query and that vector
        let mut scored: Vec<(Array1<f32>, f32)> = self
            .data
            .iter()
            .map(|v| {
                let sim = cosine_similarity(&query, v);
                (v.clone(), sim)
            })
            .collect();
        // sort of descending score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // get top-k
        scored.into_iter().take(k).collect()
    }
}


pub fn run() {
    let mut ann = ANNIndex::new();
    println!("Brute force ANN search using cosine similarity");
    let a: Array1<f32> = array![1.0, 2.0, 3.0];
    let b: Array1<f32> = array![4.0, 5.0, 6.0];
    let c: Array1<f32> = array![7.0, 8.0, 9.0];
    let d: Array1<f32> = array![-1.0, -2.0, -3.0];
    let e: Array1<f32> = array![-4.0, -5.0, -6.0];
    let f: Array1<f32> = array![-7.0, -8.0, -9.0];

    ann.insert(a);
    ann.insert(b);
    ann.insert(c);
    ann.insert(d);
    ann.insert(e);
    ann.insert(f);

    let x: Array1<f32> = array![1.0, 3.0, 5.0];

    let x_top_3 = ann.get_top_k(x.clone(), 3);
    println!("Top 3 most similar vectors to {:?}", x);
    for (i, (vec, score)) in x_top_3.iter().enumerate() {
        println!("Result {}:", i+1);
        println!("  Vector: {:?}", vec);
        println!("  Score: {:.4}", score);
    }

    let x_top_1 = ann.get_top_k(x.clone(), 1);
    println!("Top 1 most similar vectors to {:?}", x);
    for (i, (vec, score)) in x_top_1.iter().enumerate() {
        println!("Result {}:", i+1);
        println!("  Vector: {:?}", vec);
        println!("  Score: {:.4}", score);
    }

    println!("Now inserting a wrong-sized vector — should panic:");
    let g: Array1<f32> = array![6.0, 7.0];
    ann.insert(g);

}



