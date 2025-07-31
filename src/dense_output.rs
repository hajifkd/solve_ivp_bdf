// dense_output.rs

pub trait DenseOutput {
    fn call(&self, t: f64) -> Vec<f64>;
    fn t_min(&self) -> f64;
    fn t_max(&self) -> f64;
}

#[derive(Debug, Clone)]
pub struct BdfDenseOutput {
    t_old: f64,
    t: f64,
    order: usize,
    d: Vec<Vec<f64>>,
    t_shift: Vec<f64>,
    denom: Vec<f64>,
}

impl BdfDenseOutput {
    pub fn new(t_old: f64, t: f64, h: f64, order: usize, d: Vec<Vec<f64>>) -> Self {
        let t_shift: Vec<f64> = (0..order).map(|i| t - h * i as f64).collect();
        let denom: Vec<f64> = (0..order).map(|i| h * (1.0 + i as f64)).collect();
        
        Self {
            t_old,
            t,
            order,
            d,
            t_shift,
            denom,
        }
    }
}

impl DenseOutput for BdfDenseOutput {
    fn call(&self, t: f64) -> Vec<f64> {
        let x: Vec<f64> = self.t_shift.iter().zip(&self.denom)
            .map(|(&shift, &den)| (t - shift) / den)
            .collect();
            
        let mut p = vec![1.0];
        for i in 1..self.order {
            p.push(p[i-1] * x[i-1]);
        }
        
        let n = self.d[0].len();
        let mut y = self.d[0].clone();
        
        for i in 0..n {
            for j in 1..self.order {
                y[i] += self.d[j][i] * p[j];
            }
        }
        
        y
    }
    
    fn t_min(&self) -> f64 {
        self.t_old.min(self.t)
    }
    
    fn t_max(&self) -> f64 {
        self.t_old.max(self.t)
    }
}