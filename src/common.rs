// common.rs
use nalgebra::{DMatrix, DVector};
use anyhow::{Result};
use crate::IntegratorError;

pub const EPS: f64 = f64::EPSILON;
pub const MAX_ORDER: usize = 5;
pub const NEWTON_MAXITER: usize = 4;
pub const MIN_FACTOR: f64 = 0.2;
pub const MAX_FACTOR: f64 = 10.0;

#[derive(Debug, Clone, PartialEq)]
pub enum SolverStatus {
    Running,
    Finished,
    Failed,
}

pub fn norm(x: &[f64]) -> f64 {
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
    (sum_sq / x.len() as f64).sqrt()
}

pub fn validate_tol(rtol: f64, atol: f64) -> (f64, f64) {
    let rtol = if rtol < 100.0 * EPS {
        eprintln!("rtol too small, using {}", 100.0 * EPS);
        100.0 * EPS
    } else {
        rtol
    };
    
    if atol < 0.0 {
        panic!("atol must be positive");
    }
    
    (rtol, atol)
}

pub fn validate_first_step(first_step: f64, t0: f64, t_bound: f64) -> f64 {
    if first_step <= 0.0 {
        panic!("first_step must be positive");
    }
    if first_step > (t_bound - t0).abs() {
        panic!("first_step exceeds bounds");
    }
    first_step
}

pub fn select_initial_step(
    fun: impl Fn(f64, &[f64]) -> Vec<f64>,
    t0: f64,
    y0: &[f64],
    t_bound: f64,
    max_step: f64,
    f0: &[f64],
    direction: f64,
    order: usize,
    rtol: f64,
    atol: f64,
) -> f64 {
    if y0.is_empty() {
        return f64::INFINITY;
    }
    
    let interval_length = (t_bound - t0).abs();
    if interval_length == 0.0 {
        return 0.0;
    }
    
    let scale: Vec<f64> = y0.iter().map(|&yi| atol + yi.abs() * rtol).collect();
    let d0 = norm(&y0.iter().zip(&scale).map(|(&yi, &si)| yi / si).collect::<Vec<_>>());
    let d1 = norm(&f0.iter().zip(&scale).map(|(&fi, &si)| fi / si).collect::<Vec<_>>());
    
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    
    let h0 = h0.min(interval_length);
    
    let y1: Vec<f64> = y0.iter().zip(f0).map(|(&yi, &fi)| yi + h0 * direction * fi).collect();
    let f1 = fun(t0 + h0 * direction, &y1);
    
    let diff: Vec<f64> = f1.iter().zip(f0).map(|(f1i, f0i)| f1i - f0i).collect();
    let d2 = norm(&diff.iter().zip(&scale).map(|(&di, &si)| di / si).collect::<Vec<_>>()) / h0;
    
    let h1 = if d1 <= 1e-15 && d2 <= 1e-15 {
        (1e-6_f64).max(h0 * 1e-3)
    } else {
        (0.01 / d1.max(d2)).powf(1.0 / (order as f64 + 1.0))
    };
    
    (100.0 * h0).min(h1).min(interval_length).min(max_step)
}

pub fn compute_r(order: usize, factor: f64) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(order + 1, order + 1);
    
    for i in 1..=order {
        for j in 1..=order {
            m[(i, j)] = (i as f64 - 1.0 - factor * j as f64) / i as f64;
        }
    }
    m[(0, 0)] = 1.0;
    for j in 1..=order {
        m[(0, j)] = 1.0;
    }
    
    // Compute cumulative product along axis 0
    let mut result = m.clone();
    for i in 1..=order {
        for j in 0..=order {
            result[(i, j)] = m[(i, j)] * result[(i - 1, j)];
        }
    }
    result
}

pub fn change_d(d: &mut [Vec<f64>], order: usize, factor: f64) {
    let r = compute_r(order, factor);
    let u = compute_r(order, 1.0);
    let ru = &r * &u;
    
    let n = d[0].len();
    let mut d_matrix = DMatrix::zeros(order + 1, n);
    
    for i in 0..=order {
        for j in 0..n {
            d_matrix[(i, j)] = d[i][j];
        }
    }
    
    let result = ru.transpose() * d_matrix;
    
    for i in 0..=order {
        for j in 0..n {
            d[i][j] = result[(i, j)];
        }
    }
}

pub fn num_jac(
    fun: impl Fn(f64, &[f64]) -> Vec<f64>,
    t: f64,
    y: &[f64],
    f: &[f64],
    threshold: f64,
) -> DMatrix<f64> {
    let n = y.len();
    let mut jac = DMatrix::zeros(n, n);
    let h = (EPS.sqrt() * threshold).max(EPS.sqrt() * norm(y));
    
    for j in 0..n {
        let mut y_plus = y.to_vec();
        y_plus[j] += h;
        let f_plus = fun(t, &y_plus);
        
        for i in 0..n {
            jac[(i, j)] = (f_plus[i] - f[i]) / h;
        }
    }
    
    jac
}

// LU decomposition wrapper - 修正版
pub struct LUDecomposition {
    matrix: DMatrix<f64>,
}

impl LUDecomposition {
    pub fn new(matrix: DMatrix<f64>) -> Result<Self> {
        // matrixが正則かどうかチェック
        if matrix.determinant().abs() < 1e-14 {
            Err(IntegratorError::SingularMatrix.into())
        } else {
            Ok(Self { matrix })
        }
    }
    
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        let b_vec = DVector::from_vec(b.to_vec());
        
        // LU分解を使って連立方程式を解く
        self.matrix.clone().lu().solve(&b_vec)
            .map(|solution| solution.as_slice().to_vec())
            .ok_or_else(|| IntegratorError::SingularMatrix.into())
    }
}