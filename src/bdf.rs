// bdf.rs
use crate::common::*;
use crate::dense_output::*;
use crate::IntegratorError;
use nalgebra::DMatrix;
use anyhow::Result;

pub struct BdfSolver {
    pub fun: Box<dyn Fn(f64, &[f64]) -> Vec<f64>>,
    pub t: f64,
    pub y: Vec<f64>,
    pub t_bound: f64,
    pub t_old: Option<f64>,
    pub status: SolverStatus,
    pub direction: f64,
    pub n: usize,
    pub h_abs: f64,
    pub h_abs_old: Option<f64>,
    pub rtol: f64,
    pub atol: f64,
    pub max_step: f64,
    pub newton_tol: f64,
    pub nfev: usize,
    pub njev: usize,
    pub nlu: usize,
    pub j: Option<DMatrix<f64>>,
    pub lu: Option<LUDecomposition>,
    pub gamma: Vec<f64>,
    pub alpha: Vec<f64>,
    pub error_const: Vec<f64>,
    pub d: Vec<Vec<f64>>,
    pub order: usize,
    pub n_equal_steps: usize,
}

impl BdfSolver {
    pub fn new(
        fun: Box<dyn Fn(f64, &[f64]) -> Vec<f64>>,
        t0: f64,
        y0: Vec<f64>,
        t_bound: f64,
        rtol: f64,
        atol: f64,
        max_step: f64,
        first_step: Option<f64>,
    ) -> Self {
        let n = y0.len();
        let direction = if t_bound != t0 {
            (t_bound - t0).signum()
        } else {
            1.0
        };
        
        let (rtol, atol) = validate_tol(rtol, atol);
        
        let f0 = fun(t0, &y0);
        
        let h_abs = if let Some(first_step) = first_step {
            validate_first_step(first_step, t0, t_bound)
        } else {
            select_initial_step(&*fun, t0, &y0, t_bound, max_step, &f0, direction, 1, rtol, atol)
        };
        
        let newton_tol = (10.0 * EPS / rtol).max((0.03_f64).min(rtol.sqrt()));
        
        // BDF coefficients
        let kappa = vec![0.0, -0.1850, -1.0/9.0, -0.0823, -0.0415, 0.0];
        let gamma = {
            let mut gamma = vec![0.0];
            for i in 1..=MAX_ORDER {
                gamma.push(gamma[i-1] + 1.0 / i as f64);
            }
            gamma
        };
        
        let alpha: Vec<f64> = (0..=MAX_ORDER).map(|i| (1.0 - kappa[i]) * gamma[i]).collect();
        let error_const: Vec<f64> = (0..=MAX_ORDER).map(|i| 
            kappa[i] * gamma[i] + 1.0 / (i as f64 + 1.0)
        ).collect();
        
        let mut d = vec![vec![0.0; n]; MAX_ORDER + 3];
        d[0] = y0.clone();
        d[1] = f0.iter().map(|&f| f * h_abs * direction).collect();
        
        Self {
            fun,
            t: t0,
            y: y0,
            t_bound,
            t_old: None,
            status: SolverStatus::Running,
            direction,
            n,
            h_abs,
            h_abs_old: None,
            rtol,
            atol,
            max_step,
            newton_tol,
            nfev: 1, // f0 evaluation
            njev: 0,
            nlu: 0,
            j: None,
            lu: None,
            gamma,
            alpha,
            error_const,
            d,
            order: 1,
            n_equal_steps: 0,
        }
    }
    
    pub fn step(&mut self) -> Result<()> {
        let t = self.t;
        let min_step = 10.0 * (t + self.direction * f64::EPSILON - t).abs();
        
        if self.h_abs > self.max_step {
            let factor = self.max_step / self.h_abs;
            change_d(&mut self.d, self.order, factor);
            self.h_abs = self.max_step;
            self.n_equal_steps = 0;
        } else if self.h_abs < min_step {
            let factor = min_step / self.h_abs;
            change_d(&mut self.d, self.order, factor);
            self.h_abs = min_step;
            self.n_equal_steps = 0;
        }
        
        let mut step_accepted = false;
        let mut current_jac = false;
        
        while !step_accepted {
            if self.h_abs < min_step {
                self.status = SolverStatus::Failed;
                return Err(IntegratorError::StepTooSmall.into());
            }
            
            let h = self.h_abs * self.direction;
            let mut t_new = t + h;
            
            if self.direction * (t_new - self.t_bound) > 0.0 {
                t_new = self.t_bound;
                let factor = (t_new - t).abs() / self.h_abs;
                change_d(&mut self.d, self.order, factor);
                self.n_equal_steps = 0;
                self.lu = None;
            }
            
            let h = t_new - t;
            let h_abs = h.abs();
            
            // Predict solution
            let y_predict: Vec<f64> = (0..self.n)
                .map(|i| (0..=self.order).map(|j| self.d[j][i]).sum())
                .collect();
                
            let scale: Vec<f64> = y_predict.iter()
                .map(|&y| self.atol + self.rtol * y.abs())
                .collect();
                
            let psi: Vec<f64> = (0..self.n)
                .map(|i| {
                    (1..=self.order)
                        .map(|j| self.d[j][i] * self.gamma[j])
                        .sum::<f64>() / self.alpha[self.order]
                })
                .collect();
                
            let c = h / self.alpha[self.order];
            let mut converged = false;
            
            while !converged {
                if self.lu.is_none() {
                    let f_val = (self.fun)(t_new, &y_predict);
                    self.nfev += 1;
                    
                    if self.j.is_none() || !current_jac {
                        self.j = Some(num_jac(&*self.fun, t_new, &y_predict, &f_val, self.atol));
                        self.njev += 1;
                        current_jac = true;
                    }
                    
                    let identity = DMatrix::identity(self.n, self.n);
                    let matrix = &identity - c * self.j.as_ref().unwrap();
                    
                    self.lu = Some(LUDecomposition::new(matrix)?);
                    self.nlu += 1;
                }
                
                let (conv, n_iter, y_new, d) = self.solve_bdf_system(
                    t_new, &y_predict, c, &psi, &scale
                )?;
                
                converged = conv;
                
                if !converged {
                    if current_jac {
                        let factor = 0.5;
                        self.h_abs *= factor;
                        change_d(&mut self.d, self.order, factor);
                        self.n_equal_steps = 0;
                        self.lu = None;
                        break;
                    }
                    current_jac = true;
                    self.lu = None;
                } else {
                    let safety = 0.9 * (2 * NEWTON_MAXITER + 1) as f64 / 
                                (2 * NEWTON_MAXITER + n_iter) as f64;
                    
                    let scale_new: Vec<f64> = y_new.iter()
                        .map(|&y| self.atol + self.rtol * y.abs())
                        .collect();
                        
                    let error: Vec<f64> = d.iter()
                        .map(|&di| self.error_const[self.order] * di)
                        .collect();
                        
                    let error_norm = norm(&error.iter().zip(&scale_new)
                        .map(|(&e, &s)| e / s).collect::<Vec<_>>());
                    
                    if error_norm > 1.0 {
                        let factor = MIN_FACTOR.max(
                            safety * error_norm.powf(-1.0 / (self.order as f64 + 1.0))
                        );
                        self.h_abs *= factor;
                        change_d(&mut self.d, self.order, factor);
                        self.n_equal_steps = 0;
                    } else {
                        step_accepted = true;
                        self.t_old = Some(self.t);
                        self.t = t_new;
                        self.y = y_new;
                        self.h_abs = h_abs;
                        
                        // Update differences
                        let d_new = d;
                        self.d[self.order + 2] = d_new.iter().zip(&self.d[self.order + 1])
                            .map(|(&d_new_i, &d_old_i)| d_new_i - d_old_i)
                            .collect();
                        self.d[self.order + 1] = d_new;
                        
                        for i in (0..=self.order).rev() {
                            for j in 0..self.n {
                                self.d[i][j] += self.d[i + 1][j];
                            }
                        }
                        
                        self.n_equal_steps += 1;
                        
                        if self.direction * (self.t - self.t_bound) >= 0.0 {
                            self.status = SolverStatus::Finished;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn solve_bdf_system(
        &mut self,
        t_new: f64,
        y_predict: &[f64],
        c: f64,
        psi: &[f64],
        scale: &[f64],
    ) -> Result<(bool, usize, Vec<f64>, Vec<f64>)> {
        let mut d = vec![0.0; self.n];
        let mut y = y_predict.to_vec();
        let mut dy_norm_old = None;
        let mut converged = false;
        
        for k in 0..NEWTON_MAXITER {
            let f = (self.fun)(t_new, &y);
            self.nfev += 1;
            
            if !f.iter().all(|&x| x.is_finite()) {
                return Err(IntegratorError::InvalidFunction.into());
            }
            
            let rhs: Vec<f64> = f.iter().zip(psi).zip(&d)
                .map(|((&fi, &psi_i), &di)| c * fi - psi_i - di)
                .collect();
                
            let dy = if let Some(ref lu) = self.lu {
                lu.solve(&rhs)?
            } else {
                return Err(IntegratorError::NewtonFailed.into());
            };
            
            let dy_norm = norm(&dy.iter().zip(scale)
                .map(|(&dy_i, &s)| dy_i / s).collect::<Vec<_>>());
            
            let rate: Option<f64> = if let Some(old_norm) = dy_norm_old {
                Some(dy_norm / old_norm)
            } else {
                None
            };
            
            if let Some(r) = rate {
                if r >= 1.0 || r.powi(NEWTON_MAXITER as i32 - k as i32) / (1.0 - r) * dy_norm > self.newton_tol {
                    break;
                }
            }
            
            for i in 0..self.n {
                y[i] += dy[i];
                d[i] += dy[i];
            }
            
            if dy_norm == 0.0 || (rate.is_some() && rate.unwrap() / (1.0 - rate.unwrap()) * dy_norm < self.newton_tol) {
                converged = true;
                break;
            }
            
            dy_norm_old = Some(dy_norm);
        }
        
        Ok((converged, NEWTON_MAXITER, y, d))
    }
    
    pub fn dense_output(&self) -> BdfDenseOutput {
        BdfDenseOutput::new(
            self.t_old.unwrap(),
            self.t,
            self.h_abs * self.direction,
            self.order,
            self.d[..=self.order].iter().map(|v| v.clone()).collect(),
        )
    }
}