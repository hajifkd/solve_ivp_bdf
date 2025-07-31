// lib.rs
use thiserror::Error;

pub mod common;
pub mod bdf;
pub mod dense_output;


use common::*;
use bdf::*;
use dense_output::*;

#[derive(Debug, Error, Clone)]
pub enum IntegratorError {
    #[error("Step size too small: required step size is less than spacing between numbers")]
    StepTooSmall,
    
    #[error("LU decomposition failed: matrix is singular")]
    SingularMatrix,
    
    #[error("Newton iteration failed to converge")]
    NewtonFailed,
    
    #[error("Invalid function evaluation: non-finite values detected")]
    InvalidFunction,
    
    #[error("Integration bounds error: {message}")]
    BoundsError { message: String },
    
    #[error("Tolerance validation error: {message}")]
    ToleranceError { message: String },
    
    #[error("Initial step validation error: {message}")]
    InitialStepError { message: String },
    
    #[error("General integration error: {message}")]
    IntegrationError { message: String },
}

#[derive(Debug)]
pub enum SolveResult {
    Success {
        t: Vec<f64>,
        y: Vec<Vec<f64>>,
        dense_output: Option<Vec<BdfDenseOutput>>,
    },
    Failed {
        error: anyhow::Error,
        t: Vec<f64>,
        y: Vec<Vec<f64>>,
    },
    EventTerminated {
        t: Vec<f64>,
        y: Vec<Vec<f64>>,
        event_index: usize,
    },
}

pub type RhsFunction = Box<dyn Fn(f64, &[f64]) -> Vec<f64>>;
pub type EventFunction = Box<dyn Fn(f64, &[f64]) -> bool>;

pub struct Event {
    pub function: EventFunction,
    pub terminal: bool,
}

pub struct SolveIvpOptions {
    pub rtol: f64,
    pub atol: f64,
    pub max_step: f64,
    pub first_step: Option<f64>,
    pub events: Vec<Event>,
    pub dense_output: bool,
}

impl Default for SolveIvpOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-3,
            atol: 1e-6,
            max_step: f64::INFINITY,
            first_step: None,
            events: Vec::new(),
            dense_output: false,
        }
    }
}

pub fn solve_ivp_bdf(
    fun: RhsFunction,
    t0: f64,
    t1: f64,
    y0: Vec<f64>,
    options: Option<SolveIvpOptions>,
) -> SolveResult {
    let opts = options.unwrap_or_default();
    
    let mut solver = BdfSolver::new(
        fun,
        t0,
        y0,
        t1,
        opts.rtol,
        opts.atol,
        opts.max_step,
        opts.first_step,
    );
    
    let mut t_values = vec![t0];
    let mut y_values = vec![solver.y.clone()];
    let mut dense_outputs = if opts.dense_output {
        Some(Vec::<BdfDenseOutput>::new())
    } else { 
        None 
    };
    
    while solver.status == SolverStatus::Running {
        match solver.step() {
            Ok(()) => {
                // Step was successful
            }
            Err(e) => {
                return SolveResult::Failed {
                    error: e,
                    t: t_values,
                    y: y_values,
                };
            }
        }
        
        // Check events
        for (i, event) in opts.events.iter().enumerate() {
            let event_value = (event.function)(solver.t, &solver.y);
            // Simple event detection (should be improved for production use)
            if event_value {
                if event.terminal {
                    t_values.push(solver.t);
                    y_values.push(solver.y.clone());
                    return SolveResult::EventTerminated {
                        t: t_values,
                        y: y_values,
                        event_index: i,
                    };
                }
            }
        }
        
        t_values.push(solver.t);
        y_values.push(solver.y.clone());
        
        if let Some(ref mut dense_vec) = dense_outputs {
            if solver.t_old.is_some() {
                dense_vec.push(solver.dense_output());
            }
        }
    }
    
    SolveResult::Success {
        t: t_values,
        y: y_values,
        dense_output: dense_outputs,
    }
}


// lib.rs の最後に追記

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_harmonic_oscillator_success() {
        // Simple harmonic oscillator: y'' + y = 0
        // Converted to system: y' = z, z' = -y
        // Analytical solution: y(t) = cos(t), z(t) = -sin(t)
        let fun = Box::new(|_t: f64, y: &[f64]| -> Vec<f64> {
            vec![y[1], -y[0]]
        });
        
        let t0 = 0.0;
        let t1 = 2.0 * std::f64::consts::PI; // One full period
        let y0 = vec![1.0, 0.0]; // y(0) = 1, y'(0) = 0
        
        let options = SolveIvpOptions {
            rtol: 1e-8,
            atol: 1e-10,
            dense_output: true,
            ..Default::default()
        };
        
        let result = solve_ivp_bdf(fun, t0, t1, y0, Some(options));
        
        match result {
            SolveResult::Success { t, y, dense_output } => {
                // Check that we reached the final time
                assert!((t.last().unwrap() - t1).abs() < 1e-10);
                
                // Check that the final state is close to the initial state (periodic solution)
                let y_final = y.last().unwrap();
                assert!((y_final[0] - 1.0).abs() < 1e-5, "y position error: {}", (y_final[0] - 1.0).abs());
                assert!((y_final[1] - 0.0).abs() < 1e-5, "y velocity error: {}", (y_final[1] - 0.0).abs());
                
                // Check that dense output is available
                assert!(dense_output.is_some());
                
                println!("Harmonic oscillator test successful!");
                println!("Final error in position: {}", (y_final[0] - 1.0).abs());
                println!("Final error in velocity: {}", (y_final[1] - 0.0).abs());
            }
            SolveResult::Failed { error, .. } => {
                panic!("Integration failed: {}", error);
            }
            SolveResult::EventTerminated { .. } => {
                panic!("Unexpected event termination");
            }
        }
    }
    
    #[test]
    fn test_exponential_decay_success() {
        // Simple exponential decay: y' = -y
        // Analytical solution: y(t) = exp(-t)
        let fun = Box::new(|_t: f64, y: &[f64]| -> Vec<f64> {
            vec![-y[0]]
        });
        
        let t0 = 0.0;
        let t1 = 5.0;
        let y0 = vec![1.0];
        
        let options = SolveIvpOptions {
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        };
        
        let result = solve_ivp_bdf(fun, t0, t1, y0, Some(options));
        
        match result {
            SolveResult::Success { t, y, .. } => {
                // Check final time
                assert!((t.last().unwrap() - t1).abs() < 1e-10);
                
                // Check analytical solution: y(5) = exp(-5) ≈ 0.006737947
                let y_final = y.last().unwrap()[0];
                let y_analytical = (-t1).exp();
                let error = (y_final - y_analytical).abs();
                
                assert!(error < 1e-7, "Solution error too large: {}", error);
                
                println!("Exponential decay test successful!");
                println!("Analytical solution: {}", y_analytical);
                println!("Numerical solution: {}", y_final);
                println!("Absolute error: {}", error);
            }
            SolveResult::Failed { error, .. } => {
                panic!("Integration failed: {}", error);
            }
            SolveResult::EventTerminated { .. } => {
                panic!("Unexpected event termination");
            }
        }
    }
    
    #[test]
    fn test_event_termination() {
        // Simple linear system: y' = 1, starting from y(0) = 0
        // We'll set an event at y = 2.5
        let fun = Box::new(|_t: f64, _y: &[f64]| -> Vec<f64> {
            vec![1.0]
        });
        
        // Event function: trigger when y reaches 2.5
        let event_fn = Box::new(|_t: f64, y: &[f64]| -> bool {
            y[0] - 2.5 > 0.0 // Trigger when y crosses 2.5 upward
        });
        
        let event = Event {
            function: event_fn,
            terminal: true,
        };
        
        let t0 = 0.0;
        let t1 = 10.0; // We should never reach this due to event
        let y0 = vec![0.0];
        
        let options = SolveIvpOptions {
            rtol: 1e-8,
            atol: 1e-10,
            events: vec![event],
            ..Default::default()
        };
        
        let result = solve_ivp_bdf(fun, t0, t1, y0, Some(options));
        
        match result {
            SolveResult::Success { .. } => {
                panic!("Expected event termination, but got success");
            }
            SolveResult::Failed { error, .. } => {
                panic!("Integration failed: {}", error);
            }
            SolveResult::EventTerminated { t, y, event_index} => {
                assert_eq!(event_index, 0, "Wrong event index");
                
                // Check that the final y value is close to 2.5
                let y_final = y.last().unwrap()[0];
                let t_final = *t.last().unwrap();
                
                // Since y' = 1 and y(0) = 0, we have y(t) = t
                // So when y = 2.5, t should be 2.5
                assert!((y_final - 2.5).abs() < 0.1, "Event y value error: {}", (y_final - 2.5).abs());
                assert!((t_final - 2.5).abs() < 0.1, "Event time error: {}", (t_final - 2.5).abs());
                assert!(t_final < t1, "Event should occur before final time");
                
                println!("Event termination test successful!");
                println!("Event triggered at t = {}", t_final);
                println!("Event triggered at y = {}", y_final);
                println!("Expected values: t ≈ 2.5, y ≈ 2.5");
            }
        }
    }
    
    #[test]
    fn test_pendulum_with_stopping_event() {
        // Simple pendulum: θ'' + sin(θ) = 0
        // Convert to system: θ' = ω, ω' = -sin(θ)
        // We'll stop when the pendulum reaches θ = 0 after starting from θ = π/4
        let fun = Box::new(|_t: f64, y: &[f64]| -> Vec<f64> {
            let theta = y[0];
            let omega = y[1];
            vec![omega, -theta.sin()]
        });
        
        // Event: stop when θ crosses zero going downward
        let event_fn = Box::new(|_t: f64, y: &[f64]| -> bool {
            y[0] < 0.0 // Trigger when θ crosses zero downward
        });
        
        let event = Event {
            function: event_fn,
            terminal: true,
        };
        
        let t0 = 0.0;
        let t1 = 10.0;
        let y0 = vec![std::f64::consts::PI / 4.0, 0.0]; // Start at 45 degrees, no initial velocity
        
        let options = SolveIvpOptions {
            rtol: 1e-8,
            atol: 1e-10,
            events: vec![event],
            dense_output: true,
            ..Default::default()
        };
        
        let result = solve_ivp_bdf(fun, t0, t1, y0, Some(options));
        
        match result {
            SolveResult::Success { .. } => {
                panic!("Expected event termination, but got success");
            }
            SolveResult::Failed { error, .. } => {
                panic!("Integration failed: {}", error);
            }
            SolveResult::EventTerminated { t, y, event_index} => {
                assert_eq!(event_index, 0, "Wrong event index");
                
                let t_final = *t.last().unwrap();
                let y_final = y.last().unwrap();
                let theta_final = y_final[0];
                let omega_final = y_final[1];
                
                // Check that θ is close to 0
                assert!(theta_final.abs() < 0.1, "Theta should be close to 0, got {}", theta_final);
                
                // Check that ω is negative (pendulum swinging downward)
                assert!(omega_final < 0.0, "Omega should be negative, got {}", omega_final);
                
                // Check that the event occurred at a reasonable time
                assert!(t_final > 0.5 && t_final < 5.0, "Event time seems unreasonable: {}", t_final);
                
                println!("Pendulum event test successful!");
                println!("Event triggered at t = {}", t_final);
                println!("Final theta = {}", theta_final);
                println!("Final omega = {}", omega_final);
            }
        }
    }
}