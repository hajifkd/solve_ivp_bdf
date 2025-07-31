# solve_ivp_bdf

A simple port of the solve_ivp BDF method from Scipy to Rust.
This library provides a way to solve ordinary differential equations (ODEs) using the Backward Differentiation Formula (BDF) method with simple event handling.

## Features
- Solve ODEs using BDF method
- Event handling to stop integration based on conditions

## Usage
To use this library, add it to your `Cargo.toml`:

```toml
[dependencies]
solve_ivp_bdf = "0.1"
```

Then, you can use it in your Rust project:

```rust
use solve_ivp_bdf::{solve_ivp_bdf, Event};

fn main() {
    let rhs = Box::new(|t: f64, y: &[f64]| -> Vec<f64> {
        vec![-y[0]] // Example: dy/dt = -y
    });

    let event_fn = Box::new(|_t: f64, y: &[f64]| -> bool {
        y[0] - 2.5 < 0.0 // Example: stop when y crosses 2.5
    });
    
    let event = Event {
        function: event_fn,
        terminal: true,
    };
    let t0 = 0.0;
    let t1 = 10.0;
    let y0 = vec![10.0];

    let result = solve_ivp_bdf(rhs, t0, t1, y0, Some(vec![event]));
    println!("{:?}", result);
}
```
