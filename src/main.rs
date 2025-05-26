use std::collections::HashMap;
use ndarray::{Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use plotters::prelude::*;
use std::error::Error;

// Add modules
mod mnist_test;
mod ablation_study;
mod optimizers;
mod optimizer_comparison;
mod mnist_comparison;
mod statistical_analysis;
mod results_generator;

/// Standard Adam optimizer implementation
pub struct StandardAdam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: HashMap<String, Array1<f32>>,
    v: HashMap<String, Array1<f32>>,
    t: usize,
}

impl StandardAdam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
    
    pub fn step(&mut self, params: &mut HashMap<String, Array1<f32>>, grads: &HashMap<String, Array1<f32>>) {
        self.t += 1;
        
        for (name, grad) in grads.iter() {
            let param = params.get_mut(name).unwrap();
            
            // Get or initialize momentum and velocity
            let m = self.m.entry(name.clone()).or_insert_with(|| Array1::zeros(param.len()));
            let v = self.v.entry(name.clone()).or_insert_with(|| Array1::zeros(param.len()));
            
            // Update biased first and second moment estimates
            *m = self.beta1 * &*m + (1.0 - self.beta1) * grad;
            *v = self.beta2 * &*v + (1.0 - self.beta2) * (grad * grad);
            
            // Compute bias-corrected moment estimates
            let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));
            
            // Update parameters
            *param = &*param - &(self.lr * &m_hat / (v_hat.mapv(f32::sqrt) + self.epsilon));
        }
    }
}

/// Enhanced Adam optimizer with four key improvements:
/// 1. Adaptive learning rate schedule
/// 2. Dynamic beta parameters
/// 3. Direct gradient application in early iterations
/// 4. Gradient clipping
pub struct EnhancedAdam {
    base_lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: HashMap<String, Array1<f32>>,
    v: HashMap<String, Array1<f32>>,
    t: usize,
    beta1_product: f32,
    beta2_product: f32,
    clip_threshold: f32,
}

impl EnhancedAdam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            base_lr: lr,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
            beta1_product: 1.0,
            beta2_product: 1.0,
            clip_threshold: 3.0,
        }
    }
    
    fn calculate_adaptive_lr(&self) -> f32 {
        // Enhancement 1: Adaptive learning rate schedule
        if self.t < 50 {
            self.base_lr * 3.0
        } else if self.t < 200 {
            self.base_lr * 2.0
        } else if self.t < 500 {
            self.base_lr * 1.5
        } else {
            self.base_lr
        }
    }
    
    fn calculate_dynamic_betas(&self) -> (f32, f32) {
        // Enhancement 2: Dynamic beta parameters
        if self.t < 100 {
            let adjustment = 0.05 * (1.0 - (self.t as f32 / 100.0));
            (self.beta1 * (1.0 - adjustment), self.beta2 * (1.0 - adjustment))
        } else {
            (self.beta1, self.beta2)
        }
    }
    
    fn clip_gradient(&self, grad: &Array1<f32>) -> Array1<f32> {
        // Enhancement 4: Gradient clipping
        let squared_sum: f32 = grad.iter().map(|x| x * x).sum();
        let norm = squared_sum.sqrt();
        
        if norm <= self.clip_threshold || norm < 1e-8 {
            return grad.clone();
        }
        
        let scale = self.clip_threshold / norm;
        grad * scale
    }
    
    pub fn step(&mut self, params: &mut HashMap<String, Array1<f32>>, grads: &HashMap<String, Array1<f32>>) {
        self.t += 1;
        
        // Calculate adaptive learning rate
        let adaptive_lr = self.calculate_adaptive_lr();
        
        for (name, grad) in grads.iter() {
            let param = params.get_mut(name).unwrap();
            
            // Clip gradient if needed
            let clipped_grad = self.clip_gradient(grad);
            
            // Enhancement 3: Direct gradient application in early iterations
            if self.t < 10 {
                *param = &*param - &(adaptive_lr * 2.0 * &clipped_grad);
                continue;
            }
            
            // Get dynamic beta values
            let (beta1_t, beta2_t) = self.calculate_dynamic_betas();
            
            // Update cumulative beta products for bias correction
            self.beta1_product *= beta1_t;
            self.beta2_product *= beta2_t;
            
            // Get or initialize momentum and velocity
            let m = self.m.entry(name.clone()).or_insert_with(|| Array1::zeros(param.len()));
            let v = self.v.entry(name.clone()).or_insert_with(|| Array1::zeros(param.len()));
            
            // Update biased first and second moment estimates with dynamic betas
            *m = beta1_t * &*m + (1.0 - beta1_t) * &clipped_grad;
            *v = beta2_t * &*v + (1.0 - beta2_t) * (&clipped_grad * &clipped_grad);
            
            // Compute bias-corrected moment estimates using tracked beta products
            let m_hat = &*m / (1.0 - self.beta1_product);
            let v_hat = &*v / (1.0 - self.beta2_product);
            
            // Update parameters with adaptive learning rate
            *param = &*param - &(adaptive_lr * &m_hat / (v_hat.mapv(f32::sqrt) + self.epsilon));
        }
    }
}

/// Simple neural network for demonstration
#[allow(dead_code)]
struct SimpleNN {
    weights: HashMap<String, Array1<f32>>,
}

impl SimpleNN {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut weights = HashMap::new();
        
        // Initialize weights with small random values
        let w1 = Array1::random(input_dim * hidden_dim, Uniform::new(-0.1, 0.1));
        let w2 = Array1::random(hidden_dim * output_dim, Uniform::new(-0.1, 0.1));
        
        weights.insert("w1".to_string(), w1);
        weights.insert("w2".to_string(), w2);
        
        Self { weights }
    }
}

/// Test function with a plateau region
pub fn problem_with_plateau(x: &Array1<f32>) -> f32 {
    let x1 = x[0];
    let x2 = x[1];
    (x1.powi(2) + x2.powi(2)).tanh() + 0.1 * ((x1 - 3.0).powi(2) + (x2 - 3.0).powi(2))
}

/// Gradient of test function
pub fn gradient(x: &Array1<f32>) -> Array1<f32> {
    let x1 = x[0];
    let x2 = x[1];
    
    let z = x1.powi(2) + x2.powi(2);
    let sech_squared = 1.0 / (z.cosh()).powi(2);
    
    let dx1 = 2.0 * x1 * sech_squared + 0.2 * (x1 - 3.0);
    let dx2 = 2.0 * x2 * sech_squared + 0.2 * (x2 - 3.0);
    
    Array1::from_vec(vec![dx1, dx2])
}

/// Rosenbrock function for testing
pub fn rosenbrock(x: &Array1<f32>) -> f32 {
    let mut sum = 0.0;
    for i in 0..(x.len() - 1) {
        sum += 100.0 * (x[i+1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
    }
    sum
}

/// Gradient of Rosenbrock function
pub fn rosenbrock_gradient(x: &Array1<f32>) -> Array1<f32> {
    let n = x.len();
    let mut grad = Array1::zeros(n);
    
    for i in 0..(n-1) {
        grad[i] += -400.0 * x[i] * (x[i+1] - x[i].powi(2)) - 2.0 * (1.0 - x[i]);
    }
    
    for i in 1..n {
        grad[i] += 200.0 * (x[i] - x[i-1].powi(2));
    }
    
    grad
}

/// Run the plateau function test
fn run_plateau_test(learning_rate: f32, max_iter: usize) -> Result<(), Box<dyn Error>> {
    println!("Running plateau function test...");
    
    // Initialize parameters for both optimizers
    let mut standard_params = HashMap::new();
    let mut enhanced_params = HashMap::new();
    
    // Start at the same random point for fair comparison
    let init_point = Array1::random(2, Uniform::new(-0.1, 0.1));
    standard_params.insert("weights".to_string(), init_point.clone());
    enhanced_params.insert("weights".to_string(), init_point.clone());
    
    // Create optimizers
    let mut standard_adam = StandardAdam::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut enhanced_adam = EnhancedAdam::new(learning_rate, 0.9, 0.999, 1e-8);
    
    // Track losses for comparison
    let mut standard_losses = Vec::with_capacity(max_iter);
    let mut enhanced_losses = Vec::with_capacity(max_iter);
    
    // Run optimization
    for i in 0..max_iter {
        // Evaluate standard Adam
        let standard_x = standard_params.get("weights").unwrap();
        let standard_loss = problem_with_plateau(standard_x);
        standard_losses.push(standard_loss);
        
        let standard_grad = gradient(standard_x);
        let mut standard_grads = HashMap::new();
        standard_grads.insert("weights".to_string(), standard_grad);
        
        standard_adam.step(&mut standard_params, &standard_grads);
        
        // Evaluate enhanced Adam
        let enhanced_x = enhanced_params.get("weights").unwrap();
        let enhanced_loss = problem_with_plateau(enhanced_x);
        enhanced_losses.push(enhanced_loss);
        
        let enhanced_grad = gradient(enhanced_x);
        let mut enhanced_grads = HashMap::new();
        enhanced_grads.insert("weights".to_string(), enhanced_grad);
        
        enhanced_adam.step(&mut enhanced_params, &enhanced_grads);
        
        // Print progress periodically
        if i % 100 == 0 || i == max_iter - 1 {
            println!("Iteration {}/{}", i, max_iter);
            println!("  Standard Adam - Loss: {:.6}", standard_loss);
            println!("  Enhanced Adam - Loss: {:.6}", enhanced_loss);
        }
    }
    
    // Print final results
    let standard_final = standard_params.get("weights").unwrap();
    let enhanced_final = enhanced_params.get("weights").unwrap();
    
    println!("\nFinal Results:");
    println!("Standard Adam - Final point: [{:.6}, {:.6}], Loss: {:.6}", 
             standard_final[0], standard_final[1], standard_losses.last().unwrap());
    println!("Enhanced Adam - Final point: [{:.6}, {:.6}], Loss: {:.6}", 
             enhanced_final[0], enhanced_final[1], enhanced_losses.last().unwrap());
    
    // Create a plot to visualize convergence
    let root = BitMapBackend::new("figures/plateau_comparison.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Find the y-axis range with proper padding
    let max_loss = standard_losses.iter()
                                 .chain(enhanced_losses.iter())
                                 .fold(0.0f32, |a, &b| a.max(b));
    
    let min_loss = standard_losses.iter()
                                 .chain(enhanced_losses.iter())
                                 .fold(max_loss, |a, &b| a.min(b));
    
    // Add padding to prevent label overlap
    let y_range = max_loss - min_loss;
    let y_min = min_loss - y_range * 0.05;
    let y_max = max_loss + y_range * 0.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Adam Optimizer Comparison (Plateau Function)", ("sans-serif", 26).into_font())
        .margin(40)
        .x_label_area_size(80)
        .y_label_area_size(120)
        .build_cartesian_2d(0..max_iter, y_min..y_max)?;
    
    chart.configure_mesh()
         .x_desc("Iterations")
         .y_desc("Loss")
         .axis_desc_style(("sans-serif", 26))
         .label_style(("sans-serif", 26))
         .x_labels(6)
         .y_labels(6)
         .disable_mesh()
         .draw()?;
    
    chart.draw_series(LineSeries::new(
        standard_losses.iter().enumerate().map(|(i, &v)| (i, v)),
        RED.stroke_width(3),
    ))?
    .label("Standard Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
    
    chart.draw_series(LineSeries::new(
        enhanced_losses.iter().enumerate().map(|(i, &v)| (i, v)),
        BLUE.stroke_width(3),
    ))?
    .label("Enhanced Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.9))
        .border_style(&BLACK)
        .label_font(("sans-serif", 26))
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    
    println!("Plot saved as 'plateau_comparison.png'");
    
    Ok(())
}

/// Run the Rosenbrock function test
fn run_rosenbrock_test(learning_rate: f32, max_iter: usize) -> Result<(), Box<dyn Error>> {
    println!("Running Rosenbrock function test...");
    
    // Initialize parameters for both optimizers
    let mut standard_params = HashMap::new();
    let mut enhanced_params = HashMap::new();
    
    // Start at the same point for fair comparison
    let init_point = Array1::from_vec(vec![-1.2, 1.0, -0.5, 0.8, -1.0]);
    standard_params.insert("weights".to_string(), init_point.clone());
    enhanced_params.insert("weights".to_string(), init_point.clone());
    
    // Create optimizers
    let mut standard_adam = StandardAdam::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut enhanced_adam = EnhancedAdam::new(learning_rate, 0.9, 0.999, 1e-8);
    
    // Track losses for comparison
    let mut standard_losses = Vec::with_capacity(max_iter);
    let mut enhanced_losses = Vec::with_capacity(max_iter);
    
    // Run optimization
    for i in 0..max_iter {
        // Evaluate standard Adam
        let standard_x = standard_params.get("weights").unwrap();
        let standard_loss = rosenbrock(standard_x);
        standard_losses.push(standard_loss);
        
        let standard_grad = rosenbrock_gradient(standard_x);
        let mut standard_grads = HashMap::new();
        standard_grads.insert("weights".to_string(), standard_grad);
        
        standard_adam.step(&mut standard_params, &standard_grads);
        
        // Evaluate enhanced Adam
        let enhanced_x = enhanced_params.get("weights").unwrap();
        let enhanced_loss = rosenbrock(enhanced_x);
        enhanced_losses.push(enhanced_loss);
        
        let enhanced_grad = rosenbrock_gradient(enhanced_x);
        let mut enhanced_grads = HashMap::new();
        enhanced_grads.insert("weights".to_string(), enhanced_grad);
        
        enhanced_adam.step(&mut enhanced_params, &enhanced_grads);
        
        // Print progress periodically
        if i % 100 == 0 || i == max_iter - 1 {
            println!("Iteration {}/{}", i, max_iter);
            println!("  Standard Adam - Loss: {:.6}", standard_loss);
            println!("  Enhanced Adam - Loss: {:.6}", enhanced_loss);
        }
    }
    
    // Print final results
    println!("\nFinal Results:");
    println!("Standard Adam - Final loss: {:.6}", standard_losses.last().unwrap());
    println!("Enhanced Adam - Final loss: {:.6}", enhanced_losses.last().unwrap());
    
    // Calculate improvement ratio
    let improvement_ratio = standard_losses.last().unwrap() / enhanced_losses.last().unwrap();
    println!("Improvement ratio: {:.2}x", improvement_ratio);
    
    // Create a plot to visualize convergence - use log scale for Rosenbrock
    let root = BitMapBackend::new("figures/rosenbrock_comparison.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Use log scale for y-axis with Rosenbrock function
    let max_loss = standard_losses.iter()
                                 .chain(enhanced_losses.iter())
                                 .fold(0.0f32, |a, &b| a.max(b));
    
    let min_loss = standard_losses.iter()
                                 .chain(enhanced_losses.iter())
                                 .fold(max_loss, |a, &b| a.min(b))
                                 .max(0.01); // Ensure positive for log scale
    
    // Add padding for log scale
    let log_min = min_loss.ln();
    let log_max = max_loss.ln();
    let log_range = log_max - log_min;
    let y_min = log_min - log_range * 0.05;
    let y_max = log_max + log_range * 0.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Adam Optimizer Comparison (Rosenbrock Function)", ("sans-serif", 26).into_font())
        .margin(40)
        .x_label_area_size(80)
        .y_label_area_size(120)
        .build_cartesian_2d(0..max_iter, y_min..y_max)?;
    
    chart.configure_mesh()
         .x_desc("Iterations")
         .y_desc("Log(Loss)")
         .axis_desc_style(("sans-serif", 26))
         .label_style(("sans-serif", 26))
         .x_labels(6)
         .y_labels(6)
         .disable_mesh()
         .draw()?;
    
    chart.draw_series(LineSeries::new(
        standard_losses.iter().enumerate().map(|(i, &v)| (i, v.ln())),
        RED.stroke_width(3),
    ))?
    .label("Standard Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
    
    chart.draw_series(LineSeries::new(
        enhanced_losses.iter().enumerate().map(|(i, &v)| (i, v.ln())),
        BLUE.stroke_width(3),
    ))?
    .label("Enhanced Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.9))
        .border_style(&BLACK)
        .label_font(("sans-serif", 26))
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    
    println!("Plot saved as 'rosenbrock_comparison.png'");
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Enhanced Adam Optimizer Test Suite");
    println!("==================================");
    
    // Run the tests based on command line arguments
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() > 1 {
        match args[1].as_str() {
            "base" => {
                // Run only the basic tests
                run_plateau_test(0.01, 150)?;
                run_rosenbrock_test(0.001, 800)?;
            },
            "mnist" => {
                // Run only MNIST neural network test
                println!("\nRunning MNIST neural network test...");
                mnist_test::run_mnist_test()?;
            },
            "ablation" => {
                // Run only ablation study
                println!("\nRunning ablation study...");
                ablation_study::run_ablation_study()?;
            },
            "comparison" => {
                // Run only optimizer comparison
                println!("\nRunning optimizer comparison on synthetic functions...");
                optimizer_comparison::run_optimizer_comparisons()?;
            },
            "mnist-comparison" => {
                // Run only MNIST optimizer comparison
                println!("\nRunning optimizer comparison on MNIST dataset...");
                mnist_comparison::compare_optimizers_mnist()?;
            },
            "stats" => {
                // Run statistical analysis
                println!("\nRunning statistical analysis...");
                let num_runs = if args.len() > 2 {
                    args[2].parse().unwrap_or(10)
                } else {
                    10 // Default to 10 runs
                };
                statistical_analysis::run_statistical_analyses(num_runs)?;
            },
            "results" => {
                // Generate comprehensive results for paper
                println!("\nGenerating comprehensive results for scientific paper...");
                results_generator::run_comprehensive_tests()?;
            },
            "all" => {
                // Run all tests
                run_plateau_test(0.01, 1000)?;
                run_rosenbrock_test(0.001, 1000)?;
                println!("\nRunning MNIST neural network test...");
                mnist_test::run_mnist_test()?;
                println!("\nRunning ablation study...");
                ablation_study::run_ablation_study()?;
                println!("\nRunning optimizer comparison on synthetic functions...");
                optimizer_comparison::run_optimizer_comparisons()?;
                println!("\nRunning optimizer comparison on MNIST dataset...");
                mnist_comparison::compare_optimizers_mnist()?;
                println!("\nRunning statistical analysis...");
                statistical_analysis::run_statistical_analyses(10)?;
            },
            _ => {
                println!("Unknown test: {}. Available tests:", args[1]);
                println!("  base            - Run basic tests on synthetic functions");
                println!("  mnist           - Run MNIST neural network test");
                println!("  ablation        - Run ablation study");
                println!("  comparison      - Run optimizer comparison on synthetic functions");
                println!("  mnist-comparison - Run optimizer comparison on MNIST dataset");
                println!("  stats [n]       - Run statistical analysis with n runs (default: 10)");
                println!("  results         - Generate comprehensive results for scientific paper");
                println!("  all             - Run all tests");
            }
        }
    } else {
        // By default, run only the optimizer comparison on synthetic functions
        println!("\nRunning optimizer comparison on synthetic functions...");
        optimizer_comparison::run_optimizer_comparisons()?;
    }
    
    println!("\nTests completed successfully!");
    
    Ok(())
} 