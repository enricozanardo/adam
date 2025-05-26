use std::collections::HashMap;
use ndarray::Array1;
use std::error::Error;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use plotters::prelude::*;
use crate::{StandardAdam, EnhancedAdam, rosenbrock, rosenbrock_gradient, problem_with_plateau, gradient};

/// Enhanced Adam Variant A: Only adaptive learning rate
pub struct VariantA {
    base_lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: HashMap<String, Array1<f32>>,
    v: HashMap<String, Array1<f32>>,
    t: usize,
}

impl VariantA {
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            base_lr: lr,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
    
    fn calculate_adaptive_lr(&self) -> f32 {
        // Only the adaptive learning rate enhancement is active
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
    
    pub fn step(&mut self, params: &mut HashMap<String, Array1<f32>>, grads: &HashMap<String, Array1<f32>>) {
        self.t += 1;
        
        // Calculate adaptive learning rate
        let adaptive_lr = self.calculate_adaptive_lr();
        
        for (name, grad) in grads.iter() {
            let param = params.get_mut(name).unwrap();
            
            // Standard Adam with fixed betas, no gradient clipping, no direct application
            let m = self.m.entry(name.clone()).or_insert_with(|| Array1::zeros(param.len()));
            let v = self.v.entry(name.clone()).or_insert_with(|| Array1::zeros(param.len()));
            
            // Update biased first and second moment estimates with fixed betas
            *m = self.beta1 * &*m + (1.0 - self.beta1) * grad;
            *v = self.beta2 * &*v + (1.0 - self.beta2) * (grad * grad);
            
            // Compute bias-corrected moment estimates
            let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));
            
            // Update parameters with adaptive learning rate
            *param = &*param - &(adaptive_lr * &m_hat / (v_hat.mapv(f32::sqrt) + self.epsilon));
        }
    }
}

/// Enhanced Adam Variant B: Only dynamic beta parameters
pub struct VariantB {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: HashMap<String, Array1<f32>>,
    v: HashMap<String, Array1<f32>>,
    t: usize,
    beta1_product: f32,
    beta2_product: f32,
}

impl VariantB {
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
            beta1_product: 1.0,
            beta2_product: 1.0,
        }
    }
    
    fn calculate_dynamic_betas(&self) -> (f32, f32) {
        // Only the dynamic beta parameters enhancement is active
        if self.t < 100 {
            let adjustment = 0.05 * (1.0 - (self.t as f32 / 100.0));
            (self.beta1 * (1.0 - adjustment), self.beta2 * (1.0 - adjustment))
        } else {
            (self.beta1, self.beta2)
        }
    }
    
    pub fn step(&mut self, params: &mut HashMap<String, Array1<f32>>, grads: &HashMap<String, Array1<f32>>) {
        self.t += 1;
        
        // Get dynamic beta values
        let (beta1_t, beta2_t) = self.calculate_dynamic_betas();
        
        // Update cumulative beta products for bias correction
        self.beta1_product *= beta1_t;
        self.beta2_product *= beta2_t;
        
        for (name, grad) in grads.iter() {
            let param = params.get_mut(name).unwrap();
            
            // Get or initialize momentum and velocity
            let m = self.m.entry(name.clone()).or_insert_with(|| Array1::zeros(param.len()));
            let v = self.v.entry(name.clone()).or_insert_with(|| Array1::zeros(param.len()));
            
            // Update biased first and second moment estimates with dynamic betas
            *m = beta1_t * &*m + (1.0 - beta1_t) * grad;
            *v = beta2_t * &*v + (1.0 - beta2_t) * (grad * grad);
            
            // Compute bias-corrected moment estimates using tracked beta products
            let m_hat = &*m / (1.0 - self.beta1_product);
            let v_hat = &*v / (1.0 - self.beta2_product);
            
            // Update parameters with fixed learning rate
            *param = &*param - &(self.lr * &m_hat / (v_hat.mapv(f32::sqrt) + self.epsilon));
        }
    }
}

/// Enhanced Adam Variant C: Only direct gradient application
pub struct VariantC {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: HashMap<String, Array1<f32>>,
    v: HashMap<String, Array1<f32>>,
    t: usize,
}

impl VariantC {
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
            
            // Early iterations: apply direct gradient update with higher scaling
            // Only the direct gradient application enhancement is active
            if self.t < 10 {
                *param = &*param - &(self.lr * 2.0 * grad);
                continue;
            }
            
            // Standard Adam otherwise
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

/// Enhanced Adam Variant D: Only gradient clipping
pub struct VariantD {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: HashMap<String, Array1<f32>>,
    v: HashMap<String, Array1<f32>>,
    t: usize,
    clip_threshold: f32,
}

impl VariantD {
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
            clip_threshold: 3.0,
        }
    }
    
    fn clip_gradient(&self, grad: &Array1<f32>) -> Array1<f32> {
        // Only the gradient clipping enhancement is active
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
        
        for (name, grad) in grads.iter() {
            let param = params.get_mut(name).unwrap();
            
            // Clip gradient if needed
            let clipped_grad = self.clip_gradient(grad);
            
            // Standard Adam with clipped gradients
            let m = self.m.entry(name.clone()).or_insert_with(|| Array1::zeros(param.len()));
            let v = self.v.entry(name.clone()).or_insert_with(|| Array1::zeros(param.len()));
            
            // Update biased first and second moment estimates
            *m = self.beta1 * &*m + (1.0 - self.beta1) * &clipped_grad;
            *v = self.beta2 * &*v + (1.0 - self.beta2) * (&clipped_grad * &clipped_grad);
            
            // Compute bias-corrected moment estimates
            let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));
            
            // Update parameters
            *param = &*param - &(self.lr * &m_hat / (v_hat.mapv(f32::sqrt) + self.epsilon));
        }
    }
}

/// Run the plateau function test for all variants
pub fn run_plateau_ablation_study(learning_rate: f32, max_iter: usize) -> Result<(), Box<dyn Error>> {
    println!("Running ablation study on plateau function...");
    
    // Initialize parameters for all optimizers
    let mut standard_params = HashMap::new();
    let mut enhanced_params = HashMap::new();
    let mut variant_a_params = HashMap::new();
    let mut variant_b_params = HashMap::new();
    let mut variant_c_params = HashMap::new();
    let mut variant_d_params = HashMap::new();
    
    // Start at the same random point for fair comparison
    let init_point = Array1::random(2, Uniform::new(-0.1, 0.1));
    standard_params.insert("weights".to_string(), init_point.clone());
    enhanced_params.insert("weights".to_string(), init_point.clone());
    variant_a_params.insert("weights".to_string(), init_point.clone());
    variant_b_params.insert("weights".to_string(), init_point.clone());
    variant_c_params.insert("weights".to_string(), init_point.clone());
    variant_d_params.insert("weights".to_string(), init_point.clone());
    
    // Create optimizers
    let mut standard_adam = StandardAdam::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut enhanced_adam = EnhancedAdam::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut variant_a = VariantA::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut variant_b = VariantB::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut variant_c = VariantC::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut variant_d = VariantD::new(learning_rate, 0.9, 0.999, 1e-8);
    
    // Track losses for comparison
    let mut standard_losses = Vec::with_capacity(max_iter);
    let mut enhanced_losses = Vec::with_capacity(max_iter);
    let mut variant_a_losses = Vec::with_capacity(max_iter);
    let mut variant_b_losses = Vec::with_capacity(max_iter);
    let mut variant_c_losses = Vec::with_capacity(max_iter);
    let mut variant_d_losses = Vec::with_capacity(max_iter);
    
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
        
        // Evaluate Variant A
        let variant_a_x = variant_a_params.get("weights").unwrap();
        let variant_a_loss = problem_with_plateau(variant_a_x);
        variant_a_losses.push(variant_a_loss);
        
        let variant_a_grad = gradient(variant_a_x);
        let mut variant_a_grads = HashMap::new();
        variant_a_grads.insert("weights".to_string(), variant_a_grad);
        
        variant_a.step(&mut variant_a_params, &variant_a_grads);
        
        // Evaluate Variant B
        let variant_b_x = variant_b_params.get("weights").unwrap();
        let variant_b_loss = problem_with_plateau(variant_b_x);
        variant_b_losses.push(variant_b_loss);
        
        let variant_b_grad = gradient(variant_b_x);
        let mut variant_b_grads = HashMap::new();
        variant_b_grads.insert("weights".to_string(), variant_b_grad);
        
        variant_b.step(&mut variant_b_params, &variant_b_grads);
        
        // Evaluate Variant C
        let variant_c_x = variant_c_params.get("weights").unwrap();
        let variant_c_loss = problem_with_plateau(variant_c_x);
        variant_c_losses.push(variant_c_loss);
        
        let variant_c_grad = gradient(variant_c_x);
        let mut variant_c_grads = HashMap::new();
        variant_c_grads.insert("weights".to_string(), variant_c_grad);
        
        variant_c.step(&mut variant_c_params, &variant_c_grads);
        
        // Evaluate Variant D
        let variant_d_x = variant_d_params.get("weights").unwrap();
        let variant_d_loss = problem_with_plateau(variant_d_x);
        variant_d_losses.push(variant_d_loss);
        
        let variant_d_grad = gradient(variant_d_x);
        let mut variant_d_grads = HashMap::new();
        variant_d_grads.insert("weights".to_string(), variant_d_grad);
        
        variant_d.step(&mut variant_d_params, &variant_d_grads);
        
        // Print progress periodically
        if i % 100 == 0 || i == max_iter - 1 {
            println!("Iteration {}/{}", i, max_iter);
        }
    }
    
    // Print final results
    println!("\nFinal Results (Plateau Function):");
    println!("Standard Adam - Final loss: {:.6}", standard_losses.last().unwrap());
    println!("Enhanced Adam - Final loss: {:.6}", enhanced_losses.last().unwrap());
    println!("Variant A (Adaptive LR) - Final loss: {:.6}", variant_a_losses.last().unwrap());
    println!("Variant B (Dynamic Betas) - Final loss: {:.6}", variant_b_losses.last().unwrap());
    println!("Variant C (Direct Gradient) - Final loss: {:.6}", variant_c_losses.last().unwrap());
    println!("Variant D (Gradient Clipping) - Final loss: {:.6}", variant_d_losses.last().unwrap());
    
    // Calculate improvement ratios
    let standard_loss = standard_losses.last().unwrap();
    let enhanced_loss = enhanced_losses.last().unwrap();
    let variant_a_loss = variant_a_losses.last().unwrap();
    let variant_b_loss = variant_b_losses.last().unwrap();
    let variant_c_loss = variant_c_losses.last().unwrap();
    let variant_d_loss = variant_d_losses.last().unwrap();
    
    println!("\nImprovement Ratios (Plateau Function):");
    println!("Enhanced Adam: {:.2}x", standard_loss / enhanced_loss);
    println!("Variant A (Adaptive LR): {:.2}x", standard_loss / variant_a_loss);
    println!("Variant B (Dynamic Betas): {:.2}x", standard_loss / variant_b_loss);
    println!("Variant C (Direct Gradient): {:.2}x", standard_loss / variant_c_loss);
    println!("Variant D (Gradient Clipping): {:.2}x", standard_loss / variant_d_loss);
    
    // Create a plot to visualize convergence
    let root = BitMapBackend::new("figures/plateau_ablation_study.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Find the y-axis range with proper padding
    let max_loss = standard_losses.iter()
                                 .chain(enhanced_losses.iter())
                                 .chain(variant_a_losses.iter())
                                 .chain(variant_b_losses.iter())
                                 .chain(variant_c_losses.iter())
                                 .chain(variant_d_losses.iter())
                                 .fold(0.0f32, |a, &b| a.max(b));
    
    let min_loss = standard_losses.iter()
                                 .chain(enhanced_losses.iter())
                                 .chain(variant_a_losses.iter())
                                 .chain(variant_b_losses.iter())
                                 .chain(variant_c_losses.iter())
                                 .chain(variant_d_losses.iter())
                                 .fold(max_loss, |a, &b| a.min(b));
    
    // Add padding to prevent label overlap
    let y_range = max_loss - min_loss;
    let y_min = min_loss - y_range * 0.05;
    let y_max = max_loss + y_range * 0.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Adam Optimizer Ablation Study (Plateau Function)", ("sans-serif", 26).into_font())
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
    
    // Plot standard Adam (bold)
    chart.draw_series(LineSeries::new(
        standard_losses.iter().enumerate().map(|(i, &v)| (i, v)),
        RED.stroke_width(3),
    ))?
    .label("Standard Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
    
    // Plot enhanced Adam (bold)
    chart.draw_series(LineSeries::new(
        enhanced_losses.iter().enumerate().map(|(i, &v)| (i, v)),
        BLUE.stroke_width(3),
    ))?
    .label("Enhanced Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));
    
    // Plot Variant A (normal thickness)
    chart.draw_series(LineSeries::new(
        variant_a_losses.iter().enumerate().map(|(i, &v)| (i, v)),
        &GREEN,
    ))?
    .label("Variant A (Adaptive LR)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
    
    // Plot Variant B (normal thickness)
    chart.draw_series(LineSeries::new(
        variant_b_losses.iter().enumerate().map(|(i, &v)| (i, v)),
        &MAGENTA,
    ))?
    .label("Variant B (Dynamic Betas)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));
    
    // Plot Variant C (normal thickness)
    chart.draw_series(LineSeries::new(
        variant_c_losses.iter().enumerate().map(|(i, &v)| (i, v)),
        &CYAN,
    ))?
    .label("Variant C (Direct Gradient)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &CYAN));
    
    // Plot Variant D (normal thickness)
    chart.draw_series(LineSeries::new(
        variant_d_losses.iter().enumerate().map(|(i, &v)| (i, v)),
        &BLACK,
    ))?
    .label("Variant D (Gradient Clipping)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.9))
        .border_style(&BLACK)
        .label_font(("sans-serif", 26))
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    
    println!("Plateau ablation study plot saved as 'plateau_ablation_study.png'");
    
    Ok(())
}

/// Run the Rosenbrock function test for all variants
pub fn run_rosenbrock_ablation_study(learning_rate: f32, max_iter: usize) -> Result<(), Box<dyn Error>> {
    println!("Running ablation study on Rosenbrock function...");
    
    // Initialize parameters for all optimizers
    let mut standard_params = HashMap::new();
    let mut enhanced_params = HashMap::new();
    let mut variant_a_params = HashMap::new();
    let mut variant_b_params = HashMap::new();
    let mut variant_c_params = HashMap::new();
    let mut variant_d_params = HashMap::new();
    
    // Start at the same point for fair comparison
    // 5-dimensional Rosenbrock function
    let init_point = Array1::from_vec(vec![-1.2, 1.0, -0.5, 0.8, -1.0]);
    standard_params.insert("weights".to_string(), init_point.clone());
    enhanced_params.insert("weights".to_string(), init_point.clone());
    variant_a_params.insert("weights".to_string(), init_point.clone());
    variant_b_params.insert("weights".to_string(), init_point.clone());
    variant_c_params.insert("weights".to_string(), init_point.clone());
    variant_d_params.insert("weights".to_string(), init_point.clone());
    
    // Create optimizers
    let mut standard_adam = StandardAdam::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut enhanced_adam = EnhancedAdam::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut variant_a = VariantA::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut variant_b = VariantB::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut variant_c = VariantC::new(learning_rate, 0.9, 0.999, 1e-8);
    let mut variant_d = VariantD::new(learning_rate, 0.9, 0.999, 1e-8);
    
    // Track losses for comparison
    let mut standard_losses = Vec::with_capacity(max_iter);
    let mut enhanced_losses = Vec::with_capacity(max_iter);
    let mut variant_a_losses = Vec::with_capacity(max_iter);
    let mut variant_b_losses = Vec::with_capacity(max_iter);
    let mut variant_c_losses = Vec::with_capacity(max_iter);
    let mut variant_d_losses = Vec::with_capacity(max_iter);
    
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
        
        // Evaluate Variant A
        let variant_a_x = variant_a_params.get("weights").unwrap();
        let variant_a_loss = rosenbrock(variant_a_x);
        variant_a_losses.push(variant_a_loss);
        
        let variant_a_grad = rosenbrock_gradient(variant_a_x);
        let mut variant_a_grads = HashMap::new();
        variant_a_grads.insert("weights".to_string(), variant_a_grad);
        
        variant_a.step(&mut variant_a_params, &variant_a_grads);
        
        // Evaluate Variant B
        let variant_b_x = variant_b_params.get("weights").unwrap();
        let variant_b_loss = rosenbrock(variant_b_x);
        variant_b_losses.push(variant_b_loss);
        
        let variant_b_grad = rosenbrock_gradient(variant_b_x);
        let mut variant_b_grads = HashMap::new();
        variant_b_grads.insert("weights".to_string(), variant_b_grad);
        
        variant_b.step(&mut variant_b_params, &variant_b_grads);
        
        // Evaluate Variant C
        let variant_c_x = variant_c_params.get("weights").unwrap();
        let variant_c_loss = rosenbrock(variant_c_x);
        variant_c_losses.push(variant_c_loss);
        
        let variant_c_grad = rosenbrock_gradient(variant_c_x);
        let mut variant_c_grads = HashMap::new();
        variant_c_grads.insert("weights".to_string(), variant_c_grad);
        
        variant_c.step(&mut variant_c_params, &variant_c_grads);
        
        // Evaluate Variant D
        let variant_d_x = variant_d_params.get("weights").unwrap();
        let variant_d_loss = rosenbrock(variant_d_x);
        variant_d_losses.push(variant_d_loss);
        
        let variant_d_grad = rosenbrock_gradient(variant_d_x);
        let mut variant_d_grads = HashMap::new();
        variant_d_grads.insert("weights".to_string(), variant_d_grad);
        
        variant_d.step(&mut variant_d_params, &variant_d_grads);
        
        // Print progress periodically
        if i % 100 == 0 || i == max_iter - 1 {
            println!("Iteration {}/{}", i, max_iter);
        }
    }
    
    // Print final results
    println!("\nFinal Results (Rosenbrock Function):");
    println!("Standard Adam - Final loss: {:.6}", standard_losses.last().unwrap());
    println!("Enhanced Adam - Final loss: {:.6}", enhanced_losses.last().unwrap());
    println!("Variant A (Adaptive LR) - Final loss: {:.6}", variant_a_losses.last().unwrap());
    println!("Variant B (Dynamic Betas) - Final loss: {:.6}", variant_b_losses.last().unwrap());
    println!("Variant C (Direct Gradient) - Final loss: {:.6}", variant_c_losses.last().unwrap());
    println!("Variant D (Gradient Clipping) - Final loss: {:.6}", variant_d_losses.last().unwrap());
    
    // Calculate improvement ratios
    let standard_loss = standard_losses.last().unwrap();
    let enhanced_loss = enhanced_losses.last().unwrap();
    let variant_a_loss = variant_a_losses.last().unwrap();
    let variant_b_loss = variant_b_losses.last().unwrap();
    let variant_c_loss = variant_c_losses.last().unwrap();
    let variant_d_loss = variant_d_losses.last().unwrap();
    
    println!("\nImprovement Ratios (Rosenbrock Function):");
    println!("Enhanced Adam: {:.2}x", standard_loss / enhanced_loss);
    println!("Variant A (Adaptive LR): {:.2}x", standard_loss / variant_a_loss);
    println!("Variant B (Dynamic Betas): {:.2}x", standard_loss / variant_b_loss);
    println!("Variant C (Direct Gradient): {:.2}x", standard_loss / variant_c_loss);
    println!("Variant D (Gradient Clipping): {:.2}x", standard_loss / variant_d_loss);
    
    // Create a plot to visualize convergence - use log scale for Rosenbrock
    let root = BitMapBackend::new("figures/rosenbrock_ablation_study.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Filter out NaN and infinite values for plotting
    let filter_valid = |losses: &[f32]| -> Vec<f32> {
        losses.iter().filter(|&&x| x.is_finite() && x > 0.0).cloned().collect()
    };
    
    let valid_standard = filter_valid(&standard_losses);
    let valid_enhanced = filter_valid(&enhanced_losses);
    let valid_variant_a = filter_valid(&variant_a_losses);
    let valid_variant_b = filter_valid(&variant_b_losses);
    let valid_variant_c = filter_valid(&variant_c_losses);
    let valid_variant_d = filter_valid(&variant_d_losses);
    
    // Find the y-axis range - use log scale for better visualization
    let max_loss = valid_standard.iter()
                                .chain(valid_enhanced.iter())
                                .chain(valid_variant_a.iter())
                                .chain(valid_variant_b.iter())
                                .chain(valid_variant_c.iter())
                                .chain(valid_variant_d.iter())
                                .fold(0.0f32, |a, &b| a.max(b));
    
    let min_loss = valid_standard.iter()
                                .chain(valid_enhanced.iter())
                                .chain(valid_variant_a.iter())
                                .chain(valid_variant_b.iter())
                                .chain(valid_variant_c.iter())
                                .chain(valid_variant_d.iter())
                                .fold(max_loss, |a, &b| a.min(b)).max(0.01); // Ensure positive for log scale
    
    // Add padding for log scale
    let log_min = min_loss.ln();
    let log_max = max_loss.ln();
    let log_range = log_max - log_min;
    let y_min = log_min - log_range * 0.05;
    let y_max = log_max + log_range * 0.1;
    
    // Use log scale for rosenbrock visualization
    let mut chart = ChartBuilder::on(&root)
        .caption("Adam Optimizer Ablation Study (Rosenbrock Function)", ("sans-serif", 26).into_font())
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
    
    // Plot standard Adam (bold)
    chart.draw_series(LineSeries::new(
        standard_losses.iter().enumerate().filter_map(|(i, &v)| {
            if v.is_finite() && v > 0.0 { Some((i, v.ln())) } else { None }
        }),
        RED.stroke_width(3),
    ))?
    .label("Standard Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
    
    // Plot enhanced Adam (bold)
    chart.draw_series(LineSeries::new(
        enhanced_losses.iter().enumerate().filter_map(|(i, &v)| {
            if v.is_finite() && v > 0.0 { Some((i, v.ln())) } else { None }
        }),
        BLUE.stroke_width(3),
    ))?
    .label("Enhanced Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));
    
    // Plot Variant A
    chart.draw_series(LineSeries::new(
        variant_a_losses.iter().enumerate().filter_map(|(i, &v)| {
            if v.is_finite() && v > 0.0 { Some((i, v.ln())) } else { None }
        }),
        &GREEN,
    ))?
    .label("Variant A (Adaptive LR)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
    
    // Plot Variant B
    chart.draw_series(LineSeries::new(
        variant_b_losses.iter().enumerate().filter_map(|(i, &v)| {
            if v.is_finite() && v > 0.0 { Some((i, v.ln())) } else { None }
        }),
        &MAGENTA,
    ))?
    .label("Variant B (Dynamic Betas)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));
    
    // Plot Variant C (only if there are valid values)
    if !valid_variant_c.is_empty() {
        chart.draw_series(LineSeries::new(
            variant_c_losses.iter().enumerate().filter_map(|(i, &v)| {
                if v.is_finite() && v > 0.0 { Some((i, v.ln())) } else { None }
            }),
            &CYAN,
        ))?
        .label("Variant C (Direct Gradient)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &CYAN));
    }
    
    // Plot Variant D
    chart.draw_series(LineSeries::new(
        variant_d_losses.iter().enumerate().filter_map(|(i, &v)| {
            if v.is_finite() && v > 0.0 { Some((i, v.ln())) } else { None }
        }),
        &BLACK,
    ))?
    .label("Variant D (Gradient Clipping)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.9))
        .border_style(&BLACK)
        .label_font(("sans-serif", 26))
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    
    println!("Rosenbrock ablation study plot saved as 'rosenbrock_ablation_study.png'");
    
    Ok(())
}

/// Run the ablation study on both synthetic functions
pub fn run_ablation_study() -> Result<(), Box<dyn Error>> {
    println!("=== Ablation Study ===");
    println!("Testing each enhancement individually to measure its contribution");
    
    // Set up test parameters
    let learning_rate = 0.01;
    let max_iter = 150;
    let max_iter_ablation = 120;
    
    // Run ablation study on plateau function
    println!("\n=== Plateau Function ===");
    run_plateau_ablation_study(learning_rate, max_iter_ablation)?;
    
    // Run ablation study on Rosenbrock function
    println!("\n=== Rosenbrock Function ===");
    run_rosenbrock_ablation_study(learning_rate, max_iter)?;
    
    println!("\nAblation study completed. Check the output directory for plots.");
    Ok(())
}