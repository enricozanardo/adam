use std::collections::HashMap;
use std::error::Error;
use ndarray::Array1;
use plotters::prelude::*;

use crate::problem_with_plateau;
use crate::gradient;
use crate::rosenbrock;
use crate::rosenbrock_gradient;
use crate::{StandardAdam, EnhancedAdam};
use crate::optimizers::{SGDMomentum, RMSProp, AdaGrad};

/// Compare optimizers on the plateau function
pub fn compare_optimizers_plateau(max_iter: usize) -> Result<(), Box<dyn Error>> {
    println!("Running optimizer comparison on plateau function...");
    
    // Initialize parameters for all optimizers
    let mut standard_adam_params = HashMap::new();
    let mut enhanced_adam_params = HashMap::new();
    let mut sgd_momentum_params = HashMap::new();
    let mut rmsprop_params = HashMap::new();
    let mut adagrad_params = HashMap::new();
    
    // Start at the same point for fair comparison
    let init_point = Array1::from_vec(vec![0.1, 0.1]);
    standard_adam_params.insert("weights".to_string(), init_point.clone());
    enhanced_adam_params.insert("weights".to_string(), init_point.clone());
    sgd_momentum_params.insert("weights".to_string(), init_point.clone());
    rmsprop_params.insert("weights".to_string(), init_point.clone());
    adagrad_params.insert("weights".to_string(), init_point.clone());
    
    // Create optimizers
    let mut standard_adam = StandardAdam::new(0.01, 0.9, 0.999, 1e-8);
    let mut enhanced_adam = EnhancedAdam::new(0.01, 0.9, 0.999, 1e-8);
    let mut sgd_momentum = SGDMomentum::new(0.01, 0.9);
    let mut rmsprop = RMSProp::new(0.01, 0.9, 1e-8);
    let mut adagrad = AdaGrad::new(0.01, 1e-8);
    
    // Track losses for comparison
    let mut standard_adam_losses = Vec::with_capacity(max_iter);
    let mut enhanced_adam_losses = Vec::with_capacity(max_iter);
    let mut sgd_momentum_losses = Vec::with_capacity(max_iter);
    let mut rmsprop_losses = Vec::with_capacity(max_iter);
    let mut adagrad_losses = Vec::with_capacity(max_iter);
    
    // Run optimization
    for i in 0..max_iter {
        // Evaluate standard Adam
        let standard_x = standard_adam_params.get("weights").unwrap();
        let standard_loss = problem_with_plateau(standard_x);
        standard_adam_losses.push(standard_loss);
        
        let standard_grad = gradient(standard_x);
        let mut standard_grads = HashMap::new();
        standard_grads.insert("weights".to_string(), standard_grad);
        
        standard_adam.step(&mut standard_adam_params, &standard_grads);
        
        // Evaluate enhanced Adam
        let enhanced_x = enhanced_adam_params.get("weights").unwrap();
        let enhanced_loss = problem_with_plateau(enhanced_x);
        enhanced_adam_losses.push(enhanced_loss);
        
        let enhanced_grad = gradient(enhanced_x);
        let mut enhanced_grads = HashMap::new();
        enhanced_grads.insert("weights".to_string(), enhanced_grad);
        
        enhanced_adam.step(&mut enhanced_adam_params, &enhanced_grads);
        
        // Evaluate SGD with momentum
        let sgd_x = sgd_momentum_params.get("weights").unwrap();
        let sgd_loss = problem_with_plateau(sgd_x);
        sgd_momentum_losses.push(sgd_loss);
        
        let sgd_grad = gradient(sgd_x);
        let mut sgd_grads = HashMap::new();
        sgd_grads.insert("weights".to_string(), sgd_grad);
        
        sgd_momentum.step(&mut sgd_momentum_params, &sgd_grads);
        
        // Evaluate RMSProp
        let rmsprop_x = rmsprop_params.get("weights").unwrap();
        let rmsprop_loss = problem_with_plateau(rmsprop_x);
        rmsprop_losses.push(rmsprop_loss);
        
        let rmsprop_grad = gradient(rmsprop_x);
        let mut rmsprop_grads = HashMap::new();
        rmsprop_grads.insert("weights".to_string(), rmsprop_grad);
        
        rmsprop.step(&mut rmsprop_params, &rmsprop_grads);
        
        // Evaluate AdaGrad
        let adagrad_x = adagrad_params.get("weights").unwrap();
        let adagrad_loss = problem_with_plateau(adagrad_x);
        adagrad_losses.push(adagrad_loss);
        
        let adagrad_grad = gradient(adagrad_x);
        let mut adagrad_grads = HashMap::new();
        adagrad_grads.insert("weights".to_string(), adagrad_grad);
        
        adagrad.step(&mut adagrad_params, &adagrad_grads);
        
        // Print progress periodically
        if i % 100 == 0 || i == max_iter - 1 {
            println!("Iteration {}/{}", i, max_iter);
            println!("  Standard Adam - Loss: {:.6}", standard_loss);
            println!("  Enhanced Adam - Loss: {:.6}", enhanced_loss);
            println!("  SGD Momentum  - Loss: {:.6}", sgd_loss);
            println!("  RMSProp       - Loss: {:.6}", rmsprop_loss);
            println!("  AdaGrad       - Loss: {:.6}", adagrad_loss);
        }
    }
    
    // Print final results
    println!("\nFinal Results (Plateau Function):");
    println!("Standard Adam - Final loss: {:.6}", standard_adam_losses.last().unwrap());
    println!("Enhanced Adam - Final loss: {:.6}", enhanced_adam_losses.last().unwrap());
    println!("SGD Momentum  - Final loss: {:.6}", sgd_momentum_losses.last().unwrap());
    println!("RMSProp       - Final loss: {:.6}", rmsprop_losses.last().unwrap());
    println!("AdaGrad       - Final loss: {:.6}", adagrad_losses.last().unwrap());
    
    // Calculate improvement ratios relative to Standard Adam
    let standard_final = *standard_adam_losses.last().unwrap();
    let enhanced_ratio = standard_final / enhanced_adam_losses.last().unwrap();
    let sgd_ratio = standard_final / sgd_momentum_losses.last().unwrap();
    let rmsprop_ratio = standard_final / rmsprop_losses.last().unwrap();
    let adagrad_ratio = standard_final / adagrad_losses.last().unwrap();
    
    println!("\nImprovement Ratios (relative to Standard Adam):");
    println!("Enhanced Adam: {:.2}x", enhanced_ratio);
    println!("SGD Momentum:  {:.2}x", sgd_ratio);
    println!("RMSProp:       {:.2}x", rmsprop_ratio);
    println!("AdaGrad:       {:.2}x", adagrad_ratio);
    
    // Create a plot to visualize convergence
    let root = BitMapBackend::new("figures/plateau_optimizer_comparison.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Find the y-axis range with proper padding
    let all_losses = standard_adam_losses.iter()
        .chain(enhanced_adam_losses.iter())
        .chain(sgd_momentum_losses.iter())
        .chain(rmsprop_losses.iter())
        .chain(adagrad_losses.iter());
        
    let max_loss = all_losses.clone().fold(0.0f32, |a, &b| a.max(b));
    let min_loss = all_losses.fold(max_loss, |a, &b| a.min(b));
    
    // Add padding to prevent label overlap
    let y_range = max_loss - min_loss;
    let y_min = min_loss - y_range * 0.05;
    let y_max = max_loss + y_range * 0.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Optimizer Comparison (Plateau Function)", ("sans-serif", 26).into_font())
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
        standard_adam_losses.iter().enumerate().map(|(i, &v)| (i, v)),
        RED.stroke_width(3),
    ))?
    .label("Standard Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
    
    chart.draw_series(LineSeries::new(
        enhanced_adam_losses.iter().enumerate().map(|(i, &v)| (i, v)),
        BLUE.stroke_width(3),
    ))?
    .label("Enhanced Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));
    
    chart.draw_series(LineSeries::new(
        sgd_momentum_losses.iter().enumerate().map(|(i, &v)| (i, v)),
        &GREEN,
    ))?
    .label("SGD with Momentum")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
    
    chart.draw_series(LineSeries::new(
        rmsprop_losses.iter().enumerate().map(|(i, &v)| (i, v)),
        &MAGENTA,
    ))?
    .label("RMSProp")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));
    
    chart.draw_series(LineSeries::new(
        adagrad_losses.iter().enumerate().map(|(i, &v)| (i, v)),
        &CYAN,
    ))?
    .label("AdaGrad")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &CYAN));
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.9))
        .border_style(&BLACK)
        .label_font(("sans-serif", 26))
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    
    println!("Plot saved as 'plateau_optimizer_comparison.png'");
    
    Ok(())
}

/// Compare optimizers on the Rosenbrock function
pub fn compare_optimizers_rosenbrock(max_iter: usize) -> Result<(), Box<dyn Error>> {
    println!("Running optimizer comparison on Rosenbrock function...");
    
    // Initialize parameters for all optimizers
    let mut standard_adam_params = HashMap::new();
    let mut enhanced_adam_params = HashMap::new();
    let mut sgd_momentum_params = HashMap::new();
    let mut rmsprop_params = HashMap::new();
    let mut adagrad_params = HashMap::new();
    
    // Start at the same point for fair comparison
    let init_point = Array1::from_vec(vec![-1.2, 1.0, -0.5, 0.8, -1.0]);
    standard_adam_params.insert("weights".to_string(), init_point.clone());
    enhanced_adam_params.insert("weights".to_string(), init_point.clone());
    sgd_momentum_params.insert("weights".to_string(), init_point.clone());
    rmsprop_params.insert("weights".to_string(), init_point.clone());
    adagrad_params.insert("weights".to_string(), init_point.clone());
    
    // Create optimizers
    let mut standard_adam = StandardAdam::new(0.001, 0.9, 0.999, 1e-8);
    let mut enhanced_adam = EnhancedAdam::new(0.001, 0.9, 0.999, 1e-8);
    let mut sgd_momentum = SGDMomentum::new(0.001, 0.9);
    let mut rmsprop = RMSProp::new(0.001, 0.9, 1e-8);
    let mut adagrad = AdaGrad::new(0.001, 1e-8);
    
    // Track losses for comparison
    let mut standard_adam_losses = Vec::with_capacity(max_iter);
    let mut enhanced_adam_losses = Vec::with_capacity(max_iter);
    let mut sgd_momentum_losses = Vec::with_capacity(max_iter);
    let mut rmsprop_losses = Vec::with_capacity(max_iter);
    let mut adagrad_losses = Vec::with_capacity(max_iter);
    
    // Run optimization
    for i in 0..max_iter {
        // Evaluate standard Adam
        let standard_x = standard_adam_params.get("weights").unwrap();
        let standard_loss = rosenbrock(standard_x);
        standard_adam_losses.push(standard_loss);
        
        let standard_grad = rosenbrock_gradient(standard_x);
        let mut standard_grads = HashMap::new();
        standard_grads.insert("weights".to_string(), standard_grad);
        
        standard_adam.step(&mut standard_adam_params, &standard_grads);
        
        // Evaluate enhanced Adam
        let enhanced_x = enhanced_adam_params.get("weights").unwrap();
        let enhanced_loss = rosenbrock(enhanced_x);
        enhanced_adam_losses.push(enhanced_loss);
        
        let enhanced_grad = rosenbrock_gradient(enhanced_x);
        let mut enhanced_grads = HashMap::new();
        enhanced_grads.insert("weights".to_string(), enhanced_grad);
        
        enhanced_adam.step(&mut enhanced_adam_params, &enhanced_grads);
        
        // Evaluate SGD with momentum
        let sgd_x = sgd_momentum_params.get("weights").unwrap();
        let sgd_loss = rosenbrock(sgd_x);
        sgd_momentum_losses.push(sgd_loss);
        
        let sgd_grad = rosenbrock_gradient(sgd_x);
        let mut sgd_grads = HashMap::new();
        sgd_grads.insert("weights".to_string(), sgd_grad);
        
        sgd_momentum.step(&mut sgd_momentum_params, &sgd_grads);
        
        // Evaluate RMSProp
        let rmsprop_x = rmsprop_params.get("weights").unwrap();
        let rmsprop_loss = rosenbrock(rmsprop_x);
        rmsprop_losses.push(rmsprop_loss);
        
        let rmsprop_grad = rosenbrock_gradient(rmsprop_x);
        let mut rmsprop_grads = HashMap::new();
        rmsprop_grads.insert("weights".to_string(), rmsprop_grad);
        
        rmsprop.step(&mut rmsprop_params, &rmsprop_grads);
        
        // Evaluate AdaGrad
        let adagrad_x = adagrad_params.get("weights").unwrap();
        let adagrad_loss = rosenbrock(adagrad_x);
        adagrad_losses.push(adagrad_loss);
        
        let adagrad_grad = rosenbrock_gradient(adagrad_x);
        let mut adagrad_grads = HashMap::new();
        adagrad_grads.insert("weights".to_string(), adagrad_grad);
        
        adagrad.step(&mut adagrad_params, &adagrad_grads);
        
        // Print progress periodically
        if i % 100 == 0 || i == max_iter - 1 {
            println!("Iteration {}/{}", i, max_iter);
            println!("  Standard Adam - Loss: {:.6}", standard_loss);
            println!("  Enhanced Adam - Loss: {:.6}", enhanced_loss);
            println!("  SGD Momentum  - Loss: {:.6}", sgd_loss);
            println!("  RMSProp       - Loss: {:.6}", rmsprop_loss);
            println!("  AdaGrad       - Loss: {:.6}", adagrad_loss);
        }
    }
    
    // Print final results
    println!("\nFinal Results (Rosenbrock Function):");
    println!("Standard Adam - Final loss: {:.6}", standard_adam_losses.last().unwrap());
    println!("Enhanced Adam - Final loss: {:.6}", enhanced_adam_losses.last().unwrap());
    println!("SGD Momentum  - Final loss: {:.6}", sgd_momentum_losses.last().unwrap());
    println!("RMSProp       - Final loss: {:.6}", rmsprop_losses.last().unwrap());
    println!("AdaGrad       - Final loss: {:.6}", adagrad_losses.last().unwrap());
    
    // Calculate improvement ratios relative to Standard Adam
    let standard_final = *standard_adam_losses.last().unwrap();
    let enhanced_ratio = standard_final / enhanced_adam_losses.last().unwrap();
    let sgd_ratio = standard_final / sgd_momentum_losses.last().unwrap();
    let rmsprop_ratio = standard_final / rmsprop_losses.last().unwrap();
    let adagrad_ratio = standard_final / adagrad_losses.last().unwrap();
    
    println!("\nImprovement Ratios (relative to Standard Adam):");
    println!("Enhanced Adam: {:.2}x", enhanced_ratio);
    println!("SGD Momentum:  {:.2}x", sgd_ratio);
    println!("RMSProp:       {:.2}x", rmsprop_ratio);
    println!("AdaGrad:       {:.2}x", adagrad_ratio);
    
    // Create a plot to visualize convergence - use log scale for Rosenbrock
    let root = BitMapBackend::new("figures/rosenbrock_optimizer_comparison.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Find the y-axis range for log scale with proper padding
    let all_losses = standard_adam_losses.iter()
        .chain(enhanced_adam_losses.iter())
        .chain(sgd_momentum_losses.iter())
        .chain(rmsprop_losses.iter())
        .chain(adagrad_losses.iter());
        
    let max_loss = all_losses.clone().fold(0.0f32, |a, &b| a.max(b));
    let min_loss = all_losses.fold(max_loss, |a, &b| a.min(b)).max(0.01); // Ensure positive for log scale
    
    // Add padding for log scale
    let log_min = min_loss.ln();
    let log_max = max_loss.ln();
    let log_range = log_max - log_min;
    let y_min = log_min - log_range * 0.05;
    let y_max = log_max + log_range * 0.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Optimizer Comparison (Rosenbrock Function)", ("sans-serif", 26).into_font())
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
        standard_adam_losses.iter().enumerate().map(|(i, &v)| (i, v.ln())),
        RED.stroke_width(3),
    ))?
    .label("Standard Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
    
    chart.draw_series(LineSeries::new(
        enhanced_adam_losses.iter().enumerate().map(|(i, &v)| (i, v.ln())),
        BLUE.stroke_width(3),
    ))?
    .label("Enhanced Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));
    
    chart.draw_series(LineSeries::new(
        sgd_momentum_losses.iter().enumerate().map(|(i, &v)| (i, v.ln())),
        &GREEN,
    ))?
    .label("SGD with Momentum")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
    
    chart.draw_series(LineSeries::new(
        rmsprop_losses.iter().enumerate().map(|(i, &v)| (i, v.ln())),
        &MAGENTA,
    ))?
    .label("RMSProp")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));
    
    chart.draw_series(LineSeries::new(
        adagrad_losses.iter().enumerate().map(|(i, &v)| (i, v.ln())),
        &CYAN,
    ))?
    .label("AdaGrad")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &CYAN));
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.9))
        .border_style(&BLACK)
        .label_font(("sans-serif", 26))
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    
    println!("Plot saved as 'rosenbrock_optimizer_comparison.png'");
    
    Ok(())
}

/// Run optimizer comparisons on both test functions
pub fn run_optimizer_comparisons() -> Result<(), Box<dyn Error>> {
    println!("Running optimizer comparisons...");
    
    // Run comparison on plateau function
    let max_iter = 150;

    compare_optimizers_plateau(max_iter)?;
    
    // Run comparison on Rosenbrock function
    compare_optimizers_rosenbrock(1000)?;
    
    println!("Optimizer comparisons completed successfully!");
    
    Ok(())
} 