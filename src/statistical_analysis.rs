use std::collections::HashMap;
use std::error::Error;
use ndarray::Array1;
use plotters::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

use crate::problem_with_plateau;
use crate::gradient;
use crate::rosenbrock;
use crate::rosenbrock_gradient;
use crate::{StandardAdam, EnhancedAdam};
use crate::optimizers::{SGDMomentum, RMSProp, AdaGrad};

// Statistical metrics for a series of runs
pub struct StatisticalMetrics {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub confidence_interval_95: (f32, f32),
    pub all_values: Vec<f32>,
}

impl StatisticalMetrics {
    pub fn new(values: &[f32]) -> Self {
        let n = values.len() as f32;
        
        // Calculate mean
        let sum: f32 = values.iter().sum();
        let mean = sum / n;
        
        // Calculate variance and std dev
        let variance: f32 = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / n;
        let std_dev = variance.sqrt();
        
        // Min and max - fixed to avoid reference ownership issues
        let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // 95% confidence interval (using t-distribution approximation)
        // For simplicity, using 1.96 as the z-score for 95% CI when n ≥ 30
        // For smaller n, should use t-distribution critical values
        let z_score = if n >= 30.0 { 1.96 } else { 2.093 }; // 2.093 is t-critical for n=10, df=9
        let margin_of_error = z_score * (std_dev / n.sqrt());
        let confidence_interval_95 = (mean - margin_of_error, mean + margin_of_error);
        
        Self {
            mean,
            std_dev,
            min,
            max,
            confidence_interval_95,
            all_values: values.to_vec(),
        }
    }
    
    pub fn print(&self, name: &str) {
        println!("{} Statistics:", name);
        println!("  Mean: {:.6}", self.mean);
        println!("  Std Dev: {:.6}", self.std_dev);
        println!("  Min: {:.6}", self.min);
        println!("  Max: {:.6}", self.max);
        println!("  95% CI: ({:.6}, {:.6})", self.confidence_interval_95.0, self.confidence_interval_95.1);
    }
}

/// Run statistical analysis on plateau function
pub fn run_plateau_statistical_analysis(num_runs: usize, max_iter: usize) -> Result<HashMap<String, StatisticalMetrics>, Box<dyn Error>> {
    println!("Running statistical analysis on plateau function with {} runs...", num_runs);
    
    // Create a progress bar
    let progress_bar = ProgressBar::new(num_runs as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .expect("Progress bar template error")
            .progress_chars("##-")
    );
    
    // Collect final losses for each optimizer across multiple runs
    let mut standard_adam_losses = Vec::with_capacity(num_runs);
    let mut enhanced_adam_losses = Vec::with_capacity(num_runs);
    let mut sgd_momentum_losses = Vec::with_capacity(num_runs);
    let mut rmsprop_losses = Vec::with_capacity(num_runs);
    let mut adagrad_losses = Vec::with_capacity(num_runs);
    
    // Run multiple optimizations in parallel
    let results: Vec<_> = (0..num_runs).into_par_iter().map(|run| {
        // Seed for reproducibility but different for each run
        let seed = 42 + run as u64;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        // Initialize parameters with same random starting point
        let init_point = Array1::from_vec(vec![
            rng.gen_range(-0.5..0.5), 
            rng.gen_range(-0.5..0.5)
        ]);
        
        let mut standard_params = HashMap::new();
        let mut enhanced_params = HashMap::new();
        let mut sgd_params = HashMap::new();
        let mut rmsprop_params = HashMap::new();
        let mut adagrad_params = HashMap::new();
        
        standard_params.insert("weights".to_string(), init_point.clone());
        enhanced_params.insert("weights".to_string(), init_point.clone());
        sgd_params.insert("weights".to_string(), init_point.clone());
        rmsprop_params.insert("weights".to_string(), init_point.clone());
        adagrad_params.insert("weights".to_string(), init_point.clone());
        
        // Create optimizers
        let mut standard_adam = StandardAdam::new(0.01, 0.9, 0.999, 1e-8);
        let mut enhanced_adam = EnhancedAdam::new(0.01, 0.9, 0.999, 1e-8);
        let mut sgd_momentum = SGDMomentum::new(0.01, 0.9);
        let mut rmsprop = RMSProp::new(0.01, 0.9, 1e-8);
        let mut adagrad = AdaGrad::new(0.01, 1e-8);
        
        // Run optimization
        for _ in 0..max_iter {
            // Standard Adam
            let x = standard_params.get("weights").unwrap();
            let grad = gradient(x);
            let mut grads = HashMap::new();
            grads.insert("weights".to_string(), grad);
            standard_adam.step(&mut standard_params, &grads);
            
            // Enhanced Adam
            let x = enhanced_params.get("weights").unwrap();
            let grad = gradient(x);
            let mut grads = HashMap::new();
            grads.insert("weights".to_string(), grad);
            enhanced_adam.step(&mut enhanced_params, &grads);
            
            // SGD with Momentum
            let x = sgd_params.get("weights").unwrap();
            let grad = gradient(x);
            let mut grads = HashMap::new();
            grads.insert("weights".to_string(), grad);
            sgd_momentum.step(&mut sgd_params, &grads);
            
            // RMSProp
            let x = rmsprop_params.get("weights").unwrap();
            let grad = gradient(x);
            let mut grads = HashMap::new();
            grads.insert("weights".to_string(), grad);
            rmsprop.step(&mut rmsprop_params, &grads);
            
            // AdaGrad
            let x = adagrad_params.get("weights").unwrap();
            let grad = gradient(x);
            let mut grads = HashMap::new();
            grads.insert("weights".to_string(), grad);
            adagrad.step(&mut adagrad_params, &grads);
        }
        
        // Compute final losses
        let standard_final_loss = problem_with_plateau(standard_params.get("weights").unwrap());
        let enhanced_final_loss = problem_with_plateau(enhanced_params.get("weights").unwrap());
        let sgd_final_loss = problem_with_plateau(sgd_params.get("weights").unwrap());
        let rmsprop_final_loss = problem_with_plateau(rmsprop_params.get("weights").unwrap());
        let adagrad_final_loss = problem_with_plateau(adagrad_params.get("weights").unwrap());
        
        (standard_final_loss, enhanced_final_loss, sgd_final_loss, rmsprop_final_loss, adagrad_final_loss)
    }).collect();
    
    // Process results
    for (i, (standard, enhanced, sgd, rmsprop, adagrad)) in results.iter().enumerate() {
        standard_adam_losses.push(*standard);
        enhanced_adam_losses.push(*enhanced);
        sgd_momentum_losses.push(*sgd);
        rmsprop_losses.push(*rmsprop);
        adagrad_losses.push(*adagrad);
        
        progress_bar.set_position((i + 1) as u64);
    }
    
    progress_bar.finish_with_message("Plateau function statistical analysis completed!");
    
    // Compute statistics
    let standard_stats = StatisticalMetrics::new(&standard_adam_losses);
    let enhanced_stats = StatisticalMetrics::new(&enhanced_adam_losses);
    let sgd_stats = StatisticalMetrics::new(&sgd_momentum_losses);
    let rmsprop_stats = StatisticalMetrics::new(&rmsprop_losses);
    let adagrad_stats = StatisticalMetrics::new(&adagrad_losses);
    
    // Print statistics
    standard_stats.print("Standard Adam");
    enhanced_stats.print("Enhanced Adam");
    sgd_stats.print("SGD with Momentum");
    rmsprop_stats.print("RMSProp");
    adagrad_stats.print("AdaGrad");
    
    // Calculate improvement ratios with confidence intervals
    let standard_mean = standard_stats.mean;
    
    println!("\nImprovement Ratios (relative to Standard Adam):");
    println!("Enhanced Adam: {:.2}x (95% CI: {:.2}x to {:.2}x)", 
             standard_mean / enhanced_stats.mean,
             standard_mean / enhanced_stats.confidence_interval_95.1,
             standard_mean / enhanced_stats.confidence_interval_95.0);
    println!("SGD Momentum: {:.2}x (95% CI: {:.2}x to {:.2}x)", 
             standard_mean / sgd_stats.mean,
             standard_mean / sgd_stats.confidence_interval_95.1,
             standard_mean / sgd_stats.confidence_interval_95.0);
    println!("RMSProp: {:.2}x (95% CI: {:.2}x to {:.2}x)", 
             standard_mean / rmsprop_stats.mean,
             standard_mean / rmsprop_stats.confidence_interval_95.1,
             standard_mean / rmsprop_stats.confidence_interval_95.0);
    println!("AdaGrad: {:.2}x (95% CI: {:.2}x to {:.2}x)", 
             standard_mean / adagrad_stats.mean,
             standard_mean / adagrad_stats.confidence_interval_95.1,
             standard_mean / adagrad_stats.confidence_interval_95.0);
    
    // Create and save box plot
    let root = BitMapBackend::new("figures/plateau_statistical_boxplot.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Find the y-axis range
    let min_loss = vec![
        standard_stats.min,
        enhanced_stats.min,
        sgd_stats.min,
        rmsprop_stats.min,
        adagrad_stats.min
    ].into_iter().fold(f32::INFINITY, |a: f32, b| a.min(b));
    
    let max_loss: f32 = vec![
        standard_stats.max,
        enhanced_stats.max,
        sgd_stats.max,
        rmsprop_stats.max,
        adagrad_stats.max
    ].into_iter().fold(0.0f32, |a: f32, b| a.max(b));
    
    // Add padding to prevent label overlap
    let y_range = max_loss - min_loss;
    let y_min = min_loss - y_range * 0.05;
    let y_max = max_loss + y_range * 0.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Optimizer Performance Distribution (Plateau Function)", ("sans-serif", 26).into_font())
        .margin(40)
        .x_label_area_size(80)
        .y_label_area_size(120)
        .build_cartesian_2d(0.0f32..5.0f32, y_min..y_max)?;
    
    chart.configure_mesh()
         .x_labels(6)
         .x_desc("Optimizer")
         .y_desc("Final Loss")
         .axis_desc_style(("sans-serif", 26))
         .label_style(("sans-serif", 26))
         .x_label_formatter(&|x| {
             match x.floor() as i32 {
                 0 => "Standard Adam",
                 1 => "Enhanced Adam",
                 2 => "SGD+Momentum",
                 3 => "RMSProp",
                 4 => "AdaGrad",
                 _ => "",
             }.to_string()
         })
         .disable_mesh()
         .draw()?;
    
    // Draw box plots
    let box_width = 0.7;
    
    // Add series for the legend
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 0.0), (0.0, 0.0)],
        RED.mix(0.5).filled(),
    )))?.label("Standard Adam").legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], RED.mix(0.5).filled()));
    
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 0.0), (0.0, 0.0)],
        BLUE.mix(0.5).filled(),
    )))?.label("Enhanced Adam").legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], BLUE.mix(0.5).filled()));
    
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 0.0), (0.0, 0.0)],
        GREEN.mix(0.5).filled(),
    )))?.label("SGD Momentum").legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], GREEN.mix(0.5).filled()));
    
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 0.0), (0.0, 0.0)],
        MAGENTA.mix(0.5).filled(),
    )))?.label("RMSProp").legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], MAGENTA.mix(0.5).filled()));
    
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 0.0), (0.0, 0.0)],
        CYAN.mix(0.5).filled(),
    )))?.label("AdaGrad").legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], CYAN.mix(0.5).filled()));
    
    // Helper function to draw a box plot for a set of data
    let draw_boxplot = |chart: &mut ChartContext<_, _>, index: usize, data: &StatisticalMetrics, color: &RGBColor| {
        // Box (25th to 75th percentile)
        let sorted_data = {
            let mut data = data.all_values.clone();
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());
            data
        };
        
        let q1 = sorted_data[sorted_data.len() / 4];
        let q3 = sorted_data[sorted_data.len() * 3 / 4];
        
        // Draw box
        chart.draw_series(std::iter::once(Rectangle::new(
            [(index as f32 - box_width/2.0, q1), (index as f32 + box_width/2.0, q3)],
            color.clone().mix(0.5).filled(),
        ))).unwrap();
        
        // Draw median line
        let median = sorted_data[sorted_data.len() / 2];
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(index as f32 - box_width/2.0, median), (index as f32 + box_width/2.0, median)],
            color.clone().stroke_width(2),
        ))).unwrap();
        
        // Draw whiskers (min to max)
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(index as f32, data.min), (index as f32, q1)],
            color.clone().stroke_width(1),
        ))).unwrap();
        
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(index as f32, q3), (index as f32, data.max)],
            color.clone().stroke_width(1),
        ))).unwrap();
        
        // Draw caps on whiskers
        let cap_width = box_width / 3.0;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (index as f32 - cap_width/2.0, data.min), 
                (index as f32 + cap_width/2.0, data.min)
            ],
            color.clone().stroke_width(1),
        ))).unwrap();
        
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (index as f32 - cap_width/2.0, data.max), 
                (index as f32 + cap_width/2.0, data.max)
            ],
            color.clone().stroke_width(1),
        ))).unwrap();
        
        // Draw mean point
        chart.draw_series(std::iter::once(Circle::new(
            (index as f32, data.mean),
            3,
            color.clone().filled(),
        ))).unwrap();
    };
    
    draw_boxplot(&mut chart, 0, &standard_stats, &RED);
    draw_boxplot(&mut chart, 1, &enhanced_stats, &BLUE);
    draw_boxplot(&mut chart, 2, &sgd_stats, &GREEN);
    draw_boxplot(&mut chart, 3, &rmsprop_stats, &MAGENTA);
    draw_boxplot(&mut chart, 4, &adagrad_stats, &CYAN);
    
    // Add legend
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", 26))
        .draw()?;
    
    println!("Box plot saved as 'plateau_statistical_boxplot.png'");
    
    // Return all statistics in a hashmap
    let mut stats = HashMap::new();
    stats.insert("Standard Adam".to_string(), standard_stats);
    stats.insert("Enhanced Adam".to_string(), enhanced_stats);
    stats.insert("SGD with Momentum".to_string(), sgd_stats);
    stats.insert("RMSProp".to_string(), rmsprop_stats);
    stats.insert("AdaGrad".to_string(), adagrad_stats);
    
    Ok(stats)
}

/// Run statistical analysis on Rosenbrock function
pub fn run_rosenbrock_statistical_analysis(num_runs: usize, max_iter: usize) -> Result<HashMap<String, StatisticalMetrics>, Box<dyn Error>> {
    println!("Running statistical analysis on Rosenbrock function with {} runs...", num_runs);
    
    // Create a progress bar
    let progress_bar = ProgressBar::new(num_runs as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .expect("Progress bar template error")
            .progress_chars("##-")
    );
    
    // Collect final losses for each optimizer across multiple runs
    let mut standard_adam_losses = Vec::with_capacity(num_runs);
    let mut enhanced_adam_losses = Vec::with_capacity(num_runs);
    let mut sgd_momentum_losses = Vec::with_capacity(num_runs);
    let mut rmsprop_losses = Vec::with_capacity(num_runs);
    let mut adagrad_losses = Vec::with_capacity(num_runs);
    
    // Run multiple optimizations in parallel
    let results: Vec<_> = (0..num_runs).into_par_iter().map(|run| {
        // Seed for reproducibility but different for each run
        let seed = 42 + run as u64;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        // Create a base point and perturb it slightly for each run
        let base_point = vec![-1.2, 1.0, -0.5, 0.8, -1.0];
        let mut init_point = Array1::zeros(base_point.len());
        
        // Add small random perturbations to starting point
        for i in 0..base_point.len() {
            init_point[i] = base_point[i] + rng.gen_range(-0.1..0.1);
        }
        
        let mut standard_params = HashMap::new();
        let mut enhanced_params = HashMap::new();
        let mut sgd_params = HashMap::new();
        let mut rmsprop_params = HashMap::new();
        let mut adagrad_params = HashMap::new();
        
        standard_params.insert("weights".to_string(), init_point.clone());
        enhanced_params.insert("weights".to_string(), init_point.clone());
        sgd_params.insert("weights".to_string(), init_point.clone());
        rmsprop_params.insert("weights".to_string(), init_point.clone());
        adagrad_params.insert("weights".to_string(), init_point.clone());
        
        // Create optimizers
        let mut standard_adam = StandardAdam::new(0.001, 0.9, 0.999, 1e-8);
        let mut enhanced_adam = EnhancedAdam::new(0.001, 0.9, 0.999, 1e-8);
        let mut sgd_momentum = SGDMomentum::new(0.001, 0.9);
        let mut rmsprop = RMSProp::new(0.001, 0.9, 1e-8);
        let mut adagrad = AdaGrad::new(0.001, 1e-8);
        
        // Run optimization
        for _ in 0..max_iter {
            // Standard Adam
            let x = standard_params.get("weights").unwrap();
            let grad = rosenbrock_gradient(x);
            let mut grads = HashMap::new();
            grads.insert("weights".to_string(), grad);
            standard_adam.step(&mut standard_params, &grads);
            
            // Enhanced Adam
            let x = enhanced_params.get("weights").unwrap();
            let grad = rosenbrock_gradient(x);
            let mut grads = HashMap::new();
            grads.insert("weights".to_string(), grad);
            enhanced_adam.step(&mut enhanced_params, &grads);
            
            // SGD with Momentum
            let x = sgd_params.get("weights").unwrap();
            let grad = rosenbrock_gradient(x);
            let mut grads = HashMap::new();
            grads.insert("weights".to_string(), grad);
            sgd_momentum.step(&mut sgd_params, &grads);
            
            // RMSProp
            let x = rmsprop_params.get("weights").unwrap();
            let grad = rosenbrock_gradient(x);
            let mut grads = HashMap::new();
            grads.insert("weights".to_string(), grad);
            rmsprop.step(&mut rmsprop_params, &grads);
            
            // AdaGrad
            let x = adagrad_params.get("weights").unwrap();
            let grad = rosenbrock_gradient(x);
            let mut grads = HashMap::new();
            grads.insert("weights".to_string(), grad);
            adagrad.step(&mut adagrad_params, &grads);
        }
        
        // Compute final losses
        let standard_final_loss = rosenbrock(standard_params.get("weights").unwrap());
        let enhanced_final_loss = rosenbrock(enhanced_params.get("weights").unwrap());
        let sgd_final_loss = rosenbrock(sgd_params.get("weights").unwrap());
        let rmsprop_final_loss = rosenbrock(rmsprop_params.get("weights").unwrap());
        let adagrad_final_loss = rosenbrock(adagrad_params.get("weights").unwrap());
        
        (standard_final_loss, enhanced_final_loss, sgd_final_loss, rmsprop_final_loss, adagrad_final_loss)
    }).collect();
    
    // Process results
    for (i, (standard, enhanced, sgd, rmsprop, adagrad)) in results.iter().enumerate() {
        standard_adam_losses.push(*standard);
        enhanced_adam_losses.push(*enhanced);
        sgd_momentum_losses.push(*sgd);
        rmsprop_losses.push(*rmsprop);
        adagrad_losses.push(*adagrad);
        
        progress_bar.set_position((i + 1) as u64);
    }
    
    progress_bar.finish_with_message("Rosenbrock function statistical analysis completed!");
    
    // Compute statistics
    let standard_stats = StatisticalMetrics::new(&standard_adam_losses);
    let enhanced_stats = StatisticalMetrics::new(&enhanced_adam_losses);
    let sgd_stats = StatisticalMetrics::new(&sgd_momentum_losses);
    let rmsprop_stats = StatisticalMetrics::new(&rmsprop_losses);
    let adagrad_stats = StatisticalMetrics::new(&adagrad_losses);
    
    // Print statistics
    standard_stats.print("Standard Adam");
    enhanced_stats.print("Enhanced Adam");
    sgd_stats.print("SGD with Momentum");
    rmsprop_stats.print("RMSProp");
    adagrad_stats.print("AdaGrad");
    
    // Calculate improvement ratios with confidence intervals
    let standard_mean = standard_stats.mean;
    
    println!("\nImprovement Ratios (relative to Standard Adam):");
    println!("Enhanced Adam: {:.2}x (95% CI: {:.2}x to {:.2}x)", 
             standard_mean / enhanced_stats.mean,
             standard_mean / enhanced_stats.confidence_interval_95.1,
             standard_mean / enhanced_stats.confidence_interval_95.0);
    println!("SGD Momentum: {:.2}x (95% CI: {:.2}x to {:.2}x)", 
             standard_mean / sgd_stats.mean,
             standard_mean / sgd_stats.confidence_interval_95.1,
             standard_mean / sgd_stats.confidence_interval_95.0);
    println!("RMSProp: {:.2}x (95% CI: {:.2}x to {:.2}x)", 
             standard_mean / rmsprop_stats.mean,
             standard_mean / rmsprop_stats.confidence_interval_95.1,
             standard_mean / rmsprop_stats.confidence_interval_95.0);
    println!("AdaGrad: {:.2}x (95% CI: {:.2}x to {:.2}x)", 
             standard_mean / adagrad_stats.mean,
             standard_mean / adagrad_stats.confidence_interval_95.1,
             standard_mean / adagrad_stats.confidence_interval_95.0);
    
    // Create and save box plot
    let root = BitMapBackend::new("figures/rosenbrock_statistical_boxplot.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Since Rosenbrock values can vary widely, use log scale
    let min_loss = vec![
        standard_stats.min,
        enhanced_stats.min,
        sgd_stats.min,
        rmsprop_stats.min,
        adagrad_stats.min
    ].into_iter().fold(f32::INFINITY, |a: f32, b| a.min(b)).max(1e-6); // Ensure positive for log scale
    
    let max_loss: f32 = vec![
        standard_stats.max,
        enhanced_stats.max,
        sgd_stats.max,
        rmsprop_stats.max,
        adagrad_stats.max
    ].into_iter().fold(0.0f32, |a: f32, b| a.max(b));
    
    // Add padding for log scale
    let log_min = min_loss.ln();
    let log_max = max_loss.ln();
    let log_range = log_max - log_min;
    let y_min = log_min - log_range * 0.05;
    let y_max = log_max + log_range * 0.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Optimizer Performance Distribution (Rosenbrock Function)", ("sans-serif", 26).into_font())
        .margin(40)
        .x_label_area_size(80)
        .y_label_area_size(120)
        .build_cartesian_2d(0.0f32..5.0f32, y_min..y_max)?;
    
    chart.configure_mesh()
         .x_labels(6)
         .x_desc("Optimizer")
         .y_desc("Log(Final Loss)")
         .axis_desc_style(("sans-serif", 26))
         .label_style(("sans-serif", 26))
         .x_label_formatter(&|x| {
             match x.floor() as i32 {
                 0 => "Standard Adam",
                 1 => "Enhanced Adam",
                 2 => "SGD+Momentum",
                 3 => "RMSProp",
                 4 => "AdaGrad",
                 _ => "",
             }.to_string()
         })
         .disable_mesh()
         .draw()?;
    
    // Draw box plots
    let box_width = 0.7;
    
    // Add series for the legend
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 0.0), (0.0, 0.0)],
        RED.mix(0.5).filled(),
    )))?.label("Standard Adam").legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], RED.mix(0.5).filled()));
    
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 0.0), (0.0, 0.0)],
        BLUE.mix(0.5).filled(),
    )))?.label("Enhanced Adam").legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], BLUE.mix(0.5).filled()));
    
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 0.0), (0.0, 0.0)],
        GREEN.mix(0.5).filled(),
    )))?.label("SGD Momentum").legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], GREEN.mix(0.5).filled()));
    
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 0.0), (0.0, 0.0)],
        MAGENTA.mix(0.5).filled(),
    )))?.label("RMSProp").legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], MAGENTA.mix(0.5).filled()));
    
    chart.draw_series(std::iter::once(Rectangle::new(
        [(0.0, 0.0), (0.0, 0.0)],
        CYAN.mix(0.5).filled(),
    )))?.label("AdaGrad").legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], CYAN.mix(0.5).filled()));
    
    // Helper function to draw a box plot for a set of data with log transformation
    let draw_boxplot = |chart: &mut ChartContext<_, _>, index: usize, data: &StatisticalMetrics, color: &RGBColor| {
        // Transform data to log scale
        let log_data: Vec<f32> = data.all_values.iter().map(|&x| x.max(min_loss).ln()).collect();
        
        // Sort for percentiles
        let mut sorted_log_data = log_data.clone();
        sorted_log_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let q1 = sorted_log_data[sorted_log_data.len() / 4];
        let q3 = sorted_log_data[sorted_log_data.len() * 3 / 4];
        let median = sorted_log_data[sorted_log_data.len() / 2];
        let min = *sorted_log_data.first().unwrap();
        let max = *sorted_log_data.last().unwrap();
        let mean = data.mean.max(min_loss).ln();
        
        // Draw box
        chart.draw_series(std::iter::once(Rectangle::new(
            [(index as f32 - box_width/2.0, q1), (index as f32 + box_width/2.0, q3)],
            color.clone().mix(0.5).filled(),
        ))).unwrap();
        
        // Draw median line
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(index as f32 - box_width/2.0, median), (index as f32 + box_width/2.0, median)],
            color.clone().stroke_width(2),
        ))).unwrap();
        
        // Draw whiskers
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(index as f32, min), (index as f32, q1)],
            color.clone().stroke_width(1),
        ))).unwrap();
        
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(index as f32, q3), (index as f32, max)],
            color.clone().stroke_width(1),
        ))).unwrap();
        
        // Draw caps on whiskers
        let cap_width = box_width / 3.0;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (index as f32 - cap_width/2.0, min), 
                (index as f32 + cap_width/2.0, min)
            ],
            color.clone().stroke_width(1),
        ))).unwrap();
        
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (index as f32 - cap_width/2.0, max), 
                (index as f32 + cap_width/2.0, max)
            ],
            color.clone().stroke_width(1),
        ))).unwrap();
        
        // Draw mean point
        chart.draw_series(std::iter::once(Circle::new(
            (index as f32, mean),
            3,
            color.clone().filled(),
        ))).unwrap();
    };
    
    draw_boxplot(&mut chart, 0, &standard_stats, &RED);
    draw_boxplot(&mut chart, 1, &enhanced_stats, &BLUE);
    draw_boxplot(&mut chart, 2, &sgd_stats, &GREEN);
    draw_boxplot(&mut chart, 3, &rmsprop_stats, &MAGENTA);
    draw_boxplot(&mut chart, 4, &adagrad_stats, &CYAN);
    
    // Add legend
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", 26))
        .draw()?;
    
    println!("Box plot saved as 'rosenbrock_statistical_boxplot.png'");
    
    // Return all statistics in a hashmap
    let mut stats = HashMap::new();
    stats.insert("Standard Adam".to_string(), standard_stats);
    stats.insert("Enhanced Adam".to_string(), enhanced_stats);
    stats.insert("SGD with Momentum".to_string(), sgd_stats);
    stats.insert("RMSProp".to_string(), rmsprop_stats);
    stats.insert("AdaGrad".to_string(), adagrad_stats);
    
    Ok(stats)
}

/// Run all statistical analyses
pub fn run_statistical_analyses(num_runs: usize) -> Result<(), Box<dyn Error>> {
    println!("Running statistical analyses with {} runs for each test...", num_runs);
    
    // Run analyses
    run_plateau_statistical_analysis(num_runs, 1000)?;
    run_rosenbrock_statistical_analysis(num_runs, 1000)?;
    
    println!("All statistical analyses completed successfully!");
    
    Ok(())
} 