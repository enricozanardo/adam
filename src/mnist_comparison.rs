use std::collections::HashMap;
use std::error::Error;
use ndarray::{Array1, Array2, Axis};
use ndarray::s;
use plotters::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::fs::File;
use std::io::Read;

use crate::{StandardAdam, EnhancedAdam};
use crate::optimizers::{SGDMomentum, RMSProp, AdaGrad};

const BATCH_SIZE: usize = 64;
const EPOCHS: usize = 5;
const HIDDEN_SIZE: usize = 128;
const INPUT_SIZE: usize = 784; // 28x28
const OUTPUT_SIZE: usize = 10;

/// Custom struct to hold MNIST data
struct MnistData {
    train_images: Vec<u8>,
    train_labels: Vec<u8>,
    test_images: Vec<u8>,
    test_labels: Vec<u8>,
}

/// Simple MLP neural network for MNIST
struct MnistMLP {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
}

impl MnistMLP {
    fn new() -> Self {
        // Initialize with small random values
        let w1 = Array2::from_shape_fn((INPUT_SIZE, HIDDEN_SIZE), |_| {
            rand::random::<f32>() * 0.01
        });
        let b1 = Array1::zeros(HIDDEN_SIZE);
        
        let w2 = Array2::from_shape_fn((HIDDEN_SIZE, OUTPUT_SIZE), |_| {
            rand::random::<f32>() * 0.01
        });
        let b2 = Array1::zeros(OUTPUT_SIZE);
        
        Self { w1, b1, w2, b2 }
    }
    
    /// Convert network parameters to HashMap for optimizers
    fn to_params(&self) -> HashMap<String, Array1<f32>> {
        let mut params = HashMap::new();
        params.insert("w1".to_string(), self.w1.clone().into_shape(INPUT_SIZE * HIDDEN_SIZE).unwrap());
        params.insert("b1".to_string(), self.b1.clone());
        params.insert("w2".to_string(), self.w2.clone().into_shape(HIDDEN_SIZE * OUTPUT_SIZE).unwrap());
        params.insert("b2".to_string(), self.b2.clone());
        params
    }
    
    /// Update network parameters from HashMap
    fn from_params(&mut self, params: &HashMap<String, Array1<f32>>) {
        self.w1 = params.get("w1").unwrap().clone().into_shape((INPUT_SIZE, HIDDEN_SIZE)).unwrap();
        self.b1 = params.get("b1").unwrap().clone();
        self.w2 = params.get("w2").unwrap().clone().into_shape((HIDDEN_SIZE, OUTPUT_SIZE)).unwrap();
        self.b2 = params.get("b2").unwrap().clone();
    }
    
    /// Forward pass
    fn forward(&self, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        // First layer: x -> h1
        let h1_linear = x.dot(&self.w1) + &self.b1;
        let h1 = h1_linear.mapv(|x| if x > 0.0 { x } else { 0.0 }); // ReLU
        
        // Second layer: h1 -> output
        let output = h1.dot(&self.w2) + &self.b2;
        
        (h1, output)
    }
    
    /// Calculate loss and gradients
    fn loss_and_grads(&self, x: &Array2<f32>, y: &Array2<f32>) -> (f32, HashMap<String, Array1<f32>>) {
        let batch_size = x.shape()[0];
        
        // Forward pass
        let (h1, output) = self.forward(x);
        
        // Softmax and cross-entropy loss
        let mut softmax = output.clone();
        for mut row in softmax.axis_iter_mut(Axis(0)) {
            let max_val = row.fold(std::f32::NEG_INFINITY, |a, &b| a.max(b));
            row.mapv_inplace(|x| (x - max_val).exp());
            let sum: f32 = row.sum();
            row.mapv_inplace(|x| x / sum);
        }
        
        // Calculate cross-entropy loss
        let mut loss = 0.0;
        for i in 0..batch_size {
            let y_i = y.row(i);
            let y_pred = softmax.row(i);
            for j in 0..OUTPUT_SIZE {
                if y_i[j] > 0.0 {
                    loss -= y_i[j] * y_pred[j].ln();
                }
            }
        }
        loss /= batch_size as f32;
        
        // Backpropagation
        let mut d_output = softmax - y;
        d_output /= batch_size as f32;
        
        // Gradients for second layer
        let d_w2 = h1.t().dot(&d_output);
        let d_b2 = d_output.sum_axis(Axis(0));
        
        // Gradients for first layer
        let d_h1 = d_output.dot(&self.w2.t());
        let d_h1_relu = d_h1.clone();
        let d_h1_relu = d_h1_relu.mapv(|x| if x > 0.0 { x } else { 0.0 });
        
        let d_w1 = x.t().dot(&d_h1_relu);
        let d_b1 = d_h1_relu.sum_axis(Axis(0));
        
        // Collect gradients
        let mut grads = HashMap::new();
        grads.insert("w1".to_string(), d_w1.into_shape(INPUT_SIZE * HIDDEN_SIZE).unwrap());
        grads.insert("b1".to_string(), d_b1);
        grads.insert("w2".to_string(), d_w2.into_shape(HIDDEN_SIZE * OUTPUT_SIZE).unwrap());
        grads.insert("b2".to_string(), d_b2);
        
        (loss, grads)
    }
    
    /// Predict class probabilities
    fn predict(&self, x: &Array2<f32>) -> Array2<f32> {
        let (_, output) = self.forward(x);
        
        // Apply softmax
        let mut softmax = output.clone();
        for mut row in softmax.axis_iter_mut(Axis(0)) {
            let max_val = row.fold(std::f32::NEG_INFINITY, |a, &b| a.max(b));
            row.mapv_inplace(|x| (x - max_val).exp());
            let sum: f32 = row.sum();
            row.mapv_inplace(|x| x / sum);
        }
        
        softmax
    }
    
    /// Calculate accuracy
    fn accuracy(&self, x: &Array2<f32>, y: &Array2<f32>) -> f32 {
        let predictions = self.predict(x);
        let mut correct = 0;
        
        for i in 0..x.shape()[0] {
            let pred_idx = predictions.row(i).iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap().0;
                
            let true_idx = y.row(i).iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap().0;
                
            if pred_idx == true_idx {
                correct += 1;
            }
        }
        
        correct as f32 / x.shape()[0] as f32
    }
}

/// Prepare MNIST data
fn load_mnist() -> Result<MnistData, Box<dyn Error>> {
    // Define path to the data directory
    let data_path = Path::new("data");
    
    // Check if the data directory exists
    if !data_path.exists() {
        return Err("Data directory not found. Please ensure the MNIST dataset is in the 'data' directory.".into());
    }
    
    println!("Loading MNIST dataset from {:?}", data_path);
    
    // Load training images
    let train_images_path = data_path.join("train-images.idx3-ubyte");
    let mut train_images_file = File::open(&train_images_path)
        .or_else(|_| File::open(data_path.join("train-images-idx3-ubyte")))
        .or_else(|_| File::open(data_path.join("train-images.idx3-ubyte")))
        .map_err(|e| format!("Failed to open training images file: {}", e))?;
    let mut train_images = Vec::new();
    train_images_file.read_to_end(&mut train_images)?;
    
    // Load training labels
    let train_labels_path = data_path.join("train-labels.idx1-ubyte");
    let mut train_labels_file = File::open(&train_labels_path)
        .or_else(|_| File::open(data_path.join("train-labels-idx1-ubyte")))
        .or_else(|_| File::open(data_path.join("train-labels.idx1-ubyte")))
        .map_err(|e| format!("Failed to open training labels file: {}", e))?;
    let mut train_labels = Vec::new();
    train_labels_file.read_to_end(&mut train_labels)?;
    
    // Load test images
    let test_images_path = data_path.join("t10k-images.idx3-ubyte");
    let mut test_images_file = File::open(&test_images_path)
        .or_else(|_| File::open(data_path.join("t10k-images-idx3-ubyte")))
        .or_else(|_| File::open(data_path.join("t10k-images.idx3-ubyte")))
        .map_err(|e| format!("Failed to open test images file: {}", e))?;
    let mut test_images = Vec::new();
    test_images_file.read_to_end(&mut test_images)?;
    
    // Load test labels
    let test_labels_path = data_path.join("t10k-labels.idx1-ubyte");
    let mut test_labels_file = File::open(&test_labels_path)
        .or_else(|_| File::open(data_path.join("t10k-labels-idx1-ubyte")))
        .or_else(|_| File::open(data_path.join("t10k-labels.idx1-ubyte")))
        .map_err(|e| format!("Failed to open test labels file: {}", e))?;
    let mut test_labels = Vec::new();
    test_labels_file.read_to_end(&mut test_labels)?;
    
    println!("Successfully loaded MNIST dataset:");
    println!("  Training images: {} bytes", train_images.len());
    println!("  Training labels: {} bytes", train_labels.len());
    println!("  Test images: {} bytes", test_images.len());
    println!("  Test labels: {} bytes", test_labels.len());
    
    Ok(MnistData {
        train_images,
        train_labels,
        test_images,
        test_labels,
    })
}

/// Parse IDX format data
fn parse_idx_images(data: &[u8]) -> Result<Array2<f32>, Box<dyn Error>> {
    // IDX format: [magic number (4 bytes)][num images (4 bytes)][num rows (4 bytes)][num cols (4 bytes)][pixels...]
    if data.len() < 16 {
        return Err("Invalid IDX file format: file too short".into());
    }
    
    // Check magic number (should be 0x00000803 for images)
    let magic = ((data[0] as u32) << 24) | ((data[1] as u32) << 16) | ((data[2] as u32) << 8) | (data[3] as u32);
    if magic != 0x00000803 {
        return Err(format!("Invalid magic number for images: {:x}", magic).into());
    }
    
    // Parse dimensions
    let num_images = ((data[4] as u32) << 24) | ((data[5] as u32) << 16) | ((data[6] as u32) << 8) | (data[7] as u32);
    let num_rows = ((data[8] as u32) << 24) | ((data[9] as u32) << 16) | ((data[10] as u32) << 8) | (data[11] as u32);
    let num_cols = ((data[12] as u32) << 24) | ((data[13] as u32) << 16) | ((data[14] as u32) << 8) | (data[15] as u32);
    
    let num_pixels = (num_rows * num_cols) as usize;
    let expected_size = 16 + (num_images as usize) * num_pixels;
    
    if data.len() < expected_size {
        return Err(format!("Invalid IDX file: expected {} bytes, got {}", expected_size, data.len()).into());
    }
    
    println!("Parsing {} images of size {}x{}", num_images, num_rows, num_cols);
    
    // Convert to f32 and normalize to [0, 1]
    let mut images = Vec::with_capacity(num_images as usize * num_pixels);
    for i in 0..num_images as usize {
        let offset = 16 + i * num_pixels;
        for j in 0..num_pixels {
            images.push(data[offset + j] as f32 / 255.0);
        }
    }
    
    // Reshape to (num_images, num_pixels)
    let images_array = Array2::from_shape_vec((num_images as usize, num_pixels), images)?;
    
    Ok(images_array)
}

/// Parse IDX format labels
fn parse_idx_labels(data: &[u8]) -> Result<Array2<f32>, Box<dyn Error>> {
    // IDX format: [magic number (4 bytes)][num items (4 bytes)][labels...]
    if data.len() < 8 {
        return Err("Invalid IDX file format: file too short".into());
    }
    
    // Check magic number (should be 0x00000801 for labels)
    let magic = ((data[0] as u32) << 24) | ((data[1] as u32) << 16) | ((data[2] as u32) << 8) | (data[3] as u32);
    if magic != 0x00000801 {
        return Err(format!("Invalid magic number for labels: {:x}", magic).into());
    }
    
    // Parse dimensions
    let num_items = ((data[4] as u32) << 24) | ((data[5] as u32) << 16) | ((data[6] as u32) << 8) | (data[7] as u32);
    
    let expected_size = 8 + num_items as usize;
    if data.len() < expected_size {
        return Err(format!("Invalid IDX file: expected {} bytes, got {}", expected_size, data.len()).into());
    }
    
    println!("Parsing {} labels", num_items);
    
    // Convert to one-hot encoding
    let mut labels = Array2::zeros((num_items as usize, 10));
    
    for i in 0..num_items as usize {
        let label = data[8 + i] as usize;
        if label < 10 {
            labels[[i, label]] = 1.0;
        } else {
            return Err(format!("Invalid label value: {}", label).into());
        }
    }
    
    Ok(labels)
}

/// Preprocess MNIST data
fn preprocess_mnist(mnist: &MnistData) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>), Box<dyn Error>> {
    // Parse training images
    let x_train = parse_idx_images(&mnist.train_images)?;
    
    // Parse training labels
    let y_train = parse_idx_labels(&mnist.train_labels)?;
    
    // Parse test images
    let x_test = parse_idx_images(&mnist.test_images)?;
    
    // Parse test labels
    let y_test = parse_idx_labels(&mnist.test_labels)?;
    
    println!("Dataset preprocessed successfully:");
    println!("  Training: {} samples", x_train.shape()[0]);
    println!("  Testing: {} samples", x_test.shape()[0]);
    
    Ok((x_train, y_train, x_test, y_test))
}

/// Compare optimizers on MNIST dataset
pub fn compare_optimizers_mnist() -> Result<(), Box<dyn Error>> {
    println!("Running optimizer comparison on MNIST dataset...");
    
    // Load and preprocess data
    let mnist = load_mnist()?;
    let (x_train, y_train, x_test, y_test) = preprocess_mnist(&mnist)?;
    
    // Initialize networks for each optimizer
    let mut standard_adam_model = MnistMLP::new();
    let mut enhanced_adam_model = MnistMLP::new();
    let mut sgd_momentum_model = MnistMLP::new();
    let mut rmsprop_model = MnistMLP::new();
    let mut adagrad_model = MnistMLP::new();
    
    // Initialize params for optimizers
    let mut standard_adam_params = standard_adam_model.to_params();
    let mut enhanced_adam_params = enhanced_adam_model.to_params();
    let mut sgd_momentum_params = sgd_momentum_model.to_params();
    let mut rmsprop_params = rmsprop_model.to_params();
    let mut adagrad_params = adagrad_model.to_params();
    
    // Create optimizers
    let mut standard_adam = StandardAdam::new(0.001, 0.9, 0.999, 1e-8);
    let mut enhanced_adam = EnhancedAdam::new(0.001, 0.9, 0.999, 1e-8);
    let mut sgd_momentum = SGDMomentum::new(0.01, 0.9); // Higher LR for SGD
    let mut rmsprop = RMSProp::new(0.001, 0.9, 1e-8);
    let mut adagrad = AdaGrad::new(0.01, 1e-8); // Higher LR for AdaGrad
    
    // Track losses and accuracies
    let n_batches = x_train.shape()[0] / BATCH_SIZE;
    let total_iterations = EPOCHS * n_batches;
    
    let mut standard_adam_losses = Vec::with_capacity(total_iterations);
    let mut enhanced_adam_losses = Vec::with_capacity(total_iterations);
    let mut sgd_momentum_losses = Vec::with_capacity(total_iterations);
    let mut rmsprop_losses = Vec::with_capacity(total_iterations);
    let mut adagrad_losses = Vec::with_capacity(total_iterations);
    
    let mut standard_adam_accs = Vec::with_capacity(EPOCHS);
    let mut enhanced_adam_accs = Vec::with_capacity(EPOCHS);
    let mut sgd_momentum_accs = Vec::with_capacity(EPOCHS);
    let mut rmsprop_accs = Vec::with_capacity(EPOCHS);
    let mut adagrad_accs = Vec::with_capacity(EPOCHS);
    
    // Setup progress bar
    let progress_bar = ProgressBar::new((total_iterations) as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .expect("Progress bar template error")
            .progress_chars("##-")
    );
    
    // Training loop
    let mut iteration = 0;
    for epoch in 0..EPOCHS {
        for b in 0..n_batches {
            // Extract batch
            let start_idx = b * BATCH_SIZE;
            let end_idx = (b + 1) * BATCH_SIZE;
            let x_batch = x_train.slice(s![start_idx..end_idx, ..]).to_owned();
            let y_batch = y_train.slice(s![start_idx..end_idx, ..]).to_owned();
            
            // Train Standard Adam
            let (standard_loss, standard_grads) = standard_adam_model.loss_and_grads(&x_batch, &y_batch);
            standard_adam.step(&mut standard_adam_params, &standard_grads);
            standard_adam_model.from_params(&standard_adam_params);
            standard_adam_losses.push(standard_loss);
            
            // Train Enhanced Adam
            let (enhanced_loss, enhanced_grads) = enhanced_adam_model.loss_and_grads(&x_batch, &y_batch);
            enhanced_adam.step(&mut enhanced_adam_params, &enhanced_grads);
            enhanced_adam_model.from_params(&enhanced_adam_params);
            enhanced_adam_losses.push(enhanced_loss);
            
            // Train SGD with Momentum
            let (sgd_loss, sgd_grads) = sgd_momentum_model.loss_and_grads(&x_batch, &y_batch);
            sgd_momentum.step(&mut sgd_momentum_params, &sgd_grads);
            sgd_momentum_model.from_params(&sgd_momentum_params);
            sgd_momentum_losses.push(sgd_loss);
            
            // Train RMSProp
            let (rmsprop_loss, rmsprop_grads) = rmsprop_model.loss_and_grads(&x_batch, &y_batch);
            rmsprop.step(&mut rmsprop_params, &rmsprop_grads);
            rmsprop_model.from_params(&rmsprop_params);
            rmsprop_losses.push(rmsprop_loss);
            
            // Train AdaGrad
            let (adagrad_loss, adagrad_grads) = adagrad_model.loss_and_grads(&x_batch, &y_batch);
            adagrad.step(&mut adagrad_params, &adagrad_grads);
            adagrad_model.from_params(&adagrad_params);
            adagrad_losses.push(adagrad_loss);
            
            // Update progress bar
            iteration += 1;
            progress_bar.set_position(iteration as u64);
            
            if b % 20 == 0 {
                progress_bar.set_message(format!(
                    "Epoch {}/{} - Batch {}/{} - Loss: {:.4} (S) {:.4} (E) {:.4} (SGD) {:.4} (RMS) {:.4} (Ada)", 
                    epoch + 1, EPOCHS, b, n_batches,
                    standard_loss, enhanced_loss, sgd_loss, rmsprop_loss, adagrad_loss
                ));
            }
        }
        
        // Calculate test accuracy after each epoch
        standard_adam_accs.push(standard_adam_model.accuracy(&x_test, &y_test));
        enhanced_adam_accs.push(enhanced_adam_model.accuracy(&x_test, &y_test));
        sgd_momentum_accs.push(sgd_momentum_model.accuracy(&x_test, &y_test));
        rmsprop_accs.push(rmsprop_model.accuracy(&x_test, &y_test));
        adagrad_accs.push(adagrad_model.accuracy(&x_test, &y_test));
        
        println!("Epoch {}/{} completed:", epoch + 1, EPOCHS);
        println!("  Standard Adam - Test Accuracy: {:.4}", standard_adam_accs.last().unwrap());
        println!("  Enhanced Adam - Test Accuracy: {:.4}", enhanced_adam_accs.last().unwrap());
        println!("  SGD Momentum  - Test Accuracy: {:.4}", sgd_momentum_accs.last().unwrap());
        println!("  RMSProp       - Test Accuracy: {:.4}", rmsprop_accs.last().unwrap());
        println!("  AdaGrad       - Test Accuracy: {:.4}", adagrad_accs.last().unwrap());
    }
    
    progress_bar.finish_with_message("Training completed!");
    
    // Print final results
    println!("\nFinal Results (MNIST Dataset):");
    println!("Standard Adam - Final test accuracy: {:.4}", standard_adam_accs.last().unwrap());
    println!("Enhanced Adam - Final test accuracy: {:.4}", enhanced_adam_accs.last().unwrap());
    println!("SGD Momentum  - Final test accuracy: {:.4}", sgd_momentum_accs.last().unwrap());
    println!("RMSProp       - Final test accuracy: {:.4}", rmsprop_accs.last().unwrap());
    println!("AdaGrad       - Final test accuracy: {:.4}", adagrad_accs.last().unwrap());
    
    // Calculate improvement ratios relative to Standard Adam
    let standard_final = *standard_adam_accs.last().unwrap();
    let enhanced_ratio = enhanced_adam_accs.last().unwrap() / standard_final;
    let sgd_ratio = sgd_momentum_accs.last().unwrap() / standard_final;
    let rmsprop_ratio = rmsprop_accs.last().unwrap() / standard_final;
    let adagrad_ratio = adagrad_accs.last().unwrap() / standard_final;
    
    println!("\nImprovement Ratios (relative to Standard Adam):");
    println!("Enhanced Adam: {:.2}x", enhanced_ratio);
    println!("SGD Momentum:  {:.2}x", sgd_ratio);
    println!("RMSProp:       {:.2}x", rmsprop_ratio);
    println!("AdaGrad:       {:.2}x", adagrad_ratio);
    
    // Create plots
    // 1. Loss plot
    let root = BitMapBackend::new("figures/mnist_loss_optimizer_comparison.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let max_loss = standard_adam_losses.iter()
        .chain(enhanced_adam_losses.iter())
        .chain(sgd_momentum_losses.iter())
        .chain(rmsprop_losses.iter())
        .chain(adagrad_losses.iter())
        .fold(0.0f32, |a, &b| a.max(b))
        .min(10.0); // Cap for better visualization
    
    let min_loss = standard_adam_losses.iter()
        .chain(enhanced_adam_losses.iter())
        .chain(sgd_momentum_losses.iter())
        .chain(rmsprop_losses.iter())
        .chain(adagrad_losses.iter())
        .fold(max_loss, |a, &b| a.min(b));
    
    // Add padding to prevent label overlap
    let loss_range = max_loss - min_loss;
    let padded_min = (min_loss - loss_range * 0.05).max(0.0);
    let padded_max = max_loss + loss_range * 0.1;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("MNIST Loss Comparison", ("sans-serif", 26).into_font())
        .margin(40)
        .x_label_area_size(80)
        .y_label_area_size(120)
        .build_cartesian_2d(0..total_iterations, padded_min..padded_max)?;
    
    chart.configure_mesh()
         .disable_mesh()
         .x_desc("Iterations")
         .y_desc("Loss")
         .x_label_formatter(&|x| format!("{}", x))
         .y_label_formatter(&|y| format!("{:.3}", y))
         .x_labels(6)
         .y_labels(6)
         .axis_desc_style(("sans-serif", 26))
         .label_style(("sans-serif", 26))
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
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 26))
        .draw()?;
    
    println!("Loss plot saved as 'mnist_loss_optimizer_comparison.png'");
    
    // 2. Accuracy plot
    let root = BitMapBackend::new("figures/mnist_accuracy_optimizer_comparison.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let max_acc = standard_adam_accs.iter()
        .chain(enhanced_adam_accs.iter())
        .chain(sgd_momentum_accs.iter())
        .chain(rmsprop_accs.iter())
        .chain(adagrad_accs.iter())
        .fold(0.0f32, |a, &b| a.max(b));
    
    let min_acc = standard_adam_accs.iter()
        .chain(enhanced_adam_accs.iter())
        .chain(sgd_momentum_accs.iter())
        .chain(rmsprop_accs.iter())
        .chain(adagrad_accs.iter())
        .fold(max_acc, |a, &b| a.min(b));
    
    // Add padding to prevent label overlap
    let acc_range = max_acc - min_acc;
    let padded_min_acc = (min_acc - acc_range * 0.05).max(0.0);
    let padded_max_acc = (max_acc + acc_range * 0.1).min(1.0);
    
    let mut chart = ChartBuilder::on(&root)
        .caption("MNIST Accuracy Comparison", ("sans-serif", 26).into_font())
        .margin(40)
        .x_label_area_size(80)
        .y_label_area_size(120)
        .build_cartesian_2d(0..EPOCHS, padded_min_acc..padded_max_acc)?;
    
    chart.configure_mesh()
         .disable_mesh()
         .x_desc("Epochs")
         .y_desc("Accuracy")
         .x_label_formatter(&|x| format!("{}", x))
         .y_label_formatter(&|y| format!("{:.3}", y))
         .x_labels(6)
         .y_labels(6)
         .axis_desc_style(("sans-serif", 26))
         .label_style(("sans-serif", 26))
         .draw()?;
    
    chart.draw_series(LineSeries::new(
        standard_adam_accs.iter().enumerate().map(|(i, &v)| (i, v)),
        RED.stroke_width(3),
    ))?
    .label("Standard Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
    
    chart.draw_series(LineSeries::new(
        enhanced_adam_accs.iter().enumerate().map(|(i, &v)| (i, v)),
        BLUE.stroke_width(3),
    ))?
    .label("Enhanced Adam")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));
    
    chart.draw_series(LineSeries::new(
        sgd_momentum_accs.iter().enumerate().map(|(i, &v)| (i, v)),
        &GREEN,
    ))?
    .label("SGD with Momentum")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
    
    chart.draw_series(LineSeries::new(
        rmsprop_accs.iter().enumerate().map(|(i, &v)| (i, v)),
        &MAGENTA,
    ))?
    .label("RMSProp")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));
    
    chart.draw_series(LineSeries::new(
        adagrad_accs.iter().enumerate().map(|(i, &v)| (i, v)),
        &CYAN,
    ))?
    .label("AdaGrad")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &CYAN));
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("sans-serif", 26))
        .draw()?;
    
    println!("Accuracy plot saved as 'mnist_accuracy_optimizer_comparison.png'");
    
    Ok(())
} 