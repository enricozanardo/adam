use crate::{StandardAdam, EnhancedAdam};
use std::collections::HashMap;
use std::error::Error;
use ndarray::{Array, Array1, Array2, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use indicatif::{ProgressBar, ProgressStyle};
use plotters::prelude::*;
use std::path::Path;
use std::fs::File;
use std::io::Read;

/// Simple fully-connected neural network for MNIST classification
#[allow(dead_code)]
pub struct MnistMLP {
    // Layer dimensions
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    
    // Network parameters
    weights: HashMap<String, Array2<f32>>,
    biases: HashMap<String, Array1<f32>>,
    
    // Cached activations for backward pass
    activations: HashMap<String, Array2<f32>>,
}

impl MnistMLP {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut weights = HashMap::new();
        let mut biases = HashMap::new();
        
        // Initialize with Xavier/Glorot initialization
        let w1_bound = (6.0 / (input_dim + hidden_dim) as f32).sqrt();
        let w2_bound = (6.0 / (hidden_dim + output_dim) as f32).sqrt();
        
        // First layer weights and biases
        let w1 = Array::random((input_dim, hidden_dim), Uniform::new(-w1_bound, w1_bound));
        let b1 = Array::zeros(hidden_dim);
        
        // Second layer weights and biases
        let w2 = Array::random((hidden_dim, output_dim), Uniform::new(-w2_bound, w2_bound));
        let b2 = Array::zeros(output_dim);
        
        weights.insert("w1".to_string(), w1);
        weights.insert("w2".to_string(), w2);
        biases.insert("b1".to_string(), b1);
        biases.insert("b2".to_string(), b2);
        
        Self {
            input_dim,
            hidden_dim,
            output_dim,
            weights,
            biases,
            activations: HashMap::new(),
        }
    }
    
    /// ReLU activation function
    fn relu(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| if v > 0.0 { v } else { 0.0 })
    }
    
    /// Derivative of ReLU
    fn relu_prime(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }
    
    /// Softmax activation for output layer
    fn softmax(x: &Array2<f32>) -> Array2<f32> {
        let mut result = x.clone();
        
        // Apply row-wise softmax
        for mut row in result.axis_iter_mut(Axis(0)) {
            let max_val = row.fold(std::f32::NEG_INFINITY, |a, &b| a.max(b));
            row.mapv_inplace(|v| (v - max_val).exp());
            let sum: f32 = row.sum();
            row.mapv_inplace(|v| v / sum);
        }
        
        result
    }
    
    /// Forward pass through the network
    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        // First layer: input -> hidden
        let z1 = x.dot(&self.weights["w1"]) + &self.biases["b1"];
        let a1 = Self::relu(&z1);
        
        // Second layer: hidden -> output
        let z2 = a1.dot(&self.weights["w2"]) + &self.biases["b2"];
        let a2 = Self::softmax(&z2);
        
        // Cache activations for backward pass
        self.activations.insert("x".to_string(), x.clone());
        self.activations.insert("z1".to_string(), z1);
        self.activations.insert("a1".to_string(), a1);
        self.activations.insert("z2".to_string(), z2);
        
        a2
    }
    
    /// Compute loss and gradients
    pub fn backward(&mut self, y_true: &Array2<f32>) -> (f32, HashMap<String, Array2<f32>>, HashMap<String, Array1<f32>>) {
        let batch_size = y_true.shape()[0] as f32;
        let mut weight_grads = HashMap::new();
        let mut bias_grads = HashMap::new();
        
        // Get cached activations and compute forward pass again
        // (fixes mutable borrow issue)
        let x = self.activations["x"].clone();
        let a1 = self.activations["a1"].clone();
        let z1 = self.activations["z1"].clone();
        let a2 = self.forward(&x.clone()); 
        
        // Compute cross-entropy loss
        let epsilon = 1e-15;
        let clipped_output = a2.mapv(|v| v.max(epsilon).min(1.0 - epsilon));
        let losses = y_true * &clipped_output.mapv(|v| -v.ln());
        let loss = losses.sum() / batch_size;
        
        // Output layer gradients
        let delta2 = (&a2 - y_true) / batch_size;
        weight_grads.insert("w2".to_string(), a1.t().dot(&delta2));
        bias_grads.insert("b2".to_string(), delta2.sum_axis(Axis(0)));
        
        // Hidden layer gradients
        let delta1 = delta2.dot(&self.weights["w2"].t()) * Self::relu_prime(&z1);
        weight_grads.insert("w1".to_string(), x.t().dot(&delta1));
        bias_grads.insert("b1".to_string(), delta1.sum_axis(Axis(0)));
        
        (loss, weight_grads, bias_grads)
    }
    
    /// Convert weights to/from format compatible with optimizer
    fn weights_to_optimizer_format(&self) -> HashMap<String, Array1<f32>> {
        let mut flattened = HashMap::new();
        
        // Flatten 2D weight matrices to 1D for optimizer
        for (name, weight) in self.weights.iter() {
            let flat_weights = weight.clone().into_shape(weight.len()).unwrap();
            flattened.insert(format!("w_{}", name), flat_weights);
        }
        
        // Add biases
        for (name, bias) in self.biases.iter() {
            flattened.insert(format!("b_{}", name), bias.clone());
        }
        
        flattened
    }
    
    /// Update network parameters from optimizer format
    fn update_from_optimizer_format(&mut self, flattened: &HashMap<String, Array1<f32>>) {
        // Update weights
        for (name, weight) in self.weights.iter_mut() {
            let flat_key = format!("w_{}", name);
            let shape = weight.shape();
            *weight = flattened[&flat_key].clone().into_shape((shape[0], shape[1])).unwrap();
        }
        
        // Update biases
        for (name, bias) in self.biases.iter_mut() {
            let flat_key = format!("b_{}", name);
            *bias = flattened[&flat_key].clone();
        }
    }
    
    /// Convert gradients to optimizer format
    fn grads_to_optimizer_format(&self, weight_grads: HashMap<String, Array2<f32>>, bias_grads: HashMap<String, Array1<f32>>) -> HashMap<String, Array1<f32>> {
        let mut flattened = HashMap::new();
        
        // Flatten weight gradients
        for (name, grad) in weight_grads.iter() {
            let flat_grad = grad.clone().into_shape(grad.len()).unwrap();
            flattened.insert(format!("w_{}", name), flat_grad);
        }
        
        // Add bias gradients
        for (name, grad) in bias_grads.iter() {
            flattened.insert(format!("b_{}", name), grad.clone());
        }
        
        flattened
    }
    
    /// Compute accuracy on a batch of data
    pub fn accuracy(&mut self, x: &Array2<f32>, y: &Array2<f32>) -> f32 {
        let predictions = self.forward(x);
        
        let mut correct = 0;
        let total = predictions.shape()[0];
        
        for i in 0..total {
            // Find index of maximum value manually since argmax() isn't available
            let pred_row = predictions.slice(s![i, ..]);
            let mut pred_idx = 0;
            let mut max_val = pred_row[0];
            for j in 1..pred_row.len() {
                if pred_row[j] > max_val {
                    max_val = pred_row[j];
                    pred_idx = j;
                }
            }
            
            // Do the same for true labels
            let true_row = y.slice(s![i, ..]);
            let mut true_idx = 0;
            let mut max_val = true_row[0];
            for j in 1..true_row.len() {
                if true_row[j] > max_val {
                    max_val = true_row[j];
                    true_idx = j;
                }
            }
            
            if pred_idx == true_idx {
                correct += 1;
            }
        }
        
        correct as f32 / total as f32
    }
}

/// Custom struct to hold MNIST data
#[allow(dead_code)]
struct MnistData {
    train_images: Vec<u8>,
    train_labels: Vec<u8>,
    test_images: Vec<u8>,
    test_labels: Vec<u8>,
}

/// Prepare MNIST data for neural network
fn prepare_mnist_data() -> Result<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>), Box<dyn Error>> {
    println!("Loading MNIST dataset from existing files...");
    
    // Define path to the data directory
    let data_path = Path::new("data");
    
    // Check if the data directory exists
    if !data_path.exists() {
        return Err("Data directory not found. Please ensure the MNIST dataset is in the 'data' directory.".into());
    }
    
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
    
    // Parse the IDX format
    let train_data = parse_idx_images(&train_images)?;
    let test_data = parse_idx_images(&test_images)?;
    
    let train_labels = parse_idx_labels(&train_labels)?;
    let test_labels = parse_idx_labels(&test_labels)?;
    
    println!("MNIST dataset processed successfully");
    println!("  Training data: {} samples", train_data.shape()[0]);
    println!("  Test data: {} samples", test_data.shape()[0]);
    
    Ok((train_data, train_labels, test_data, test_labels))
}

/// Parse IDX format images
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

/// Train MLP with Standard Adam optimizer
fn train_with_standard_adam(
    train_data: &Array2<f32>,
    train_labels: &Array2<f32>,
    test_data: &Array2<f32>,
    test_labels: &Array2<f32>,
    batch_size: usize,
    epochs: usize,
    learning_rate: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn Error>> {
    println!("Training with Standard Adam optimizer...");
    
    // Initialize model
    let mut model = MnistMLP::new(784, 128, 10);
    let mut optimizer = StandardAdam::new(learning_rate, 0.9, 0.999, 1e-8);
    
    let num_batches = train_data.shape()[0] / batch_size;
    let mut train_losses = Vec::with_capacity(epochs);
    let mut train_accuracies = Vec::with_capacity(epochs);
    let mut test_accuracies = Vec::with_capacity(epochs);
    
    let progress_bar = ProgressBar::new((epochs * num_batches) as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-")
    );
    
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        
        // Shuffle training data for each epoch
        let indices: Vec<usize> = (0..train_data.shape()[0]).collect();
        let shuffled_indices = shuffle_indices(&indices);
        
        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (batch_idx + 1).min(train_data.shape()[0] / batch_size) * batch_size;
            
            // Get batch data using shuffled indices
            let batch_indices = &shuffled_indices[start..end];
            let x_batch = get_batch(train_data, batch_indices);
            let y_batch = get_batch(train_labels, batch_indices);
            
            // Forward and backward pass
            let (loss, weight_grads, bias_grads) = {
                model.forward(&x_batch);
                model.backward(&y_batch)
            };
            
            epoch_loss += loss;
            
            // Convert model parameters and gradients to optimizer format
            let mut params = model.weights_to_optimizer_format();
            let grads = model.grads_to_optimizer_format(weight_grads, bias_grads);
            
            // Update parameters with optimizer
            optimizer.step(&mut params, &grads);
            
            // Update model with optimized parameters
            model.update_from_optimizer_format(&params);
            
            progress_bar.inc(1);
        }
        
        // Compute training accuracy on a subset
        let train_acc = model.accuracy(
            &train_data.slice(s![0..1000, ..]).to_owned(),
            &train_labels.slice(s![0..1000, ..]).to_owned()
        );
        
        // Compute test accuracy
        let test_acc = model.accuracy(test_data, test_labels);
        
        let avg_loss = epoch_loss / num_batches as f32;
        train_losses.push(avg_loss);
        train_accuracies.push(train_acc);
        test_accuracies.push(test_acc);
        
        progress_bar.set_message(format!(
            "Epoch {}/{} - Loss: {:.4} - Train Acc: {:.4} - Test Acc: {:.4}",
            epoch + 1, epochs, avg_loss, train_acc, test_acc
        ));
    }
    
    progress_bar.finish_with_message("Training with Standard Adam completed!");
    
    Ok((train_losses, train_accuracies, test_accuracies))
}

/// Train MLP with Enhanced Adam optimizer
fn train_with_enhanced_adam(
    train_data: &Array2<f32>,
    train_labels: &Array2<f32>,
    test_data: &Array2<f32>,
    test_labels: &Array2<f32>,
    batch_size: usize,
    epochs: usize,
    learning_rate: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), Box<dyn Error>> {
    println!("Training with Enhanced Adam optimizer...");
    
    // Initialize model
    let mut model = MnistMLP::new(784, 128, 10);
    let mut optimizer = EnhancedAdam::new(learning_rate, 0.9, 0.999, 1e-8);
    
    let num_batches = train_data.shape()[0] / batch_size;
    let mut train_losses = Vec::with_capacity(epochs);
    let mut train_accuracies = Vec::with_capacity(epochs);
    let mut test_accuracies = Vec::with_capacity(epochs);
    
    let progress_bar = ProgressBar::new((epochs * num_batches) as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-")
    );
    
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        
        // Shuffle training data for each epoch
        let indices: Vec<usize> = (0..train_data.shape()[0]).collect();
        let shuffled_indices = shuffle_indices(&indices);
        
        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (batch_idx + 1).min(train_data.shape()[0] / batch_size) * batch_size;
            
            // Get batch data using shuffled indices
            let batch_indices = &shuffled_indices[start..end];
            let x_batch = get_batch(train_data, batch_indices);
            let y_batch = get_batch(train_labels, batch_indices);
            
            // Forward and backward pass
            let (loss, weight_grads, bias_grads) = {
                model.forward(&x_batch);
                model.backward(&y_batch)
            };
            
            epoch_loss += loss;
            
            // Convert model parameters and gradients to optimizer format
            let mut params = model.weights_to_optimizer_format();
            let grads = model.grads_to_optimizer_format(weight_grads, bias_grads);
            
            // Update parameters with optimizer
            optimizer.step(&mut params, &grads);
            
            // Update model with optimized parameters
            model.update_from_optimizer_format(&params);
            
            progress_bar.inc(1);
        }
        
        // Compute training accuracy on a subset
        let train_acc = model.accuracy(
            &train_data.slice(s![0..1000, ..]).to_owned(),
            &train_labels.slice(s![0..1000, ..]).to_owned()
        );
        
        // Compute test accuracy
        let test_acc = model.accuracy(test_data, test_labels);
        
        let avg_loss = epoch_loss / num_batches as f32;
        train_losses.push(avg_loss);
        train_accuracies.push(train_acc);
        test_accuracies.push(test_acc);
        
        progress_bar.set_message(format!(
            "Epoch {}/{} - Loss: {:.4} - Train Acc: {:.4} - Test Acc: {:.4}",
            epoch + 1, epochs, avg_loss, train_acc, test_acc
        ));
    }
    
    progress_bar.finish_with_message("Training with Enhanced Adam completed!");
    
    Ok((train_losses, train_accuracies, test_accuracies))
}

/// Helper function to shuffle array indices
fn shuffle_indices(indices: &[usize]) -> Vec<usize> {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let mut shuffled = indices.to_vec();
    shuffled.shuffle(&mut rng);
    shuffled
}

/// Helper function to get batch data by indices
fn get_batch(data: &Array2<f32>, indices: &[usize]) -> Array2<f32> {
    let batch_size = indices.len();
    let feature_dim = data.shape()[1];
    
    let mut batch = Array2::zeros((batch_size, feature_dim));
    
    for (i, &idx) in indices.iter().enumerate() {
        batch.slice_mut(s![i, ..]).assign(&data.slice(s![idx, ..]));
    }
    
    batch
}

/// Plot comparison of Adam optimizers on MNIST
fn plot_mnist_comparison(
    standard_train_losses: &[f32],
    standard_train_accs: &[f32],
    standard_test_accs: &[f32],
    enhanced_train_losses: &[f32],
    enhanced_train_accs: &[f32],
    enhanced_test_accs: &[f32],
) -> Result<(), Box<dyn Error>> {
    // Create loss comparison plot
    {
        let root = BitMapBackend::new("figures/mnist_loss_comparison.png", (1200, 800)).into_drawing_area();
        root.fill(&WHITE)?;
        
        let epochs = standard_train_losses.len();
        
        let max_loss = standard_train_losses.iter()
            .chain(enhanced_train_losses.iter())
            .fold(0.0f32, |a, &b| a.max(b));
        
        let min_loss = standard_train_losses.iter()
            .chain(enhanced_train_losses.iter())
            .fold(max_loss, |a, &b| a.min(b));
        
        // Add padding to prevent label overlap
        let y_range = max_loss - min_loss;
        let y_min = (min_loss - y_range * 0.05).max(0.0);
        let y_max = max_loss + y_range * 0.1;
        
        let mut chart = ChartBuilder::on(&root)
            .caption("MNIST Training Loss: Adam Optimizer Comparison", ("sans-serif", 26).into_font())
            .margin(40)
            .x_label_area_size(80)
            .y_label_area_size(120)
            .build_cartesian_2d(0..epochs, y_min..y_max)?;
        
        chart.configure_mesh()
             .x_desc("Epochs")
             .y_desc("Loss")
             .axis_desc_style(("sans-serif", 26))
             .label_style(("sans-serif", 26))
             .x_labels(6)
             .y_labels(6)
             .disable_mesh()
             .draw()?;
        
        // Plot standard Adam loss
        chart.draw_series(LineSeries::new(
            standard_train_losses.iter().enumerate().map(|(i, &v)| (i, v)),
            RED.stroke_width(3),
        ))?
        .label("Standard Adam")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
        
        // Plot enhanced Adam loss
        chart.draw_series(LineSeries::new(
            enhanced_train_losses.iter().enumerate().map(|(i, &v)| (i, v)),
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
        
        println!("MNIST loss comparison plot saved as 'mnist_loss_comparison.png'");
    }
    
    // Create accuracy comparison plot
    {
        let root = BitMapBackend::new("figures/mnist_accuracy_comparison.png", (1200, 800)).into_drawing_area();
        root.fill(&WHITE)?;
        
        let epochs = standard_test_accs.len();
        
        // Find proper y-axis range for accuracy
        let all_accs = standard_test_accs.iter()
            .chain(enhanced_test_accs.iter())
            .chain(standard_train_accs.iter())
            .chain(enhanced_train_accs.iter());
        
        let max_acc = all_accs.clone().fold(0.0f32, |a, &b| a.max(b)) as f64;
        let min_acc = all_accs.fold(1.0f32, |a, &b| a.min(b)) as f64;
        
        // Add padding to prevent label overlap
        let y_range = max_acc - min_acc;
        let y_min = (min_acc - y_range * 0.05).max(0.0);
        let y_max = (max_acc + y_range * 0.1).min(1.0);
        
        let mut chart = ChartBuilder::on(&root)
            .caption("MNIST Accuracy: Adam Optimizer Comparison", ("sans-serif", 26).into_font())
            .margin(40)
            .x_label_area_size(80)
            .y_label_area_size(120)
            .build_cartesian_2d(0..epochs, y_min..y_max)?;
        
        chart.configure_mesh()
             .x_desc("Epochs")
             .y_desc("Accuracy")
             .axis_desc_style(("sans-serif", 26))
             .label_style(("sans-serif", 26))
             .x_labels(6)
             .y_labels(6)
             .disable_mesh()
             .draw()?;
        
        // Plot standard Adam test accuracy
        chart.draw_series(LineSeries::new(
            standard_test_accs.iter().enumerate().map(|(i, &v)| (i, v as f64)),
            RED.stroke_width(3),
        ))?
        .label("Standard Adam (Test)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(3)));
        
        // Plot enhanced Adam test accuracy
        chart.draw_series(LineSeries::new(
            enhanced_test_accs.iter().enumerate().map(|(i, &v)| (i, v as f64)),
            BLUE.stroke_width(3),
        ))?
        .label("Enhanced Adam (Test)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(3)));
        
        // Plot standard Adam training accuracy (with lighter color to distinguish it)
        chart.draw_series(LineSeries::new(
            standard_train_accs.iter().enumerate().map(|(i, &v)| (i, v as f64)),
            RED.mix(0.5),
        ))?
        .label("Standard Adam (Train)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED.mix(0.5)));
        
        // Plot enhanced Adam training accuracy (with lighter color to distinguish it)
        chart.draw_series(LineSeries::new(
            enhanced_train_accs.iter().enumerate().map(|(i, &v)| (i, v as f64)),
            BLUE.mix(0.5),
        ))?
        .label("Enhanced Adam (Train)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE.mix(0.5)));
        
        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.9))
            .border_style(&BLACK)
            .label_font(("sans-serif", 26))
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
        
        println!("MNIST accuracy comparison plot saved as 'mnist_accuracy_comparison.png'");
    }
    
    Ok(())
}

/// Run the MNIST experiment comparing standard and enhanced Adam
pub fn run_mnist_test() -> Result<(), Box<dyn Error>> {
    println!("=== MNIST Neural Network Test ===");
    
    // Load and prepare MNIST dataset
    let (train_data, train_labels, test_data, test_labels) = prepare_mnist_data()?;
    
    // Training parameters
    let batch_size = 64;
    let epochs = 10;
    let learning_rate = 0.001;
    
    // Train with standard Adam
    println!("\nTraining with Standard Adam:");
    let (standard_train_losses, standard_train_accs, standard_test_accs) = 
        train_with_standard_adam(&train_data, &train_labels, &test_data, &test_labels, 
                                batch_size, epochs, learning_rate)?;
    
    // Train with enhanced Adam
    println!("\nTraining with Enhanced Adam:");
    let (enhanced_train_losses, enhanced_train_accs, enhanced_test_accs) = 
        train_with_enhanced_adam(&train_data, &train_labels, &test_data, &test_labels, 
                                batch_size, epochs, learning_rate)?;
    
    // Plot comparison results
    plot_mnist_comparison(
        &standard_train_losses,
        &standard_train_accs,
        &standard_test_accs,
        &enhanced_train_losses,
        &enhanced_train_accs,
        &enhanced_test_accs
    )?;
    
    // Print final results
    println!("\nFinal Results:");
    println!("Standard Adam - Final test accuracy: {:.4}", standard_test_accs.last().unwrap());
    println!("Enhanced Adam - Final test accuracy: {:.4}", enhanced_test_accs.last().unwrap());
    
    let accuracy_improvement = (enhanced_test_accs.last().unwrap() - standard_test_accs.last().unwrap()) * 100.0;
    println!("Accuracy improvement: {:.2}%", accuracy_improvement);
    
    Ok(())
}