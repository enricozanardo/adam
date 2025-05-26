use std::collections::HashMap;
use ndarray::Array1;

/// SGD with Momentum optimizer
pub struct SGDMomentum {
    lr: f32,
    momentum: f32,
    velocity: HashMap<String, Array1<f32>>,
}

impl SGDMomentum {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            velocity: HashMap::new(),
        }
    }
    
    pub fn step(&mut self, params: &mut HashMap<String, Array1<f32>>, grads: &HashMap<String, Array1<f32>>) {
        for (name, grad) in grads.iter() {
            let param = params.get_mut(name).unwrap();
            
            // Get or initialize velocity
            let v = self.velocity.entry(name.clone()).or_insert_with(|| Array1::zeros(param.len()));
            
            // Update velocity
            *v = self.momentum * &*v - self.lr * grad;
            
            // Update parameters
            *param = &*param + &*v;
        }
    }
}

/// RMSProp optimizer
pub struct RMSProp {
    lr: f32,
    decay_rate: f32,
    epsilon: f32,
    square_grads: HashMap<String, Array1<f32>>,
}

impl RMSProp {
    pub fn new(lr: f32, decay_rate: f32, epsilon: f32) -> Self {
        Self {
            lr,
            decay_rate,
            epsilon,
            square_grads: HashMap::new(),
        }
    }
    
    pub fn step(&mut self, params: &mut HashMap<String, Array1<f32>>, grads: &HashMap<String, Array1<f32>>) {
        for (name, grad) in grads.iter() {
            let param = params.get_mut(name).unwrap();
            
            // Get or initialize square_grads
            let square_grad = self.square_grads
                .entry(name.clone())
                .or_insert_with(|| Array1::zeros(param.len()));
            
            // Update running average of squared gradients
            *square_grad = self.decay_rate * &*square_grad + (1.0 - self.decay_rate) * (grad * grad);
            
            // Update parameters
            let learning_rates = self.lr / (square_grad.mapv(f32::sqrt) + self.epsilon);
            *param = &*param - &(learning_rates * grad);
        }
    }
}

/// AdaGrad optimizer
pub struct AdaGrad {
    lr: f32,
    epsilon: f32,
    square_grads: HashMap<String, Array1<f32>>,
}

impl AdaGrad {
    pub fn new(lr: f32, epsilon: f32) -> Self {
        Self {
            lr,
            epsilon,
            square_grads: HashMap::new(),
        }
    }
    
    pub fn step(&mut self, params: &mut HashMap<String, Array1<f32>>, grads: &HashMap<String, Array1<f32>>) {
        for (name, grad) in grads.iter() {
            let param = params.get_mut(name).unwrap();
            
            // Get or initialize square_grads
            let square_grad = self.square_grads
                .entry(name.clone())
                .or_insert_with(|| Array1::zeros(param.len()));
            
            // Accumulate squared gradients
            *square_grad = &*square_grad + (grad * grad);
            
            // Update parameters
            let learning_rates = self.lr / (square_grad.mapv(f32::sqrt) + self.epsilon);
            *param = &*param - &(learning_rates * grad);
        }
    }
} 