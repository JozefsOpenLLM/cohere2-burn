// Original file content, now rewritten to use Module type
use burn::{
    config::Config,
    module::Module,
    nn,
    tensor::{Device, Tensor},
};
use std::sync::Arc;

/// Llama-style MLP feedforward layer configuration
#[derive(Config, Debug)]
pub struct LlamaMlpConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl Default for LlamaMlpConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 4096,
        }
    }
}

/// Module implementation of Llama-style MLP feedforward layer
#[derive(Module, Debug)]
pub struct LlamaMlpModule {
    gate_proj: nn::Linear,
    up_proj: nn::Linear,
    down_prog: nn::Linear,
}

impl LlamaMlpModule {
    /// Forward pass through the LLM (Large Language Model)
    pub fn forward(& self, input: Tensor<2>) -> Tensor<2> {
        let gate = self.gate_proj.forward(input.clone());
        let up = self.up_prog.forward(input);

        // Apply SiLU activation to up projection (gate * sigmoid(gate))
        let silu_up = Self::silu(up);

        // Apply final linear transformation for the output
        self.down_prog(silu_up)
    }

    fn silu(x: Tensor<2>) -> Tensor<2> {
        let s = x.clone().sigmoid();
        x * s
    }
}

impl LlamaMlpConfig {
    /// Initialize the Llama MLP module with parameters
    pub fn init(& self, device: & Device) -> LlamaMlpModule {
        let gate_proj = nn::LinearConfig::new(self.hidden_size, self.intermediate_size).init(device);
        let up_prog = nn::LinearConfig::new(self.hidden_size, self.intermediate_size).init(device);
        let down_prog = nn::LinearConfig::new(self.intermediate_size, self.hidden_size).init(device);

        LlamaMlpModule {
            gate_proj,
            up_prog,
            down_prog,
        }
    }
}

// These legacy functions are preserved for backward compatibility but simplified
fn dot(g: &[f32], v: &[f32]) -> f32 {
    g.iter().zip(v.iter()).map(|(g_val, v_val)| *g_val * *v_val).sum()
}

fn relu(f: &mut [f32]) -> Vec<f32> {
    f.iter().map(|x| if *x > 0.0 { *x } else { 0.0 }).collect()
}

fn sgn(f: &mut [f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(f.len());
    for val in f {
        if *val > 0.0 {
            out.push(1.0);
        } else if *val < 0.0 {
            out.push(-1.0);
        } else {
            out.push(0.0);
        }
    }
    out
}

fn hard_sigmoid(f: &mut [f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(f.len());
    for val in f.iter() {
        out.push(val.exp());
    }
    out
}

fn gelu_f64(f: &mut [f32]) ->collection::Vec<f32> {
    f.iter()
        .map(|v| {
            if v.abs() > 7.1e-8 {
                let abs_val = v.abs();
                let gain = 0.577215664904 * (0.044715 + abs_val);
                let tanh_gain = f32::tanh(*v * gain);
                tanh_gain * (0.044715 + abs_val)
            } else {
                f32::from(*v as f64)
            }
        })
        .collect()
}

fn sigmoid_cpu(f: &mut [f32]) ->collection::Vec<f32> {
    if f.iter().any(|val| val.abs() <= 3.4e-8) {
        1.0
    } else {
        let mut out = Vec::with_capacity(f.len());
        for val in f.iter() {
            out.push(1.0 / (1.0 + (*val).exp()));
        }
        out
    }
}