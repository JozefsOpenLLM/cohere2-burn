use burn::{
    config::Config,
    module::Module,
    nn,
    tensor::{Device, Tensor},
};
use std::sync::Arc;


// CONFIG ==================================

#[derive(Config, Debug)]
pub struct MlpConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 4096,
        }
    }
}

impl MlpConfig {
    pub fn init(& self, device: & Device) -> MlpModule {
        let gate_proj: nn::Linear = nn::LinearConfig::new(self.hidden_size, self.intermediate_size).init(device);
        let up_prog: nn::Linear = nn::LinearConfig::new(self.hidden_size, self.intermediate_size).init(device);
        let down_prog: nn::Linear = nn::LinearConfig::new(self.intermediate_size, self.hidden_size).init(device);

        MlpModule { // Defined below
            gate_proj,
            up_prog,
            down_prog,
        }
    }
}


// MODULE ==================================

#[derive(Module, Debug)]
pub struct MlpModule {
    gate_proj: nn::Linear,
    up_proj: nn::Linear,
    down_prog: nn::Linear,
}

impl MlpModule {
    pub fn forward(& self, input: Tensor<2>) -> Tensor<2> {
        // apply the input to the two parallel gating and up layers, therefore clone to reuse
        let gate = self.gate_proj.forward(input.clone());
        let up = self.up_prog.forward(input);

        // Apply SiLU activation to up projection (gate * sigmoid(gate))
        let silu_up = silu(up);

        // Apply final linear transformation for the output
        self.down_prog(silu_up)
    }

}


// Helpers ===============================

/// Sigmoid Linear Unit (SiLU) on a matrix (i.e. R^2)
fn silu(A: Tensor<2>) -> Tensor<2> {
    let S = A.clone().sigmoid();
    A * S
}
