use burn::{
    config::Config,
    module::Module,
    nn,
    tensor::{Device, Tensor},
    prelude::Backend
};


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
    pub fn init<B: Backend>(&self, device: &Device<B>) -> MlpModule<B> {
        let gate_proj: nn::Linear<B> = nn::LinearConfig::new(self.hidden_size, self.intermediate_size).init(device);
        let up_proj: nn::Linear<B> = nn::LinearConfig::new(self.hidden_size, self.intermediate_size).init(device);
        let down_proj: nn::Linear<B> = nn::LinearConfig::new(self.intermediate_size, self.hidden_size).init(device);

        MlpModule { // Defined below
            gate_proj,
            up_proj,
            down_proj,
        }
    }
}


// MODULE ==================================

#[derive(Module, Debug)]
pub struct MlpModule<B: Backend> {
    gate_proj: nn::Linear<B>,
    up_proj: nn::Linear<B>,
    down_proj: nn::Linear<B>,
}

impl<B: Backend> MlpModule<B> {
    pub fn forward(& self, input: Tensor<B, 2>) -> Tensor<B, 2> {

        // apply the input to the two parallel gating and up layers, therefore clone to reuse input
        let gate = self.gate_proj.forward(input.clone());
        let up = self.up_proj.forward(input);

        // Apply SiLU activation to gate projection (gate * sigmoid(gate))
        let silu_gate = silu(gate);

        // apply hadamard product so the gate can actually gate off the standard up proj
        let comb = silu_gate * up;

        // Apply final linear transformation for the output
        self.down_proj.forward(comb)
    }

}


// Helpers ===============================

/// Sigmoid Linear Unit (SiLU) on a matrix (i.e. R^2)
fn silu<B: Backend>(a_matrix: Tensor<B, 2>) -> Tensor<B, 2> {
    let s_matrix = sigmoid::<B>(a_matrix.clone()); // sigmoid defined below
    a_matrix * s_matrix
}

/// sigmoid
fn sigmoid<B: Backend>(b_matrix: Tensor<B, 2>) -> Tensor<B, 2> {
    unimplemented!()
}

