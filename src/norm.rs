use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{Device, Tensor},
};

// CONFIG ===================================

#[derive(Config, Debug)]
pub struct LayerNormConfig {
    // Small constant added for numerical stability to avoid division by zero
    pub variance_epsilon: f64,
    // dlsp = acronym for dimensions_to_be_used_for_learned_scaling_params
    // (hidden_size, head_dimension)
    // head_dimension is optional, used for QKNorm to normalize across head dimension
    // head_dimension = how many dimensions belong to each attention head
    // These will be used for creating matrices for learned scaling parameters
    pub dlsp: (i64, Option<i64>),
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        LayerNormConfig {
            variance_epsilon: 1e-6,
            dlsp: (768, None)
        }
    }
}
