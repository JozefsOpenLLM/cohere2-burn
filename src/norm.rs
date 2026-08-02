use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{Device, Tensor, TensorData, Shape, DType},
    prelude::Backend
};

// CONFIG ===================================

#[derive(Config, Debug)]
pub struct LayerNormConfig {
    // Small constant added for numerical stability to avoid division by zero
    pub variance_epsilon: f64,
    // will be used for creating the vector for learned scaling parameters
    pub layer_dim: usize,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        LayerNormConfig {
            variance_epsilon: 1e-6,
            layer_dim: 768
        }
    }
}

impl LayerNormConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> LayerNorm<B> {
        // set to no scaling by default (pre-training)
        let learned_scale = Param::from_tensor(
            Tensor::<B, 1>::ones(Shape::new([self.layer_dim]), device)
        );
        LayerNorm { variance_epsilon: self.variance_epsilon, learned_scale }
    }
    // load in data from open weights
    pub fn from_pretrained<B: Backend>(&self, pretrained_scale_tensor: TensorData, device: &Device<B>) -> LayerNorm<B> {
        let learned_scale = Param::from_tensor(
            Tensor::<B,1>::from_data(pretrained_scale_tensor, device)
        );
        LayerNorm { variance_epsilon: self.variance_epsilon, learned_scale }
    }
}

// MODULE ===============================================

#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    // wrap the vector in Param in order to have it editable during training, and loaded during inference
    // this parameter is also called "gamma" sometimes; also "weight" in cohere documentation
    variance_epsilon: f64,
    learned_scale: Param<Tensor<B, 1>>,
}

impl<B: Backend> LayerNorm<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = x.dtype();
        // this type conversion is done in the original model's implementation
        let mut x_f32 = x.clone().cast(DType::F32);
        let mean = x_f32.clone().mean_dim(D - 1);
        let variance = ( x_f32.clone() - mean.clone() ).powf_scalar(2.0).mean_dim(D-1);
        // normalize
        x_f32 = (x_f32 - mean) * (variance + self.variance_epsilon).sqrt().recip();
        // apply the learned scaling parameter
        let learned_scale_f32: Tensor<B, D> = self.learned_scale.val().cast(DType::F32).unsqueeze();
        x_f32 = x_f32 * learned_scale_f32;
        x_f32.cast(dtype)
    }
}
