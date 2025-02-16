use std::slice;

use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{Dtype, SafeTensors};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        let get_tensor = |name: &str| {
            let tensor = safetensor.tensor(name).unwrap();
            match tensor.dtype() {
                Dtype::F32 => {
                    let data = unsafe {
                        let data = tensor.data();
                        assert!(data.len() == tensor.shape().iter().product::<usize>() * 4);
                        slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                    };
                    Tensor::new(data.to_vec(), &tensor.shape().to_vec())
                }
                _ => todo!(),
            }
        };

        let mut params = LLamaParams {
            embedding_table: if config.tie_word_embeddings {
                get_tensor("lm_head.weight")
            } else {
                get_tensor("model.embed_tokens.weight")
            },
            rms_att_w: vec![],
            wq: vec![],
            wk: vec![],
            wv: vec![],
            wo: vec![],
            rms_ffn_w: vec![],
            w_up: vec![],
            w_gate: vec![],
            w_down: vec![],
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        };

        for i in 0..config.num_hidden_layers {
            params.rms_att_w.push(get_tensor(&format!(
                "model.layers.{}.input_layernorm.weight",
                i
            )));
            params.wq.push(get_tensor(&format!(
                "model.layers.{}.self_attn.q_proj.weight",
                i
            )));
            params.wk.push(get_tensor(&format!(
                "model.layers.{}.self_attn.k_proj.weight",
                i
            )));
            params.wv.push(get_tensor(&format!(
                "model.layers.{}.self_attn.v_proj.weight",
                i
            )));
            params.wo.push(get_tensor(&format!(
                "model.layers.{}.self_attn.o_proj.weight",
                i
            )));
            params.rms_ffn_w.push(get_tensor(&format!(
                "model.layers.{}.post_attention_layernorm.weight",
                i
            )));
            params.w_up.push(get_tensor(&format!(
                "model.layers.{}.mlp.up_proj.weight",
                i
            )));
            params.w_gate.push(get_tensor(&format!(
                "model.layers.{}.mlp.gate_proj.weight",
                i
            )));
            params.w_down.push(get_tensor(&format!(
                "model.layers.{}.mlp.down_proj.weight",
                i
            )));
        }

        params
    }
}
