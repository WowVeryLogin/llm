use blas_src as _;
use ndarray::{Array, Array1, Array2, Array3, Array4, ArrayRef1, Axis, Dimension, s};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt};
use rand_distr::{Distribution, weighted::WeightedIndex};
use rayon::prelude::*;
use std::{collections::HashMap, error::Error, fs};

const BATCH_SIZE: usize = 64;
const BLOCK_SIZE: usize = 256;
const MAX_ITERS: usize = 5000;
const EVAL_INTERVAL: usize = 500;
const EVAL_ITERS: usize = 200;
const LEARNING_RATE: f32 = 3e-4;
const N_EMBD: usize = 128;
const N_HEAD: usize = 4;
const N_LAYER: usize = 3;
const DROPOUT: f32 = 0.2;
const SEED: u64 = 1337;
const DROPOUT_CHUNK_SIZE: usize = 4096;

struct Vocabulary {
    stoi: HashMap<u8, usize>,
    itos: Vec<u8>,
}

impl Vocabulary {
    fn new(text: &[u8]) -> Self {
        let mut itos = text.to_vec();
        itos.sort_unstable();
        itos.dedup();

        let stoi = itos
            .iter()
            .copied()
            .enumerate()
            .map(|(i, ch)| (ch, i))
            .collect();

        Self { stoi, itos }
    }

    fn len(&self) -> usize {
        self.itos.len()
    }

    fn encode(&self, text: &[u8]) -> Vec<usize> {
        text.iter().map(|ch| self.stoi[ch]).collect()
    }

    fn decode(&self, tokens: &[usize]) -> Vec<u8> {
        tokens.iter().map(|&token| self.itos[token]).collect()
    }
}

struct AdamW {
    beta1: f32,
    beta2: f32,
    beta1_pow: f32,
    beta2_pow: f32,
    lr: f32,
    eps: f32,
    weight_decay: f32,
}

#[derive(Clone, Copy)]
struct AdamStep {
    beta1: f32,
    beta2: f32,
    beta1_correction: f32,
    beta2_correction: f32,
    lr: f32,
    eps: f32,
    weight_decay: f32,
}

impl AdamW {
    fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            beta1_pow: 1.0,
            beta2_pow: 1.0,
            lr,
            eps: 1e-8,
            weight_decay,
        }
    }

    fn next_step(&mut self) -> AdamStep {
        self.beta1_pow *= self.beta1;
        self.beta2_pow *= self.beta2;

        AdamStep {
            beta1: self.beta1,
            beta2: self.beta2,
            beta1_correction: 1.0 - self.beta1_pow,
            beta2_correction: 1.0 - self.beta2_pow,
            lr: self.lr,
            eps: self.eps,
            weight_decay: self.weight_decay,
        }
    }
}

struct Param2D {
    value: Array2<f32>,
    grad: Array2<f32>,
    m: Array2<f32>,
    v: Array2<f32>,
}

impl Param2D {
    fn rand<R: Rng + ?Sized>(rows: usize, cols: usize, scale: f32, rng: &mut R) -> Self {
        let value =
            Array2::from_shape_fn((rows, cols), |_| (rng.random::<f32>() * 2.0 - 1.0) * scale);
        Self::from_value(value)
    }

    fn from_value(value: Array2<f32>) -> Self {
        let dim = value.dim();
        Self {
            value,
            grad: Array2::zeros(dim),
            m: Array2::zeros(dim),
            v: Array2::zeros(dim),
        }
    }

    fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }

    fn step(&mut self, step: AdamStep) {
        self.m = step.beta1 * &self.m + (1.0 - step.beta1) * &self.grad;
        self.v = step.beta2 * &self.v + (1.0 - step.beta2) * &self.grad * &self.grad;

        let m_hat = &self.m / step.beta1_correction;
        let v_hat = &self.v / step.beta2_correction;
        let update = step.lr * m_hat / (v_hat.mapv(f32::sqrt) + step.eps);

        self.value *= 1.0 - step.lr * step.weight_decay;
        self.value -= &update;
    }
}

struct Param1D {
    value: Array1<f32>,
    grad: Array1<f32>,
    m: Array1<f32>,
    v: Array1<f32>,
}

impl Param1D {
    fn zeros(len: usize) -> Self {
        Self::from_value(Array1::zeros(len))
    }

    fn ones(len: usize) -> Self {
        Self::from_value(Array1::from_elem(len, 1.0))
    }

    fn from_value(value: Array1<f32>) -> Self {
        let len = value.len();
        Self {
            value,
            grad: Array1::zeros(len),
            m: Array1::zeros(len),
            v: Array1::zeros(len),
        }
    }

    fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }

    fn step(&mut self, step: AdamStep) {
        self.m = step.beta1 * &self.m + (1.0 - step.beta1) * &self.grad;
        self.v = step.beta2 * &self.v + (1.0 - step.beta2) * &self.grad * &self.grad;

        let m_hat = &self.m / step.beta1_correction;
        let v_hat = &self.v / step.beta2_correction;
        let update = step.lr * m_hat / (v_hat.mapv(f32::sqrt) + step.eps);

        self.value *= 1.0 - step.lr * step.weight_decay;
        self.value -= &update;
    }
}

struct Linear {
    weight: Param2D,
    bias: Option<Param1D>,
}

impl Linear {
    fn new<R: Rng + ?Sized>(
        in_features: usize,
        out_features: usize,
        bias: bool,
        rng: &mut R,
    ) -> Self {
        Self {
            weight: Param2D::rand(in_features, out_features, 0.02, rng),
            bias: bias.then(|| Param1D::zeros(out_features)),
        }
    }

    fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let (batch, time, _) = input.dim();
        let out_features = self.weight.value.dim().1;
        let mut output = Array3::zeros((batch, time, out_features));
        for (b, mut out) in output.outer_iter_mut().enumerate() {
            out.assign(&input.index_axis(Axis(0), b).dot(&self.weight.value));
            if let Some(bias) = &self.bias {
                for mut row in out.outer_iter_mut() {
                    row += &bias.value;
                }
            }
        }

        output
    }

    fn backward(&mut self, input: &Array3<f32>, grad_output: &Array3<f32>) -> Array3<f32> {
        let mut grad_input = Array3::zeros(input.dim());
        for (b, mut out) in grad_input.outer_iter_mut().enumerate() {
            let input_batch = input.slice(s![b, .., ..]);
            let grad_batch = grad_output.slice(s![b, .., ..]);
            self.weight.grad += &input_batch.t().dot(&grad_batch);
            if let Some(bias) = &mut self.bias {
                bias.grad += &grad_batch.sum_axis(Axis(0));
            }
            out.assign(&grad_batch.dot(&self.weight.value.t()));
        }

        grad_input
    }

    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        if let Some(bias) = &mut self.bias {
            bias.zero_grad();
        }
    }

    fn step(&mut self, step: AdamStep) {
        self.weight.step(step);
        if let Some(bias) = &mut self.bias {
            bias.step(step);
        }
    }
}

struct LayerNorm {
    gamma: Param1D,
    beta: Param1D,
    eps: f32,
}

impl LayerNorm {
    fn new(channels: usize) -> Self {
        Self {
            gamma: Param1D::ones(channels),
            beta: Param1D::zeros(channels),
            eps: 1e-5,
        }
    }

    fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        let (batch, time, channels) = input.dim();
        let mut output = Array3::zeros((batch, time, channels));

        for (b, mut out) in output.outer_iter_mut().enumerate() {
            for (t, mut out) in out.outer_iter_mut().enumerate() {
                let row = input.slice(s![b, t, ..]);
                let mean = row.sum() / channels as f32;
                let var = row.var(0.0);
                let inv_std = 1.0 / (var + self.eps).sqrt();
                let x_hat = row.mapv(|value| (value - mean) * inv_std);
                out.assign(&(&self.gamma.value * &x_hat + &self.beta.value));
            }
        }

        output
    }

    fn backward(&mut self, input: &Array3<f32>, grad_output: &Array3<f32>) -> Array3<f32> {
        let mut grad_input = Array3::zeros(input.dim());
        for (b, mut out) in grad_input.outer_iter_mut().enumerate() {
            for (t, mut out) in out.outer_iter_mut().enumerate() {
                let input_row = input.slice(s![b, t, ..]);
                let grad_row = grad_output.slice(s![b, t, ..]);

                let mean = input_row.mean().unwrap();
                let var = input_row.var(0.0);

                let inv_std = 1.0 / (var + self.eps).sqrt();

                let x_hat = input_row.mapv(|value| (value - mean) * inv_std);

                self.gamma.grad += &(&grad_row * &x_hat);
                self.beta.grad += &grad_row;

                let mut grad_x_hat = &grad_row * &self.gamma.value;
                let sum_grad_x_hat = grad_x_hat.sum();
                let sum_grad_x_hat_x_hat = (&grad_x_hat * &x_hat).sum();
                let channels = grad_x_hat.len();
                grad_x_hat = (&grad_x_hat * channels as f32
                    - sum_grad_x_hat
                    - &x_hat * sum_grad_x_hat_x_hat)
                    * (inv_std / channels as f32);
                out.assign(&grad_x_hat);
            }
        }

        grad_input
    }

    fn zero_grad(&mut self) {
        self.gamma.zero_grad();
        self.beta.zero_grad();
    }

    fn step(&mut self, step: AdamStep) {
        self.gamma.step(step);
        self.beta.step(step);
    }
}

struct AttentionCache {
    input: Array3<f32>,
    q: Array4<f32>,
    k: Array4<f32>,
    v: Array4<f32>,
    att_probs: Array4<f32>,
    att_dropped: Array4<f32>,
    att_dropout_mask: Option<Array4<f32>>,
    context: Array3<f32>,
    proj_dropout_mask: Option<Array3<f32>>,
}

struct MultiHeadSelfAttention {
    n_head: usize,
    head_size: usize,
    qkv: Linear,
    proj: Linear,
}

impl MultiHeadSelfAttention {
    fn new<R: Rng + ?Sized>(n_embd: usize, n_head: usize, rng: &mut R) -> Self {
        assert_eq!(n_embd % n_head, 0, "n_embd must be divisible by n_head");
        Self {
            n_head,
            head_size: n_embd / n_head,
            qkv: Linear::new(n_embd, 3 * n_embd, false, rng),
            proj: Linear::new(n_embd, n_embd, true, rng),
        }
    }

    fn forward<R: Rng + ?Sized>(
        &self,
        input: &Array3<f32>,
        training: bool,
        dropout_v: f32,
        rng: &mut R,
    ) -> (Array3<f32>, AttentionCache) {
        let (batch, time, n_embd) = input.dim();
        let qkv = self.qkv.forward(input);

        let mut q = Array4::zeros((batch, self.n_head, time, self.head_size));
        let mut k = Array4::zeros((batch, self.n_head, time, self.head_size));
        let mut v = Array4::zeros((batch, self.n_head, time, self.head_size));

        // let qkv = qkv
        //     .into_shape_with_order((batch, time, 3, self.n_head, self.head_size))
        //     .unwrap()
        //     .permuted_axes([0, 3, 1, 2, 4]);
        // let mut q = qkv.slice(s![.., .., .., 0, ..]).to_owned();
        // let mut k = qkv.slice(s![.., .., .., 1, ..]).to_owned();
        // let mut v = qkv.slice(s![.., .., .., 2, ..]).to_owned();

        for h in 0..self.n_head {
            let start = h * self.head_size;
            let end = start + self.head_size;
            q.slice_mut(s![.., h, .., ..])
                .assign(&qkv.slice(s![.., .., start..end]));
            k.slice_mut(s![.., h, .., ..]).assign(&qkv.slice(s![
                ..,
                ..,
                n_embd + start..n_embd + end
            ]));
            v.slice_mut(s![.., h, .., ..]).assign(&qkv.slice(s![
                ..,
                ..,
                2 * n_embd + start..2 * n_embd + end
            ]));
        }

        let scale = 1.0 / (self.head_size as f32).sqrt();
        let mut att_probs = Array4::zeros((batch, self.n_head, time, time));
        for (b, mut out) in att_probs.outer_iter_mut().enumerate() {
            for (h, mut out) in out.outer_iter_mut().enumerate() {
                let q_head = q.slice(s![b, h, .., ..]);
                let k_head = k.slice(s![b, h, .., ..]);
                out.assign(&q_head.dot(&k_head.t()));
                out *= scale;
                out.outer_iter_mut().enumerate().for_each(|(i, mut row)| {
                    row.slice_mut(s![i + 1..]).fill(f32::NEG_INFINITY);
                    softmax_slice(&mut row);
                });
            }
        }

        let (att_dropped, att_dropout_mask) = dropout(&att_probs, dropout_v, training, rng);
        let mut context = Array3::zeros((batch, time, n_embd));
        for (b, out) in att_dropped.outer_iter().enumerate() {
            for (h, _) in out.outer_iter().enumerate() {
                let start = h * self.head_size;
                let head_context =
                    att_dropped
                        .slice(s![b, h, .., ..])
                        .dot(&v.slice(s![b, h, .., ..]));
                let end = start + self.head_size;
                context
                    .slice_mut(s![b, .., start..end])
                    .assign(&head_context);
            }
        }

        let projected = self.proj.forward(&context);
        let (output, proj_dropout_mask) = dropout(&projected, dropout_v, training, rng);
        let cache = AttentionCache {
            input: input.clone(),
            q,
            k,
            v,
            att_probs,
            att_dropped,
            att_dropout_mask,
            context,
            proj_dropout_mask,
        };

        (output, cache)
    }

    fn backward(&mut self, cache: &AttentionCache, grad_output: &Array3<f32>) -> Array3<f32> {
        let (batch, time, n_embd) = cache.input.dim();
        let mut grad_projected = grad_output.clone();
        if let Some(mask) = &cache.proj_dropout_mask {
            grad_projected *= mask;
        }

        let grad_context = self.proj.backward(&cache.context, &grad_projected);
        let mut grad_q: Array4<f32> = Array4::zeros(cache.q.dim());
        let mut grad_k: Array4<f32> = Array4::zeros(cache.k.dim());
        let mut grad_v: Array4<f32> = Array4::zeros(cache.v.dim());
        let mut grad_att_dropped: Array4<f32> = Array4::zeros(cache.att_probs.dim());

        for (b, mut out) in grad_att_dropped.outer_iter_mut().enumerate() {
            for (h, mut out) in out.outer_iter_mut().enumerate() {
                let start = h * self.head_size;
                let end = start + self.head_size;
                let grad_context_head = grad_context.slice(s![b, .., start..end]);
                let v_head = cache.v.slice(s![b, h, .., ..]);
                let att_head = cache.att_dropped.slice(s![b, h, .., ..]);

                let grad_att_head = grad_context_head.dot(&v_head.t());
                let grad_v_head = att_head.t().dot(&grad_context_head);

                out.assign(&grad_att_head);
                grad_v.slice_mut(s![b, h, .., ..]).assign(&grad_v_head);
            }
        }

        let mut grad_att = grad_att_dropped;
        if let Some(mask) = &cache.att_dropout_mask {
            grad_att *= mask;
        }

        let scale = 1.0 / (self.head_size as f32).sqrt();
        let mut grad_scores = Array2::zeros((time, time));

        for (b, (mut q_out, mut k_out)) in grad_q
            .outer_iter_mut()
            .zip(grad_k.outer_iter_mut())
            .enumerate()
        {
            for (h, (mut q_out, mut k_out)) in q_out
                .outer_iter_mut()
                .zip(k_out.outer_iter_mut())
                .enumerate()
            {
                for i in 0..time {
                    let grad_att_row = grad_att.slice(s![b, h, i, ..]);
                    let att_probs_row = cache.att_probs.slice(s![b, h, i, ..]);
                    let weighted_grad = grad_att_row.dot(&att_probs_row);

                    for j in 0..=i {
                        grad_scores[[i, j]] = scale
                            * cache.att_probs[[b, h, i, j]]
                            * (grad_att[[b, h, i, j]] - weighted_grad);
                    }
                }

                let q_head = cache.q.slice(s![b, h, .., ..]);
                let k_head = cache.k.slice(s![b, h, .., ..]);
                let grad_q_head = grad_scores.dot(&k_head);
                let grad_k_head = grad_scores.t().dot(&q_head);
                q_out.assign(&grad_q_head);
                k_out.assign(&grad_k_head);
            }
        }

        let mut grad_qkv = Array3::zeros((batch, time, 3 * n_embd));
        for h in 0..self.n_head {
            let start = h * self.head_size;
            let end = start + self.head_size;
            grad_qkv
                .slice_mut(s![.., .., start..end])
                .assign(&grad_q.slice(s![.., h, .., ..]));
            grad_qkv
                .slice_mut(s![.., .., n_embd + start..n_embd + end])
                .assign(&grad_k.slice(s![.., h, .., ..]));
            grad_qkv
                .slice_mut(s![.., .., 2 * n_embd + start..2 * n_embd + end])
                .assign(&grad_v.slice(s![.., h, .., ..]));
        }

        self.qkv.backward(&cache.input, &grad_qkv)
    }

    fn zero_grad(&mut self) {
        self.qkv.zero_grad();
        self.proj.zero_grad();
    }

    fn step(&mut self, step: AdamStep) {
        self.qkv.step(step);
        self.proj.step(step);
    }
}

struct FeedForwardCache {
    input: Array3<f32>,
    hidden_pre: Array3<f32>,
    hidden: Array3<f32>,
    dropout_mask: Option<Array3<f32>>,
}

struct FeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl FeedForward {
    fn new<R: Rng + ?Sized>(n_embd: usize, rng: &mut R) -> Self {
        Self {
            fc1: Linear::new(n_embd, 4 * n_embd, true, rng),
            fc2: Linear::new(4 * n_embd, n_embd, true, rng),
        }
    }

    fn forward<R: Rng + ?Sized>(
        &self,
        input: &Array3<f32>,
        training: bool,
        dropout_v: f32,
        rng: &mut R,
    ) -> (Array3<f32>, FeedForwardCache) {
        let hidden_pre = self.fc1.forward(input);
        let hidden = hidden_pre.mapv(|x| x.max(0.0));
        let projected = self.fc2.forward(&hidden);
        let (output, dropout_mask) = dropout(&projected, dropout_v, training, rng);
        let cache = FeedForwardCache {
            input: input.clone(),
            hidden_pre,
            hidden,
            dropout_mask,
        };

        (output, cache)
    }

    fn backward(&mut self, cache: &FeedForwardCache, grad_output: &Array3<f32>) -> Array3<f32> {
        let mut grad_projected = grad_output.clone();
        if let Some(mask) = &cache.dropout_mask {
            grad_projected *= mask;
        }

        let mut grad_hidden = self.fc2.backward(&cache.hidden, &grad_projected);
        grad_hidden.zip_mut_with(&cache.hidden_pre, |grad, pre| {
            if *pre <= 0.0 {
                *grad = 0.0;
            }
        });
        self.fc1.backward(&cache.input, &grad_hidden)
    }

    fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
    }

    fn step(&mut self, step: AdamStep) {
        self.fc1.step(step);
        self.fc2.step(step);
    }
}

struct BlockCache {
    input: Array3<f32>,
    residual_after_attention: Array3<f32>,
    attention: AttentionCache,
    feed_forward: FeedForwardCache,
}

struct Block {
    ln1: LayerNorm,
    attention: MultiHeadSelfAttention,
    ln2: LayerNorm,
    feed_forward: FeedForward,
}

impl Block {
    fn new<R: Rng + ?Sized>(n_embd: usize, n_head: usize, rng: &mut R) -> Self {
        Self {
            ln1: LayerNorm::new(n_embd),
            attention: MultiHeadSelfAttention::new(n_embd, n_head, rng),
            ln2: LayerNorm::new(n_embd),
            feed_forward: FeedForward::new(n_embd, rng),
        }
    }

    fn forward<R: Rng + ?Sized>(
        &self,
        input: &Array3<f32>,
        training: bool,
        dropout: f32,
        rng: &mut R,
    ) -> (Array3<f32>, BlockCache) {
        let ln1_output = self.ln1.forward(input);
        let (attention_output, attention_cache) =
            self.attention.forward(&ln1_output, training, dropout, rng);
        let residual_after_attention = input + &attention_output;
        let ln2_output = self.ln2.forward(&residual_after_attention);
        let (feed_forward_output, feed_forward_cache) =
            self.feed_forward
                .forward(&ln2_output, training, dropout, rng);
        let output = &residual_after_attention + &feed_forward_output;

        let cache = BlockCache {
            input: input.clone(),
            residual_after_attention,
            attention: attention_cache,
            feed_forward: feed_forward_cache,
        };

        (output, cache)
    }

    fn backward(&mut self, cache: &BlockCache, grad_output: &Array3<f32>) -> Array3<f32> {
        let mut grad_residual_after_attention = grad_output.clone();
        let grad_feed_forward = self.feed_forward.backward(&cache.feed_forward, grad_output);
        let grad_ln2_input = self
            .ln2
            .backward(&cache.residual_after_attention, &grad_feed_forward);
        grad_residual_after_attention += &grad_ln2_input;

        let mut grad_input = grad_residual_after_attention.clone();
        let grad_attention = self
            .attention
            .backward(&cache.attention, &grad_residual_after_attention);
        let grad_ln1_input = self.ln1.backward(&cache.input, &grad_attention);
        grad_input += &grad_ln1_input;

        grad_input
    }

    fn zero_grad(&mut self) {
        self.ln1.zero_grad();
        self.attention.zero_grad();
        self.ln2.zero_grad();
        self.feed_forward.zero_grad();
    }

    fn step(&mut self, step: AdamStep) {
        self.ln1.step(step);
        self.attention.step(step);
        self.ln2.step(step);
        self.feed_forward.step(step);
    }
}

#[derive(Clone, Copy)]
struct GptConfig {
    vocab_size: usize,
    block_size: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    dropout: f32,
}

struct GptCache {
    idx: Array2<usize>,
    before_final_norm: Array3<f32>,
    final_norm_output: Array3<f32>,
    blocks: Vec<BlockCache>,
}

struct GptLanguageModel {
    config: GptConfig,
    token_embedding: Param2D,
    position_embedding: Param2D,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl GptLanguageModel {
    fn new<R: Rng + ?Sized>(config: GptConfig, rng: &mut R) -> Self {
        let blocks = (0..config.n_layer)
            .map(|_| Block::new(config.n_embd, config.n_head, rng))
            .collect();

        Self {
            config,
            token_embedding: Param2D::rand(config.vocab_size, config.n_embd, 0.02, rng),
            position_embedding: Param2D::rand(config.block_size, config.n_embd, 0.02, rng),
            blocks,
            ln_f: LayerNorm::new(config.n_embd),
            lm_head: Linear::new(config.n_embd, config.vocab_size, true, rng),
        }
    }

    fn forward<R: Rng + ?Sized>(
        &self,
        idx: &Array2<usize>,
        training: bool,
        rng: &mut R,
    ) -> (Array3<f32>, GptCache) {
        let (batch, time) = idx.dim();
        assert!(
            time <= self.config.block_size,
            "sequence length exceeds block size"
        );

        let mut x = Array3::zeros((batch, time, self.config.n_embd));
        for (b, mut out) in x.outer_iter_mut().enumerate() {
            for (t, mut out) in out.outer_iter_mut().enumerate() {
                let token = idx[[b, t]];
                out.assign(&self.token_embedding.value.row(token));
                out += &self.position_embedding.value.row(t);
            }
        }

        let mut block_caches = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (next_x, cache) = block.forward(&x, training, self.config.dropout, rng);
            block_caches.push(cache);
            x = next_x;
        }

        let before_final_norm = x;
        let final_norm_output = self.ln_f.forward(&before_final_norm);
        let logits = self.lm_head.forward(&final_norm_output);
        let cache = GptCache {
            idx: idx.clone(),
            before_final_norm,
            final_norm_output,
            blocks: block_caches,
        };

        (logits, cache)
    }

    fn loss<R: Rng + ?Sized>(
        &self,
        idx: &Array2<usize>,
        targets: &Array2<usize>,
        rng: &mut R,
    ) -> f32 {
        let (mut logits, _) = self.forward(idx, false, rng);
        let (loss, _) = cross_entropy_loss(&mut logits, targets);
        loss
    }

    fn train_batch<R: Rng + ?Sized>(
        &mut self,
        idx: &Array2<usize>,
        targets: &Array2<usize>,
        optimizer: &mut AdamW,
        rng: &mut R,
    ) -> f32 {
        self.zero_grad();
        let (mut logits, cache) = self.forward(idx, true, rng);
        let (loss, grad_logits) = cross_entropy_loss(&mut logits, targets);
        self.backward(&cache, &grad_logits);
        self.step(optimizer);
        loss
    }

    fn backward(&mut self, cache: &GptCache, grad_logits: &Array3<f32>) {
        let mut grad = self.lm_head.backward(&cache.final_norm_output, grad_logits);
        grad = self.ln_f.backward(&cache.before_final_norm, &grad);

        for (block, block_cache) in self.blocks.iter_mut().rev().zip(cache.blocks.iter().rev()) {
            grad = block.backward(block_cache, &grad);
        }

        for (b, grad_row) in grad.outer_iter().enumerate() {
            for (t, grad_row) in grad_row.outer_iter().enumerate() {
                let token = cache.idx[[b, t]];
                self.token_embedding
                    .grad
                    .row_mut(token)
                    .scaled_add(1.0, &grad_row);
                self.position_embedding
                    .grad
                    .row_mut(t)
                    .scaled_add(1.0, &grad_row);
            }
        }
    }

    fn generate<R: Rng + ?Sized>(
        &self,
        context: &[usize],
        max_new_tokens: usize,
        rng: &mut R,
    ) -> Vec<usize> {
        assert!(!context.is_empty(), "generation needs a non-empty context");
        let mut tokens = context.to_vec();

        for _ in 0..max_new_tokens {
            let start = tokens.len().saturating_sub(self.config.block_size);
            let window = &tokens[start..];
            let idx = Array2::from_shape_vec((1, window.len()), window.to_vec()).unwrap();
            let (mut logits, _) = self.forward(&idx, false, rng);
            let mut last_logits = logits.slice_mut(s![0, window.len() - 1, ..]);
            softmax_slice(&mut last_logits);
            let dist = WeightedIndex::new(last_logits.iter())
                .expect("softmax should produce a valid probability distribution");
            tokens.push(dist.sample(rng));
        }

        tokens
    }

    fn zero_grad(&mut self) {
        self.token_embedding.zero_grad();
        self.position_embedding.zero_grad();
        for block in &mut self.blocks {
            block.zero_grad();
        }
        self.ln_f.zero_grad();
        self.lm_head.zero_grad();
    }

    fn step(&mut self, optimizer: &mut AdamW) {
        let step = optimizer.next_step();
        self.token_embedding.step(step);
        self.position_embedding.step(step);
        for block in &mut self.blocks {
            block.step(step);
        }
        self.ln_f.step(step);
        self.lm_head.step(step);
    }
}

fn softmax_slice(logits: &mut ArrayRef1<f32>) {
    let max = logits.iter().cloned().reduce(f32::max).unwrap_or(0.);
    logits.mapv_inplace(|x| (x - max).exp());
    *logits /= logits.sum();
}

fn dropout_mask_values<R: Rng + ?Sized>(len: usize, keep: f32, rng: &mut R) -> Vec<f32> {
    let base_seed = rng.random::<u64>();
    let mut values = vec![0.0; len];

    values
        .par_chunks_mut(DROPOUT_CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let seed = base_seed ^ (chunk_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut local_rng = SmallRng::seed_from_u64(seed);
            for value in chunk {
                *value = if local_rng.random::<f32>() < keep {
                    1.0 / keep
                } else {
                    0.0
                };
            }
        });

    values
}

fn dropout<R: Rng + ?Sized, D: Dimension>(
    input: &Array<f32, D>,
    p: f32,
    training: bool,
    rng: &mut R,
) -> (Array<f32, D>, Option<Array<f32, D>>) {
    if !training || p == 0.0 {
        return (input.clone(), None);
    }

    assert!((0.0..1.0).contains(&p), "dropout must be in [0, 1)");
    let keep = 1.0 - p;
    let mask_values = dropout_mask_values(input.len(), keep, rng);
    let mask = Array::from_shape_vec(input.dim(), mask_values).unwrap();

    (input * &mask, Some(mask))
}

fn cross_entropy_loss(logits: &mut Array3<f32>, targets: &Array2<usize>) -> (f32, Array3<f32>) {
    let (batch, time, _) = logits.dim();
    assert_eq!(targets.dim(), (batch, time));

    let normalizer = (batch * time) as f32;
    let mut grad = Array3::zeros(logits.dim());

    let mut total_loss = 0.0;
    for (b, mut out) in grad.outer_iter_mut().enumerate() {
        for (h, mut out) in out.outer_iter_mut().enumerate() {
            let mut row = logits.slice_mut(s![b, h, ..]);
            softmax_slice(&mut row);
            let target = targets[[b, h]];
            let loss = -row[target].max(1e-12).ln();
            out.assign(&(&row / normalizer));
            out[target] -= 1.0 / normalizer;
            total_loss += loss;
        }
    }

    (total_loss / normalizer, grad)
}

fn get_batch<R: Rng + ?Sized>(
    data: &[usize],
    batch_size: usize,
    block_size: usize,
    rng: &mut R,
) -> (Array2<usize>, Array2<usize>) {
    assert!(
        data.len() > block_size,
        "dataset must be longer than the model block size"
    );

    let max_start = data.len() - block_size - 1;
    let mut input = Array2::zeros((batch_size, block_size));
    let mut targets = Array2::zeros((batch_size, block_size));

    for batch in 0..batch_size {
        let start = rng.random_range(0..=max_start);
        let input_slice = Array1::from_vec(data[start..start + block_size].to_vec());
        let target_slice = Array1::from_vec(data[start + 1..start + block_size + 1].to_vec());

        input.slice_mut(s![batch, ..]).assign(&input_slice);
        targets.slice_mut(s![batch, ..]).assign(&target_slice);
    }

    (input, targets)
}

fn estimate_loss<R: Rng + ?Sized>(
    model: &GptLanguageModel,
    train_data: &[usize],
    val_data: &[usize],
    rng: &mut R,
) -> (f32, f32) {
    let mut train_loss = 0.0;
    let mut val_loss = 0.0;

    for _ in 0..EVAL_ITERS {
        let (input, targets) = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, rng);
        train_loss += model.loss(&input, &targets, rng);

        let (input, targets) = get_batch(val_data, BATCH_SIZE, BLOCK_SIZE, rng);
        val_loss += model.loss(&input, &targets, rng);
    }

    (train_loss / EVAL_ITERS as f32, val_loss / EVAL_ITERS as f32)
}

fn main() -> Result<(), Box<dyn Error>> {
    let text = fs::read("set.txt")?;
    let vocab = Vocabulary::new(&text);
    let data = vocab.encode(&text);
    let split_at = data.len() * 9 / 10;
    let train_data = &data[..split_at];
    let val_data = &data[split_at..];
    let mut rng = SmallRng::seed_from_u64(SEED);
    let config = GptConfig {
        vocab_size: vocab.len(),
        block_size: BLOCK_SIZE,
        n_embd: N_EMBD,
        n_head: N_HEAD,
        n_layer: N_LAYER,
        dropout: DROPOUT,
    };
    let mut model = GptLanguageModel::new(config, &mut rng);
    let mut optimizer = AdamW::new(LEARNING_RATE, 0.01);

    println!(
        "{} characters, {} unique tokens, {} parameters",
        text.len(),
        vocab.len(),
        parameter_count(&model)
    );

    for iter in 0..MAX_ITERS {
        if iter % EVAL_INTERVAL == 0 || iter == MAX_ITERS - 1 {
            let (train_loss, val_loss) = estimate_loss(&model, train_data, val_data, &mut rng);
            println!("step {iter}: train loss {train_loss:.4}, val loss {val_loss:.4}");
        }

        let (input, targets) = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, &mut rng);
        model.train_batch(&input, &targets, &mut optimizer, &mut rng);
    }

    let generated = model.generate(&[0], 500, &mut rng);
    println!("{}", String::from_utf8_lossy(&vocab.decode(&generated)));

    Ok(())
}

fn parameter_count(model: &GptLanguageModel) -> usize {
    let mut total = model.token_embedding.value.len() + model.position_embedding.value.len();
    for block in &model.blocks {
        total += block.ln1.gamma.value.len() + block.ln1.beta.value.len();
        total += block.attention.qkv.weight.value.len();
        total += block.attention.proj.weight.value.len();
        total += block
            .attention
            .proj
            .bias
            .as_ref()
            .map_or(0, |bias| bias.value.len());
        total += block.ln2.gamma.value.len() + block.ln2.beta.value.len();
        total += block.feed_forward.fc1.weight.value.len();
        total += block
            .feed_forward
            .fc1
            .bias
            .as_ref()
            .map_or(0, |bias| bias.value.len());
        total += block.feed_forward.fc2.weight.value.len();
        total += block
            .feed_forward
            .fc2
            .bias
            .as_ref()
            .map_or(0, |bias| bias.value.len());
    }
    total += model.ln_f.gamma.value.len() + model.ln_f.beta.value.len();
    total += model.lm_head.weight.value.len();
    total += model
        .lm_head
        .bias
        .as_ref()
        .map_or(0, |bias| bias.value.len());
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_round_trips_bytes() {
        let text = b"banana";
        let vocab = Vocabulary::new(text);
        let encoded = vocab.encode(text);

        assert_eq!(vocab.decode(&encoded), text);
    }

    #[test]
    fn softmax_is_stable_and_normalized() {
        let mut probs = Array1::from_iter([1000.0, 1001.0, 999.0].into_iter());
        softmax_slice(&mut probs);
        let sum: f32 = probs.iter().sum();

        assert!(probs.iter().all(|p| p.is_finite() && *p > 0.0));
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dropout_is_disabled_during_eval_and_scaled_during_training() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let input = Array3::from_elem((2, 2, 2), 1.0);

        let (eval_output, eval_mask) = dropout(&input, 0.5, false, &mut rng);
        assert!(eval_mask.is_none());
        assert_eq!(eval_output, input);

        let (train_output, train_mask) = dropout(&input, 0.5, true, &mut rng);
        let train_mask = train_mask.unwrap();
        assert_eq!(train_output, &input * &train_mask);
        assert!(
            train_mask
                .iter()
                .all(|value| *value == 0.0 || (*value - 2.0).abs() < 1e-6)
        );
    }

    #[test]
    fn cross_entropy_gradients_sum_to_zero_per_token() {
        let mut logits = Array3::zeros((2, 3, 5));
        let targets = Array2::from_shape_vec((2, 3), vec![0, 1, 2, 3, 4, 0]).unwrap();
        let (loss, grad) = cross_entropy_loss(&mut logits, &targets);

        assert!((loss - (5.0f32).ln()).abs() < 1e-6);
        for b in 0..2 {
            for t in 0..3 {
                let row_sum: f32 = grad.slice(s![b, t, ..]).sum();
                assert!(row_sum.abs() < 1e-6);
            }
        }
    }

    #[test]
    fn gpt_forward_backward_and_generation_work() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let config = GptConfig {
            vocab_size: 7,
            block_size: 4,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            dropout: 0.2,
        };
        let mut model = GptLanguageModel::new(config, &mut rng);
        let mut optimizer = AdamW::new(1e-3, 0.0);
        let input = Array2::from_shape_vec((2, 4), vec![0, 1, 2, 3, 1, 2, 3, 4]).unwrap();
        let targets = Array2::from_shape_vec((2, 4), vec![1, 2, 3, 4, 2, 3, 4, 5]).unwrap();

        let loss = model.train_batch(&input, &targets, &mut optimizer, &mut rng);
        let generated = model.generate(&[0], 5, &mut rng);

        assert!(loss.is_finite());
        assert_eq!(generated.len(), 6);
        assert!(generated.iter().all(|token| *token < config.vocab_size));
    }
}
