use blas_src as _;
use ndarray::{
    Array, Array1, Array2, Array3, ArrayRef1, ArrayView2, ArrayViewMut2, Axis, Dimension, Zip, s,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt};
use rand_distr::{Distribution, weighted::WeightedIndex};
use rayon::prelude::*;
use std::{collections::HashMap, error::Error, fs};

const BATCH_SIZE: usize = 64;
const BLOCK_SIZE: usize = 256;
const MAX_ITERS: usize = 2000;
const EVAL_INTERVAL: usize = 500;
const EVAL_ITERS: usize = 200;
const LEARNING_RATE: f32 = 3e-4;
const N_EMBD: usize = 128;
const N_HEAD: usize = 4;
const N_LAYER: usize = 2;
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

impl AdamStep {
    fn without_weight_decay(self) -> Self {
        Self {
            weight_decay: 0.0,
            ..self
        }
    }
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

    fn step_without_weight_decay(&mut self, step: AdamStep) {
        self.step(step.without_weight_decay());
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
        let step = step.without_weight_decay();
        self.m = step.beta1 * &self.m + (1.0 - step.beta1) * &self.grad;
        self.v = step.beta2 * &self.v + (1.0 - step.beta2) * &self.grad * &self.grad;

        let m_hat = &self.m / step.beta1_correction;
        let v_hat = &self.v / step.beta2_correction;
        let update = step.lr * m_hat / (v_hat.mapv(f32::sqrt) + step.eps);

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
        let (batch, time, in_features) = input.dim();
        let out_features = self.weight.value.dim().1;
        let input_flat = ArrayView2::from_shape(
            (batch * time, in_features),
            input
                .as_slice_memory_order()
                .expect("linear input should be contiguous"),
        )
        .unwrap();
        let mut output_flat = input_flat.dot(&self.weight.value);
        if let Some(bias) = &self.bias {
            for mut row in output_flat.outer_iter_mut() {
                row += &bias.value;
            }
        }

        let (output, offset) = output_flat.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        Array3::from_shape_vec((batch, time, out_features), output).unwrap()
    }

    fn backward(&mut self, input: &Array3<f32>, grad_output: &Array3<f32>) -> Array3<f32> {
        let (batch, time, in_features) = input.dim();
        let out_features = self.weight.value.dim().1;
        let input_flat = ArrayView2::from_shape(
            (batch * time, in_features),
            input
                .as_slice_memory_order()
                .expect("linear input should be contiguous"),
        )
        .unwrap();
        let grad_output_flat = ArrayView2::from_shape(
            (batch * time, out_features),
            grad_output
                .as_slice_memory_order()
                .expect("linear gradient output should be contiguous"),
        )
        .unwrap();

        self.weight.grad += &input_flat.t().dot(&grad_output_flat);
        if let Some(bias) = &mut self.bias {
            bias.grad += &grad_output_flat.sum_axis(Axis(0));
        }
        let grad_input_flat = grad_output_flat.dot(&self.weight.value.t());

        let (grad_input, offset) = grad_input_flat.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        Array3::from_shape_vec((batch, time, in_features), grad_input).unwrap()
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

struct LayerNormCache {
    x_hat: Array3<f32>,
    inv_std: Array2<f32>,
}

impl LayerNorm {
    fn new(channels: usize) -> Self {
        Self {
            gamma: Param1D::ones(channels),
            beta: Param1D::zeros(channels),
            eps: 1e-5,
        }
    }

    fn forward(&self, input: &Array3<f32>) -> (Array3<f32>, LayerNormCache) {
        let (batch, time, channels) = input.dim();
        let mut output = Array3::zeros((batch, time, channels));
        let mut x_hat_cache = Array3::zeros((batch, time, channels));
        let mut inv_std_cache = Array2::zeros((batch, time));

        for (b, mut out) in output.outer_iter_mut().enumerate() {
            for (t, mut out) in out.outer_iter_mut().enumerate() {
                let row = input.slice(s![b, t, ..]);
                let mean = row.sum() / channels as f32;
                let var = row
                    .iter()
                    .map(|value| {
                        let centered = value - mean;
                        centered * centered
                    })
                    .sum::<f32>()
                    / channels as f32;
                let inv_std = 1.0 / (var + self.eps).sqrt();
                let x_hat = row.mapv(|value| (value - mean) * inv_std);
                out.assign(&(&self.gamma.value * &x_hat + &self.beta.value));
                x_hat_cache.slice_mut(s![b, t, ..]).assign(&x_hat);
                inv_std_cache[[b, t]] = inv_std;
            }
        }

        (
            output,
            LayerNormCache {
                x_hat: x_hat_cache,
                inv_std: inv_std_cache,
            },
        )
    }

    fn backward(&mut self, cache: &LayerNormCache, grad_output: &Array3<f32>) -> Array3<f32> {
        let mut grad_input = Array3::zeros(grad_output.dim());
        for (b, mut out) in grad_input.outer_iter_mut().enumerate() {
            for (t, mut out) in out.outer_iter_mut().enumerate() {
                let grad_row = grad_output.slice(s![b, t, ..]);
                let x_hat = cache.x_hat.slice(s![b, t, ..]);
                let inv_std = cache.inv_std[[b, t]];

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
    q: Array3<f32>,
    k: Array3<f32>,
    v: Array3<f32>,
    att_probs: Array3<f32>,
    att_dropped: Array3<f32>,
    att_dropout_mask: Option<Array3<f32>>,
    context: Array3<f32>,
    proj_dropout_mask: Option<Array3<f32>>,
}

struct MultiHeadSelfAttention {
    n_head: usize,
    head_size: usize,
    qkv: Linear,
    proj: Linear,
    causal_mask: Array2<bool>,
}

impl MultiHeadSelfAttention {
    fn new<R: Rng + ?Sized>(n_embd: usize, n_head: usize, block_size: usize, rng: &mut R) -> Self {
        assert_eq!(n_embd % n_head, 0, "n_embd must be divisible by n_head");
        let mut causal_mask = Array2::from_elem((block_size, block_size), false);
        for i in 0..block_size {
            if i + 1 < block_size {
                causal_mask.slice_mut(s![i, i + 1..]).fill(true);
            }
        }

        Self {
            n_head,
            head_size: n_embd / n_head,
            qkv: Linear::new(n_embd, 3 * n_embd, false, rng),
            proj: Linear::new(n_embd, n_embd, true, rng),
            causal_mask,
        }
    }

    fn pack_qkv(&self, qkv: &Array3<f32>) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
        let (batch, time, qkv_channels) = qkv.dim();
        let n_embd = self.n_head * self.head_size;
        assert_eq!(qkv_channels, 3 * n_embd);

        let heads = batch * self.n_head;
        let mut q = Array3::zeros((heads, time, self.head_size));
        let mut k = Array3::zeros((heads, time, self.head_size));
        let mut v = Array3::zeros((heads, time, self.head_size));
        let head_values = time * self.head_size;

        q.as_slice_memory_order_mut()
            .expect("packed q should be contiguous")
            .par_chunks_mut(head_values)
            .zip(
                k.as_slice_memory_order_mut()
                    .expect("packed k should be contiguous")
                    .par_chunks_mut(head_values),
            )
            .zip(
                v.as_slice_memory_order_mut()
                    .expect("packed v should be contiguous")
                    .par_chunks_mut(head_values),
            )
            .enumerate()
            .for_each(|(head_idx, ((q_out, k_out), v_out))| {
                let b = head_idx / self.n_head;
                let h = head_idx % self.n_head;
                let start = h * self.head_size;
                let end = start + self.head_size;
                let mut q_out = ArrayViewMut2::from_shape((time, self.head_size), q_out).unwrap();
                let mut k_out = ArrayViewMut2::from_shape((time, self.head_size), k_out).unwrap();
                let mut v_out = ArrayViewMut2::from_shape((time, self.head_size), v_out).unwrap();

                q_out.assign(&qkv.slice(s![b, .., start..end]));
                k_out.assign(&qkv.slice(s![b, .., n_embd + start..n_embd + end]));
                v_out.assign(&qkv.slice(s![b, .., 2 * n_embd + start..2 * n_embd + end]));
            });

        (q, k, v)
    }

    fn unpack_qkv_grads(
        &self,
        grad_q: &Array3<f32>,
        grad_k: &Array3<f32>,
        grad_v: &Array3<f32>,
        batch: usize,
        time: usize,
    ) -> Array3<f32> {
        let n_embd = self.n_head * self.head_size;
        let mut grad_qkv = Array3::zeros((batch, time, 3 * n_embd));
        let batch_values = time * 3 * n_embd;

        grad_qkv
            .as_slice_memory_order_mut()
            .expect("qkv gradients should be contiguous")
            .par_chunks_mut(batch_values)
            .enumerate()
            .for_each(|(b, out)| {
                let mut out = ArrayViewMut2::from_shape((time, 3 * n_embd), out).unwrap();
                for h in 0..self.n_head {
                    let head_idx = b * self.n_head + h;
                    let start = h * self.head_size;
                    let end = start + self.head_size;
                    out.slice_mut(s![.., start..end])
                        .assign(&grad_q.index_axis(Axis(0), head_idx));
                    out.slice_mut(s![.., n_embd + start..n_embd + end])
                        .assign(&grad_k.index_axis(Axis(0), head_idx));
                    out.slice_mut(s![.., 2 * n_embd + start..2 * n_embd + end])
                        .assign(&grad_v.index_axis(Axis(0), head_idx));
                }
            });

        grad_qkv
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
        let (q, k, v) = self.pack_qkv(&qkv);
        drop(qkv);

        let scale = 1.0 / (self.head_size as f32).sqrt();
        let heads = batch * self.n_head;
        let mut att_probs = Array3::zeros((heads, time, time));
        let mask = self.causal_mask.slice(s![..time, ..time]);
        let head_score_size = time * time;
        att_probs
            .as_slice_memory_order_mut()
            .expect("attention probabilities should be contiguous")
            .par_chunks_mut(head_score_size)
            .enumerate()
            .for_each(|(head_idx, out)| {
                let q_head = q.index_axis(Axis(0), head_idx);
                let k_head = k.index_axis(Axis(0), head_idx);
                let mut out = ArrayViewMut2::from_shape((time, time), out).unwrap();
                out.assign(&q_head.dot(&k_head.t()));
                out *= scale;
                Zip::from(&mut out).and(&mask).for_each(|score, masked| {
                    if *masked {
                        *score = f32::NEG_INFINITY;
                    }
                });
                out.outer_iter_mut().for_each(|mut row| {
                    softmax_slice(&mut row);
                });
            });

        let (att_dropped, att_dropout_mask) = dropout(&att_probs, dropout_v, training, rng);
        let mut context = Array3::zeros((batch, time, n_embd));
        let batch_context_size = time * n_embd;
        context
            .as_slice_memory_order_mut()
            .expect("attention context should be contiguous")
            .par_chunks_mut(batch_context_size)
            .enumerate()
            .for_each(|(b, out)| {
                let mut out = ArrayViewMut2::from_shape((time, n_embd), out).unwrap();
                for h in 0..self.n_head {
                    let head_idx = b * self.n_head + h;
                    let start = h * self.head_size;
                    let end = start + self.head_size;
                    let att_head = att_dropped.index_axis(Axis(0), head_idx);
                    let v_head = v.index_axis(Axis(0), head_idx);
                    let head_context = att_head.dot(&v_head);
                    out.slice_mut(s![.., start..end]).assign(&head_context);
                }
            });

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
        let (batch, time, _) = cache.input.dim();
        let mut grad_projected = grad_output.clone();
        if let Some(mask) = &cache.proj_dropout_mask {
            grad_projected *= mask;
        }

        let grad_context = self.proj.backward(&cache.context, &grad_projected);
        let heads = batch * self.n_head;
        let mut grad_q = Array3::zeros((heads, time, self.head_size));
        let mut grad_k = Array3::zeros((heads, time, self.head_size));
        let mut grad_v = Array3::zeros((heads, time, self.head_size));
        let mut grad_att_dropped = Array3::zeros(cache.att_probs.dim());
        let head_score_size = time * time;
        let head_values = time * self.head_size;

        grad_att_dropped
            .as_slice_memory_order_mut()
            .expect("attention gradient should be contiguous")
            .par_chunks_mut(head_score_size)
            .zip(
                grad_v
                    .as_slice_memory_order_mut()
                    .expect("value gradient should be contiguous")
                    .par_chunks_mut(head_values),
            )
            .enumerate()
            .for_each(|(head_idx, (att_out, v_out))| {
                let b = head_idx / self.n_head;
                let h = head_idx % self.n_head;
                let start = h * self.head_size;
                let end = start + self.head_size;
                let grad_context_head = grad_context.slice(s![b, .., start..end]);
                let v_head = cache.v.index_axis(Axis(0), head_idx);
                let att_head = cache.att_dropped.index_axis(Axis(0), head_idx);

                let grad_att_head = grad_context_head.dot(&v_head.t());
                let grad_v_head = att_head.t().dot(&grad_context_head);

                ArrayViewMut2::from_shape((time, time), att_out)
                    .unwrap()
                    .assign(&grad_att_head);
                ArrayViewMut2::from_shape((time, self.head_size), v_out)
                    .unwrap()
                    .assign(&grad_v_head);
            });

        let mut grad_att = grad_att_dropped;
        if let Some(mask) = &cache.att_dropout_mask {
            grad_att *= mask;
        }

        let scale = 1.0 / (self.head_size as f32).sqrt();
        let mut grad_scores = Array2::zeros((time, time));

        for head_idx in 0..heads {
            grad_scores.fill(0.0);
            for i in 0..time {
                let p = cache.att_probs.slice(s![head_idx, i, ..]);
                let g = grad_att.slice(s![head_idx, i, ..]);
                let dot = g.dot(&p);
                let mut row = grad_scores.slice_mut(s![i, ..]);
                row.assign(&(&p * &(g.to_owned() - dot)));
                row *= scale;
                if i + 1 < time {
                    row.slice_mut(s![i + 1..]).fill(0.0);
                }
            }

            let q_head = cache.q.index_axis(Axis(0), head_idx);
            let k_head = cache.k.index_axis(Axis(0), head_idx);
            let grad_q_head = grad_scores.dot(&k_head);
            let grad_k_head = grad_scores.t().dot(&q_head);
            grad_q.slice_mut(s![head_idx, .., ..]).assign(&grad_q_head);
            grad_k.slice_mut(s![head_idx, .., ..]).assign(&grad_k_head);
        }

        let grad_qkv = self.unpack_qkv_grads(&grad_q, &grad_k, &grad_v, batch, time);
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
    ln1: LayerNormCache,
    ln2: LayerNormCache,
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
    fn new<R: Rng + ?Sized>(n_embd: usize, n_head: usize, block_size: usize, rng: &mut R) -> Self {
        Self {
            ln1: LayerNorm::new(n_embd),
            attention: MultiHeadSelfAttention::new(n_embd, n_head, block_size, rng),
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
        let (ln1_output, ln1_cache) = self.ln1.forward(input);
        let (attention_output, attention_cache) =
            self.attention.forward(&ln1_output, training, dropout, rng);
        let residual_after_attention = input + &attention_output;
        let (ln2_output, ln2_cache) = self.ln2.forward(&residual_after_attention);
        let (feed_forward_output, feed_forward_cache) =
            self.feed_forward
                .forward(&ln2_output, training, dropout, rng);
        let output = &residual_after_attention + &feed_forward_output;

        let cache = BlockCache {
            ln1: ln1_cache,
            ln2: ln2_cache,
            attention: attention_cache,
            feed_forward: feed_forward_cache,
        };

        (output, cache)
    }

    fn backward(&mut self, cache: &BlockCache, grad_output: &Array3<f32>) -> Array3<f32> {
        let mut grad_residual_after_attention = grad_output.clone();
        let grad_feed_forward = self.feed_forward.backward(&cache.feed_forward, grad_output);
        let grad_ln2_input = self.ln2.backward(&cache.ln2, &grad_feed_forward);
        grad_residual_after_attention += &grad_ln2_input;

        let mut grad_input = grad_residual_after_attention.clone();
        let grad_attention = self
            .attention
            .backward(&cache.attention, &grad_residual_after_attention);
        let grad_ln1_input = self.ln1.backward(&cache.ln1, &grad_attention);
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
    final_norm: LayerNormCache,
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
            .map(|_| Block::new(config.n_embd, config.n_head, config.block_size, rng))
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
        let token_values = self
            .token_embedding
            .value
            .as_slice_memory_order()
            .expect("token embeddings should be contiguous");
        let position_values = self
            .position_embedding
            .value
            .as_slice_memory_order()
            .expect("position embeddings should be contiguous");
        let idx_values = idx
            .as_slice_memory_order()
            .expect("token indices should be contiguous");
        x.as_slice_memory_order_mut()
            .expect("embedding output should be contiguous")
            .par_chunks_mut(self.config.n_embd)
            .enumerate()
            .for_each(|(token_idx, out)| {
                let token_offset = idx_values[token_idx] * self.config.n_embd;
                let position_offset = (token_idx % time) * self.config.n_embd;
                for channel in 0..self.config.n_embd {
                    out[channel] = token_values[token_offset + channel]
                        + position_values[position_offset + channel];
                }
            });

        let mut block_caches = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (next_x, cache) = block.forward(&x, training, self.config.dropout, rng);
            block_caches.push(cache);
            x = next_x;
        }

        let (final_norm_output, final_norm_cache) = self.ln_f.forward(&x);
        let logits = self.lm_head.forward(&final_norm_output);
        let cache = GptCache {
            idx: idx.clone(),
            final_norm: final_norm_cache,
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
        grad = self.ln_f.backward(&cache.final_norm, &grad);

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
        self.token_embedding.step_without_weight_decay(step);
        self.position_embedding.step_without_weight_decay(step);
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

    #[test]
    fn adamw_decays_only_matrix_weights() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let config = GptConfig {
            vocab_size: 7,
            block_size: 4,
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            dropout: 0.0,
        };
        let mut model = GptLanguageModel::new(config, &mut rng);
        model
            .blocks
            .first_mut()
            .unwrap()
            .attention
            .proj
            .bias
            .as_mut()
            .unwrap()
            .value
            .fill(1.0);

        let token_embedding = model.token_embedding.value.clone();
        let position_embedding = model.position_embedding.value.clone();
        let ln_gamma = model.blocks[0].ln1.gamma.value.clone();
        let proj_bias = model.blocks[0]
            .attention
            .proj
            .bias
            .as_ref()
            .unwrap()
            .value
            .clone();
        let qkv_weight = model.blocks[0].attention.qkv.weight.value.clone();

        model.zero_grad();
        let mut optimizer = AdamW::new(1e-3, 0.1);
        model.step(&mut optimizer);

        assert_eq!(model.token_embedding.value, token_embedding);
        assert_eq!(model.position_embedding.value, position_embedding);
        assert_eq!(model.blocks[0].ln1.gamma.value, ln_gamma);
        assert_eq!(
            model.blocks[0].attention.proj.bias.as_ref().unwrap().value,
            proj_bias
        );
        assert_ne!(model.blocks[0].attention.qkv.weight.value, qkv_weight);
    }
}
