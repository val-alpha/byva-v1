//! The ne2410 training loop: K4 token-budget batches (constant FLOPs per
//! step), one fresh single-use graph per step, AdamW (default) or SignTrust,
//! warmup + cosine LR schedule, held-out eval, and exact-resume
//! checkpointing.

use byva_autodiff::{AdamW, AdamWConfig, Graph, NodeId, SignTrust, SignTrustConfig};
use byva_data::{Lang, Sequences};

use crate::checkpoint::{
    self, CheckpointMeta, EvalRecord, FORMAT_ID, MixtureRecord, MixtureState, OptimStateDesc,
    TensorDesc,
};
use crate::config::OptimizerKind;
use crate::config::TrainConfig;
use crate::error::TrainerError;
use crate::model::{ParamLayout, build_layout, forward, init_params};

enum Optimizer {
    AdamW(AdamW),
    SignTrust(SignTrust),
}

impl Optimizer {
    fn set_learning_rate(&mut self, lr: f64) -> Result<(), TrainerError> {
        match self {
            Optimizer::AdamW(optimizer) => optimizer.set_learning_rate(lr as f32)?,
            Optimizer::SignTrust(optimizer) => optimizer.set_learning_rate(lr as f32)?,
        }
        Ok(())
    }

    fn step(
        &mut self,
        params: &mut [f32],
        grads: &[f32],
        layout: &ParamLayout,
    ) -> Result<(), TrainerError> {
        match self {
            Optimizer::AdamW(optimizer) => optimizer.step(params, grads)?,
            Optimizer::SignTrust(optimizer) => {
                for (index, spec) in layout.specs.iter().enumerate() {
                    let span = spec.offset..spec.offset + spec.len;
                    optimizer.step_layer(index, &mut params[span.clone()], &grads[span])?;
                }
            }
        }
        Ok(())
    }

    fn kind(&self) -> &'static str {
        match self {
            Optimizer::AdamW(_) => "adamw",
            Optimizer::SignTrust(_) => "signtrust",
        }
    }

    fn blobs(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        match self {
            Optimizer::AdamW(optimizer) => {
                let state = optimizer.state();
                vec![
                    ("adamw.m".to_string(), vec![state.m.len()], state.m),
                    ("adamw.v".to_string(), vec![state.v.len()], state.v),
                ]
            }
            Optimizer::SignTrust(optimizer) => {
                let state = optimizer.state();
                state
                    .momenta
                    .into_iter()
                    .enumerate()
                    .map(|(index, momentum)| {
                        (
                            format!("signtrust.momentum.{index}"),
                            vec![momentum.len()],
                            momentum,
                        )
                    })
                    .collect()
            }
        }
    }
}

/// The ne2410 trainer over one packed train/val dataset.
pub struct Trainer {
    config: TrainConfig,
    layout: ParamLayout,
    params: Vec<f32>,
    optimizer: Optimizer,
    train: Sequences,
    val: Sequences,
    eval_rows: usize,
    batch_size: usize,
    step: u64,
    batch_cursor: usize,
    loss_history: Vec<f32>,
    eval_history: Vec<EvalRecord>,
    train_langs: Option<Vec<Lang>>,
    val_langs: Option<Vec<Lang>>,
    id_rows: Vec<usize>,
    en_rows: Vec<usize>,
    val_id_rows: Vec<usize>,
    val_en_rows: Vec<usize>,
    id_cursor: usize,
    en_cursor: usize,
    id_share: f64,
    mixture_history: Vec<MixtureRecord>,
    train_mask: Option<Vec<f32>>,
    val_mask: Option<Vec<f32>>,
}

impl Trainer {
    /// Creates a trainer from a validated config and packed sequences.
    pub fn new(
        config: TrainConfig,
        train: Sequences,
        val: Sequences,
    ) -> Result<Self, TrainerError> {
        config.validate()?;
        let invalid = |reason: String| TrainerError::DataInvalid { reason };
        let batch_size = config.batch_size()?;
        if train.seq_len() != config.model.seq_len {
            return Err(invalid(format!(
                "train sequences use seq_len {} but the config uses {}",
                train.seq_len(),
                config.model.seq_len
            )));
        }
        if val.seq_len() != config.model.seq_len {
            return Err(invalid(format!(
                "val sequences use seq_len {} but the config uses {}",
                val.seq_len(),
                config.model.seq_len
            )));
        }
        if train.count() < batch_size {
            return Err(invalid(format!(
                "train split has {} window(s) but the token budget needs {} per step",
                train.count(),
                batch_size
            )));
        }
        if val.count() == 0 {
            return Err(invalid("val split has no windows".to_string()));
        }
        let layout = build_layout(&config.model)?;
        let params = init_params(&config.model, &layout, config.seed)?;
        let optimizer = match config.optimizer {
            OptimizerKind::AdamW => Optimizer::AdamW(AdamW::new(
                AdamWConfig {
                    lr: config.learning_rate as f32,
                    beta1: config.beta1 as f32,
                    beta2: config.beta2 as f32,
                    eps: config.eps as f32,
                    weight_decay: config.weight_decay as f32,
                },
                layout.total,
            )?),
            OptimizerKind::SignTrust => {
                let counts: Vec<usize> = layout.specs.iter().map(|spec| spec.len).collect();
                Optimizer::SignTrust(SignTrust::new(
                    SignTrustConfig {
                        lr: config.learning_rate as f32,
                        beta1: config.beta1 as f32,
                        weight_decay: config.weight_decay as f32,
                        clamp_min: config.clamp_min as f32,
                        clamp_max: config.clamp_max as f32,
                    },
                    &counts,
                )?)
            }
        };
        let id_share = config.mixture_target_id.unwrap_or(0.0);
        Ok(Self {
            train_mask: None,
            val_mask: None,
            eval_rows: config.eval_rows.min(val.count()),
            config,
            layout,
            params,
            optimizer,
            train,
            val,
            batch_size,
            step: 0,
            batch_cursor: 0,
            loss_history: Vec::new(),
            eval_history: Vec::new(),
            train_langs: None,
            val_langs: None,
            id_rows: Vec::new(),
            en_rows: Vec::new(),
            val_id_rows: Vec::new(),
            val_en_rows: Vec::new(),
            id_cursor: 0,
            en_cursor: 0,
            id_share,
            mixture_history: Vec::new(),
        })
    }

    /// Creates a masked-loss trainer (Phase 6 SFT): the loss covers only
    /// positions whose mask weight is positive (response tokens only).
    pub fn new_masked(
        config: TrainConfig,
        train: Sequences,
        val: Sequences,
        train_mask: Vec<f32>,
        val_mask: Vec<f32>,
    ) -> Result<Self, TrainerError> {
        let mut trainer = Self::new(config, train, val)?;
        trainer.set_loss_masks(train_mask, val_mask)?;
        Ok(trainer)
    }

    /// Starts a fine-tune from a `byva-train-rust-checkpoint-v1` base:
    /// weights (and the position-embedding prefix when the fine-tune uses a
    /// shorter `seq_len`) come from the checkpoint, while the optimizer,
    /// step counter, and loss histories start fresh.
    pub fn fine_tune(
        base_path: &std::path::Path,
        config: TrainConfig,
        train: Sequences,
        val: Sequences,
        train_mask: Option<Vec<f32>>,
        val_mask: Option<Vec<f32>>,
    ) -> Result<Self, TrainerError> {
        config.validate()?;
        let loaded = checkpoint::load(base_path)?;
        let meta = loaded.meta;
        if meta.format != FORMAT_ID {
            return Err(TrainerError::UnknownCheckpointFormat { found: meta.format });
        }
        let base_model = meta.config.model;
        let model = config.model.clone();
        let mismatch = |reason: String| TrainerError::ConfigMismatch { reason };
        if (
            model.vocab,
            model.d_model,
            model.n_layers,
            model.n_heads,
            model.ffn,
        ) != (
            base_model.vocab,
            base_model.d_model,
            base_model.n_layers,
            base_model.n_heads,
            base_model.ffn,
        ) {
            return Err(mismatch(
                "fine-tune model dimensions differ from the base checkpoint".to_string(),
            ));
        }
        if model.seq_len > base_model.seq_len {
            return Err(mismatch(format!(
                "fine-tune seq_len {} exceeds the base checkpoint's {}",
                model.seq_len, base_model.seq_len
            )));
        }
        let layout = build_layout(&model)?;
        let mut params = vec![0.0f32; layout.total];
        for spec in &layout.specs {
            let base_index = meta
                .params
                .iter()
                .position(|desc| desc.name == spec.name)
                .ok_or_else(|| {
                    mismatch(format!("base checkpoint lacks parameter {}", spec.name))
                })?;
            let base_offset = meta
                .params
                .iter()
                .take(base_index)
                .map(|desc| desc.len)
                .sum::<usize>();
            let base_spec = &meta.params[base_index];
            let dst = &mut params[spec.offset..spec.offset + spec.len];
            if spec.name == "position_embedding.weight" {
                let rows = model.seq_len * model.d_model;
                if base_spec.len < rows {
                    return Err(mismatch(format!(
                        "base position embedding holds {} elements, need {}",
                        base_spec.len, rows
                    )));
                }
                dst.copy_from_slice(&loaded.params[base_offset..base_offset + rows]);
            } else {
                if base_spec.shape != spec.shape || base_spec.len != spec.len {
                    return Err(mismatch(format!(
                        "parameter {} layout differs: base {:?}/{:?}, config {:?}/{:?}",
                        spec.name, base_spec.shape, base_spec.len, spec.shape, spec.len
                    )));
                }
                dst.copy_from_slice(&loaded.params[spec.offset..spec.offset + spec.len]);
            }
        }
        let optimizer = Self::fresh_optimizer(&config, &layout)?;
        let id_share = config.mixture_target_id.unwrap_or(0.0);
        let mut trainer = Self {
            eval_rows: config.eval_rows.min(val.count()),
            batch_size: config.batch_size()?,
            config,
            layout,
            params,
            optimizer,
            train,
            val,
            step: 0,
            batch_cursor: 0,
            loss_history: Vec::new(),
            eval_history: Vec::new(),
            train_langs: None,
            val_langs: None,
            id_rows: Vec::new(),
            en_rows: Vec::new(),
            val_id_rows: Vec::new(),
            val_en_rows: Vec::new(),
            id_cursor: 0,
            en_cursor: 0,
            id_share,
            mixture_history: Vec::new(),
            train_mask: None,
            val_mask: None,
        };
        if let (Some(train_mask), Some(val_mask)) = (train_mask, val_mask) {
            trainer.set_loss_masks(train_mask, val_mask)?;
        }
        Ok(trainer)
    }

    fn fresh_optimizer(
        config: &TrainConfig,
        layout: &ParamLayout,
    ) -> Result<Optimizer, TrainerError> {
        match config.optimizer {
            OptimizerKind::AdamW => Ok(Optimizer::AdamW(AdamW::new(
                AdamWConfig {
                    lr: config.learning_rate as f32,
                    beta1: config.beta1 as f32,
                    beta2: config.beta2 as f32,
                    eps: config.eps as f32,
                    weight_decay: config.weight_decay as f32,
                },
                layout.total,
            )?)),
            OptimizerKind::SignTrust => {
                let counts: Vec<usize> = layout.specs.iter().map(|spec| spec.len).collect();
                Ok(Optimizer::SignTrust(SignTrust::new(
                    SignTrustConfig {
                        lr: config.learning_rate as f32,
                        beta1: config.beta1 as f32,
                        weight_decay: config.weight_decay as f32,
                        clamp_min: config.clamp_min as f32,
                        clamp_max: config.clamp_max as f32,
                    },
                    &counts,
                )?))
            }
        }
    }

    /// Attaches per-position loss masks (Phase 6 SFT): one weight per target
    /// position of every train and val window. The K5 mixture is
    /// incompatible with masked loss and is rejected here.
    pub fn set_loss_masks(
        &mut self,
        train_mask: Vec<f32>,
        val_mask: Vec<f32>,
    ) -> Result<(), TrainerError> {
        if self.config.mixture_target_id.is_some() {
            return Err(TrainerError::ConfigInvalid {
                reason: "masked loss cannot be combined with the K5 mixture".to_string(),
            });
        }
        Self::validate_mask(
            "train",
            &train_mask,
            self.train.count(),
            self.config.model.seq_len,
        )?;
        Self::validate_mask(
            "val",
            &val_mask,
            self.val.count(),
            self.config.model.seq_len,
        )?;
        self.train_mask = Some(train_mask);
        self.val_mask = Some(val_mask);
        Ok(())
    }

    fn validate_mask(
        name: &str,
        mask: &[f32],
        count: usize,
        seq_len: usize,
    ) -> Result<(), TrainerError> {
        let invalid = |reason: String| TrainerError::DataInvalid { reason };
        if mask.len() != count * seq_len {
            return Err(invalid(format!(
                "{} mask holds {} weights for {} windows of {} positions",
                name,
                mask.len(),
                count,
                seq_len
            )));
        }
        if mask
            .iter()
            .any(|&weight| !weight.is_finite() || !(0.0..=1.0).contains(&weight))
        {
            return Err(invalid(format!(
                "{} mask weights must be finite and within [0, 1]",
                name
            )));
        }
        if mask.iter().all(|&weight| weight == 0.0) {
            return Err(invalid(format!("{} mask has no supervised position", name)));
        }
        Ok(())
    }

    /// Attaches per-window language tags for the K5 id:en mixture (D-0016).
    ///
    /// One tag per train window and one per val window, in window order. When
    /// the mixture is enabled the trainer pools windows per language, samples
    /// each step's batch from the pools at the current Indonesian share, and
    /// adapts that share at every evaluation within the configured band.
    pub fn set_language_tags(
        &mut self,
        train: Vec<Lang>,
        val: Vec<Lang>,
    ) -> Result<(), TrainerError> {
        let invalid = |reason: String| TrainerError::DataInvalid { reason };
        if train.len() != self.train.count() {
            return Err(invalid(format!(
                "{} train language tags for {} train windows",
                train.len(),
                self.train.count()
            )));
        }
        if val.len() != self.val.count() {
            return Err(invalid(format!(
                "{} val language tags for {} val windows",
                val.len(),
                self.val.count()
            )));
        }
        if self.config.mixture_target_id.is_some() {
            let id_train = train.iter().filter(|lang| **lang == Lang::Id).count();
            if id_train == 0 || id_train == train.len() {
                return Err(invalid(format!(
                    "mixture needs both languages in the train split, found {}/{} id windows",
                    id_train,
                    train.len()
                )));
            }
        }
        let pool = |langs: &[Lang], lang: Lang| {
            langs
                .iter()
                .enumerate()
                .filter(|(_, tag)| **tag == lang)
                .map(|(row, _)| row)
                .collect::<Vec<usize>>()
        };
        self.val_id_rows = pool(&val, Lang::Id);
        self.val_en_rows = pool(&val, Lang::En);
        self.id_rows = pool(&train, Lang::Id);
        self.en_rows = pool(&train, Lang::En);
        self.train_langs = Some(train);
        self.val_langs = Some(val);
        Ok(())
    }

    /// Restores a trainer from a checkpoint plus the same packed sequences.
    ///
    /// The checkpoint config must match the run (model dims, optimizer,
    /// schedule); the dataloader position, loss history, parameters, and
    /// optimizer state are restored so the continuation is exact.
    pub fn resume(
        path: &std::path::Path,
        train: Sequences,
        val: Sequences,
    ) -> Result<Self, TrainerError> {
        Self::resume_inner(path, train, val, None)
    }

    /// Restores a masked-loss (SFT) trainer from a checkpoint plus the same
    /// masked dataset. The checkpoint must have been written by a masked run.
    pub fn resume_masked(
        path: &std::path::Path,
        train: Sequences,
        val: Sequences,
        train_mask: Vec<f32>,
        val_mask: Vec<f32>,
    ) -> Result<Self, TrainerError> {
        Self::resume_inner(path, train, val, Some((train_mask, val_mask)))
    }

    fn resume_inner(
        path: &std::path::Path,
        train: Sequences,
        val: Sequences,
        masks: Option<(Vec<f32>, Vec<f32>)>,
    ) -> Result<Self, TrainerError> {
        let loaded = checkpoint::load(path)?;
        let meta = loaded.meta;
        let config = meta.config.clone();
        config.validate()?;
        let mismatch = |reason: String| TrainerError::ConfigMismatch { reason };
        if meta.format != FORMAT_ID {
            return Err(TrainerError::UnknownCheckpointFormat { found: meta.format });
        }
        if train.seq_len() != config.model.seq_len || val.seq_len() != config.model.seq_len {
            return Err(mismatch(
                "sequences do not use the checkpoint's seq_len".to_string(),
            ));
        }
        let batch_size = config.batch_size()?;
        if train.count() < batch_size {
            return Err(mismatch(format!(
                "train split has {} window(s) but the checkpoint needs {} per step",
                train.count(),
                batch_size
            )));
        }
        if val.count() == 0 {
            return Err(mismatch("val split has no windows".to_string()));
        }
        if meta.batch_cursor >= train.count() {
            return Err(mismatch(format!(
                "dataloader cursor {} is outside the train split ({} windows)",
                meta.batch_cursor,
                train.count()
            )));
        }
        if meta.mixture_state.is_some() && config.mixture_target_id.is_none() {
            return Err(mismatch(
                "checkpoint holds mixture state but its config disables the mixture".to_string(),
            ));
        }
        if meta.masked_loss != masks.is_some() {
            return Err(mismatch(
                "checkpoint and resume disagree on masked loss; masked checkpoints resume only through resume_masked with the same dataset"
                    .to_string(),
            ));
        }
        if meta.step > config.steps {
            return Err(mismatch(format!(
                "checkpoint step {} exceeds the configured total {}",
                meta.step, config.steps
            )));
        }
        let layout = build_layout(&config.model)?;
        if layout.total != loaded.params.len() {
            return Err(mismatch(format!(
                "checkpoint holds {} parameters but the config needs {}",
                loaded.params.len(),
                layout.total
            )));
        }
        for (spec, desc) in layout.specs.iter().zip(&meta.params) {
            if spec.name != desc.name || spec.shape != desc.shape || spec.len != desc.len {
                return Err(mismatch(format!(
                    "parameter layout differs at {}: checkpoint {:?}/{:?}, config {:?}/{:?}",
                    spec.name, desc.name, desc.shape, spec.name, spec.shape
                )));
            }
        }
        let expected_kind = match config.optimizer {
            OptimizerKind::AdamW => "adamw",
            OptimizerKind::SignTrust => "signtrust",
        };
        if meta.optimizer.kind != expected_kind {
            return Err(mismatch(format!(
                "checkpoint optimizer is {} but the config says {}",
                meta.optimizer.kind, expected_kind
            )));
        }
        let optimizer = match config.optimizer {
            OptimizerKind::AdamW => {
                let mut optimizer = AdamW::new(
                    AdamWConfig {
                        lr: config.learning_rate as f32,
                        beta1: config.beta1 as f32,
                        beta2: config.beta2 as f32,
                        eps: config.eps as f32,
                        weight_decay: config.weight_decay as f32,
                    },
                    layout.total,
                )?;
                let m = loaded
                    .optim_blobs
                    .first()
                    .ok_or_else(|| mismatch("checkpoint is missing adamw.m".to_string()))?
                    .clone();
                let v = loaded
                    .optim_blobs
                    .get(1)
                    .ok_or_else(|| mismatch("checkpoint is missing adamw.v".to_string()))?
                    .clone();
                optimizer.restore(byva_autodiff::AdamWState {
                    m,
                    v,
                    step: meta.step,
                })?;
                Optimizer::AdamW(optimizer)
            }
            OptimizerKind::SignTrust => {
                let counts: Vec<usize> = layout.specs.iter().map(|spec| spec.len).collect();
                let mut optimizer = SignTrust::new(
                    SignTrustConfig {
                        lr: config.learning_rate as f32,
                        beta1: config.beta1 as f32,
                        weight_decay: config.weight_decay as f32,
                        clamp_min: config.clamp_min as f32,
                        clamp_max: config.clamp_max as f32,
                    },
                    &counts,
                )?;
                if loaded.optim_blobs.len() != counts.len() {
                    return Err(mismatch(format!(
                        "checkpoint holds {} momentum buffers but the model has {} parameters",
                        loaded.optim_blobs.len(),
                        counts.len()
                    )));
                }
                optimizer.restore(byva_autodiff::SignTrustState {
                    momenta: loaded.optim_blobs.clone(),
                    steps: meta.step.saturating_mul(layout.specs.len() as u64),
                })?;
                Optimizer::SignTrust(optimizer)
            }
        };
        let (id_cursor, en_cursor, id_share) = match meta.mixture_state {
            Some(state) => (state.id_cursor, state.en_cursor, state.id_share),
            None => (0, 0, config.mixture_target_id.unwrap_or(0.0)),
        };
        let mut trainer = Self {
            eval_rows: config.eval_rows.min(val.count()),
            config,
            layout,
            params: loaded.params,
            optimizer,
            train,
            val,
            batch_size,
            step: meta.step,
            batch_cursor: meta.batch_cursor,
            loss_history: meta.loss_history,
            eval_history: meta.eval_history,
            train_langs: None,
            val_langs: None,
            id_rows: Vec::new(),
            en_rows: Vec::new(),
            val_id_rows: Vec::new(),
            val_en_rows: Vec::new(),
            id_cursor,
            en_cursor,
            id_share,
            mixture_history: meta.mixture_history,
            train_mask: None,
            val_mask: None,
        };
        if let Some((train_mask, val_mask)) = masks {
            trainer.set_loss_masks(train_mask, val_mask)?;
        }
        Ok(trainer)
    }

    /// Overrides the total step target of a resumed run.
    ///
    /// The completed prefix stays untouched; the schedule tail is recomputed
    /// from the new target, so extending a run past its original target
    /// intentionally re-plans the cosine decay.
    pub fn set_total_steps(&mut self, steps: u64) -> Result<(), TrainerError> {
        if steps < self.step.max(1) {
            return Err(TrainerError::ConfigInvalid {
                reason: format!(
                    "new step target {steps} is below completed step {}",
                    self.step
                ),
            });
        }
        self.config.steps = steps;
        Ok(())
    }

    /// Runs train steps until the configured total is reached.
    pub fn run(&mut self) -> Result<(), TrainerError> {
        while self.step < self.config.steps {
            self.train_step()?;
        }
        Ok(())
    }

    /// One K4 step: constant token budget, forward, backward, optimizer
    /// update, loss recorded; evaluates when the schedule says so.
    pub fn train_step(&mut self) -> Result<f32, TrainerError> {
        if self.config.mixture_target_id.is_some() && self.train_langs.is_none() {
            return Err(TrainerError::ConfigInvalid {
                reason: "mixture_target_id is set but language tags were never attached"
                    .to_string(),
            });
        }
        let seq_len = self.config.model.seq_len;
        let count = self.train.count();
        let rows = self.batch_rows();
        let train_mask = self.train_mask.as_ref();
        let mut tokens = Vec::with_capacity(self.batch_size * seq_len);
        let mut targets = Vec::with_capacity(self.batch_size * seq_len);
        let mut mask = train_mask.map(|_| Vec::with_capacity(self.batch_size * seq_len));
        for row in rows {
            let span = row * seq_len..(row + 1) * seq_len;
            tokens.extend_from_slice(&self.train.x()[span.clone()]);
            targets.extend_from_slice(&self.train.y()[span.clone()]);
            if let (Some(mask), Some(weights)) = (mask.as_mut(), train_mask) {
                mask.extend_from_slice(&weights[span]);
            }
        }
        let mut graph = Graph::new();
        let leaves = self.leaves(&mut graph)?;
        let loss_node = forward(
            &mut graph,
            &self.config.model,
            &self.layout,
            &leaves,
            &tokens,
            &targets,
            mask.as_deref(),
        )?;
        let loss_value = graph.value(loss_node)?.data()[0];
        graph.backward(loss_node)?;
        let mut grads = vec![0.0f32; self.layout.total];
        for (spec, leaf) in self.layout.specs.iter().zip(&leaves) {
            if let Some(gradient) = graph.grad(*leaf)? {
                grads[spec.offset..spec.offset + spec.len].copy_from_slice(gradient);
            }
        }
        let supervised = mask
            .as_deref()
            .map(|weights| weights.iter().sum::<f32>())
            .unwrap_or(1.0);
        let next_step = self.step + 1;
        if supervised > 0.0 {
            let lr = self.config.learning_rate_at(next_step);
            self.optimizer.set_learning_rate(lr)?;
            self.optimizer
                .step(&mut self.params, &grads, &self.layout)?;
        }
        self.batch_cursor = (self.batch_cursor + self.batch_size) % count;
        self.step = next_step;
        self.loss_history.push(loss_value);

        if self.config.eval_every > 0 && self.step % self.config.eval_every == 0 {
            let held_out = self.evaluate()?;
            let mut record = EvalRecord {
                step: self.step,
                loss: held_out,
                loss_id: None,
                loss_en: None,
            };
            if self.mixture_enabled() {
                let id_loss = self.evaluate_language(&self.val_id_rows)?;
                let en_loss = self.evaluate_language(&self.val_en_rows)?;
                if let (Some(id_loss), Some(en_loss)) = (id_loss, en_loss) {
                    let target = self.config.mixture_target_id.unwrap_or(0.5);
                    let band = self.config.mixture_band;
                    let adjusted = (target + (id_loss as f64 - en_loss as f64))
                        .clamp(target - band, target + band);
                    self.id_share = adjusted;
                    self.mixture_history.push(MixtureRecord {
                        step: self.step,
                        id_share: adjusted,
                        id_loss,
                        en_loss,
                    });
                }
                record.loss_id = id_loss;
                record.loss_en = en_loss;
            }
            self.eval_history.push(record);
        }
        Ok(loss_value)
    }

    /// Whether the K5 id:en mixture is enabled and its pools are attached.
    fn mixture_enabled(&self) -> bool {
        self.config.mixture_target_id.is_some() && self.train_langs.is_some()
    }

    /// The train-window rows feeding this step: the natural K4 order, or the
    /// K5 pool sampling at the current Indonesian share (id rows first, then
    /// en rows; both pools cycle independently).
    fn batch_rows(&mut self) -> Vec<usize> {
        if !self.mixture_enabled() {
            let count = self.train.count();
            return (0..self.batch_size)
                .map(|offset| (self.batch_cursor + offset) % count)
                .collect();
        }
        let mut id_count = ((self.batch_size as f64) * self.id_share).round() as usize;
        id_count = id_count.min(self.batch_size);
        let mut rows = Vec::with_capacity(self.batch_size);
        for _ in 0..id_count {
            let row = self.id_rows[self.id_cursor % self.id_rows.len()];
            rows.push(row);
            self.id_cursor += 1;
        }
        for _ in 0..(self.batch_size - id_count) {
            let row = self.en_rows[self.en_cursor % self.en_rows.len()];
            rows.push(row);
            self.en_cursor += 1;
        }
        rows
    }

    /// Mean held-out loss over the given val rows (up to `eval_rows`).
    fn evaluate_language(&self, rows: &[usize]) -> Result<Option<f32>, TrainerError> {
        if rows.is_empty() {
            return Ok(None);
        }
        let seq_len = self.config.model.seq_len;
        let count = rows.len().min(self.eval_rows);
        let mut tokens = Vec::with_capacity(count * seq_len);
        let mut targets = Vec::with_capacity(count * seq_len);
        for &row in &rows[..count] {
            let span = row * seq_len..(row + 1) * seq_len;
            tokens.extend_from_slice(&self.val.x()[span.clone()]);
            targets.extend_from_slice(&self.val.y()[span]);
        }
        Ok(Some(self.forward_loss(&tokens, &targets, None)?))
    }

    /// Mean held-out loss over the first `eval_rows` val windows (no
    /// backward pass).
    pub fn evaluate(&self) -> Result<f32, TrainerError> {
        let seq_len = self.config.model.seq_len;
        let val_mask = self.val_mask.as_ref();
        let selected: Vec<usize> = match val_mask {
            Some(mask) => (0..self.val.count())
                .filter(|&row| {
                    mask[row * seq_len..(row + 1) * seq_len]
                        .iter()
                        .any(|&weight| weight > 0.0)
                })
                .take(self.eval_rows)
                .collect(),
            None => (0..self.eval_rows.min(self.val.count())).collect(),
        };
        let rows = selected.len();
        if rows == 0 {
            return Err(TrainerError::DataInvalid {
                reason: "val split has no windows".to_string(),
            });
        }
        let mut tokens = Vec::with_capacity(rows * seq_len);
        let mut targets = Vec::with_capacity(rows * seq_len);
        let mut mask = val_mask.map(|_| Vec::with_capacity(rows * seq_len));
        for &row in &selected {
            let span = row * seq_len..(row + 1) * seq_len;
            tokens.extend_from_slice(&self.val.x()[span.clone()]);
            targets.extend_from_slice(&self.val.y()[span.clone()]);
            if let (Some(mask), Some(weights)) = (mask.as_mut(), val_mask) {
                mask.extend_from_slice(&weights[span]);
            }
        }
        self.forward_loss(&tokens, &targets, mask.as_deref())
    }

    /// Computes the (optionally masked) mean cross-entropy loss of the
    /// current parameters over one batch of tokens and targets.
    ///
    /// # Errors
    /// Fails when the batch shape is invalid or the forward pass errors.
    pub fn forward_loss(
        &self,
        tokens: &[u32],
        targets: &[u32],
        mask: Option<&[f32]>,
    ) -> Result<f32, TrainerError> {
        let mut graph = Graph::new();
        let leaves = self.leaves(&mut graph)?;
        let loss = forward(
            &mut graph,
            &self.config.model,
            &self.layout,
            &leaves,
            tokens,
            targets,
            mask,
        )?;
        Ok(graph.value(loss)?.data()[0])
    }

    fn leaves(&self, graph: &mut Graph) -> Result<Vec<NodeId>, TrainerError> {
        let mut leaves = Vec::with_capacity(self.layout.specs.len());
        for spec in &self.layout.specs {
            let span = spec.offset..spec.offset + spec.len;
            leaves.push(graph.leaf(&spec.shape, self.params[span].to_vec())?);
        }
        Ok(leaves)
    }

    /// Writes a checkpoint capturing the exact resumable state.
    pub fn save(&self, path: &std::path::Path) -> Result<(), TrainerError> {
        let params: Vec<TensorDesc> = self
            .layout
            .specs
            .iter()
            .map(|spec| TensorDesc {
                name: spec.name.clone(),
                shape: spec.shape.clone(),
                len: spec.len,
            })
            .collect();
        let blobs = self.optimizer.blobs();
        let optimizer = OptimStateDesc {
            kind: self.optimizer.kind().to_string(),
            blobs: blobs
                .iter()
                .map(|(name, shape, _)| TensorDesc {
                    name: name.clone(),
                    shape: shape.clone(),
                    len: shape.first().copied().unwrap_or(0),
                })
                .collect(),
        };
        let mixture_state = if self.mixture_enabled() {
            Some(MixtureState {
                id_cursor: self.id_cursor,
                en_cursor: self.en_cursor,
                id_share: self.id_share,
            })
        } else {
            None
        };
        let meta = CheckpointMeta {
            format: FORMAT_ID.to_string(),
            config: self.config.clone(),
            step: self.step,
            batch_cursor: self.batch_cursor,
            loss_history: self.loss_history.clone(),
            eval_history: self.eval_history.clone(),
            mixture_history: self.mixture_history.clone(),
            mixture_state,
            masked_loss: self.train_mask.is_some(),
            params,
            optimizer,
        };
        let raw: Vec<Vec<f32>> = blobs.into_iter().map(|(_, _, values)| values).collect();
        checkpoint::save(path, &meta, &self.params, &raw)
    }

    /// The current flat parameter buffer (canonical layout order).
    pub fn params(&self) -> &[f32] {
        &self.params
    }

    /// Completed steps.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Per-step training loss history (mean-reduced).
    pub fn loss_history(&self) -> &[f32] {
        &self.loss_history
    }

    /// Held-out evaluation history.
    pub fn eval_history(&self) -> &[EvalRecord] {
        &self.eval_history
    }

    /// Every recorded K5 mixture adjustment (empty when the mixture is off).
    pub fn mixture_history(&self) -> &[MixtureRecord] {
        &self.mixture_history
    }

    /// The Indonesian window share currently in effect (K5; 0 when off).
    pub fn id_share(&self) -> f64 {
        if self.mixture_enabled() {
            self.id_share
        } else {
            0.0
        }
    }

    /// Whether the run trains with a per-position loss mask (Phase 6 SFT).
    pub fn masked_loss(&self) -> bool {
        self.train_mask.is_some()
    }

    /// The run configuration.
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    /// Tokens consumed per step (K4: constant).
    pub fn tokens_per_step(&self) -> usize {
        self.batch_size * self.config.model.seq_len
    }
}
