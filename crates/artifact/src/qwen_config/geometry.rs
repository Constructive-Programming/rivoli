//! Qwen3.8-Flash-Next's DERIVED widths — every number a tensor shape has that the config states
//! only as a head count, plus the two per-layer predicates the census bounds its indices with.
//!
//! **Split from `qwen_config.rs` under the 800-line soft cap**, and the split is by job rather
//! than by size: that file declares the schema and refuses a document, this one answers "how wide
//! is `in_proj_qkv`". Inherent `impl` blocks in a sibling module, so nothing about the call sites
//! changed — `cfg.hc_stream()` reads the same from anywhere in the crate.
//!
//! **Every accessor here is a shape some census family HAS**, and `crates/cli/tests/
//! qwen_convert.rs` confronts the two — so a wrong derivation reddens against the checkpoint's own
//! shard headers instead of producing a self-consistent artifact with one projection mis-split.
//! That pairing is the reason these are methods rather than comments.

use anyhow::{Result, ensure};

use super::{FULL_ATTENTION_ALIAS, QwenTextConfig};

impl QwenTextConfig {
    /// The hyper-connection residual stream, `hc_count x hidden` = 10240. The width of every
    /// `hc_norm`, `block_inject_weight` column and `input_mix_weight_*` long axis.
    pub fn hc_stream(&self) -> usize {
        self.hc_count * self.hidden
    }

    /// GDN's Q (and K) width, `linear_num_key_heads x linear_key_head_dim` = 2048.
    pub fn gdn_key_width(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// GDN's V width, `linear_num_value_heads x linear_value_head_dim` = 6144 — the width of
    /// `in_proj_z` and of `out_proj`'s input.
    pub fn gdn_value_width(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// `in_proj_qkv`'s output width, `q + k + v` = 10240 — and also `conv1d`'s channel count,
    /// since the short convolution runs over the whole fused projection.
    pub fn gdn_qkv_width(&self) -> usize {
        2 * self.gdn_key_width() + self.gdn_value_width()
    }

    /// `q_proj`'s output width, **`2 x n_heads x head_dim` = 12288**.
    ///
    /// **The factor of two is a reading the census forces, and it is labelled rather than
    /// assumed.** `q_proj` is [12288, 2560] while `o_proj` takes 6144 = `n_heads x head_dim`, and
    /// `num_attention_heads` is 24 — so the extra 6144 columns are not query heads. The config's
    /// `output_gate_type: "sigmoid"` and GDN's own separate `in_proj_z` gate make "q ‖
    /// output-gate" the reading; S2's anchor is what will confirm the ORDER of the two halves.
    /// Nothing downstream depends on the label: the width is 12288 either way, so a wrong name
    /// here costs a comment and not an artifact.
    pub fn qsa_q_width(&self) -> usize {
        2 * self.qsa_out_width()
    }

    /// The attention output width, `n_heads x head_dim` = 6144 — `o_proj`'s input.
    pub fn qsa_out_width(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// `k_proj`/`v_proj`'s output width, `num_key_value_heads x head_dim` = 512.
    pub fn qsa_kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// `index_qk_proj`'s output width, `(indexer_n_heads + indexer_kv_heads) x indexer_head_dim`
    /// = 640, split `[4*128 | 1*128]` by MOD:645-650.
    pub fn indexer_qk_width(&self) -> usize {
        (self.indexer_n_heads + self.indexer_kv_heads) * self.indexer_head_dim
    }

    /// The rotated prefix width, `int(head_dim x partial_rotary_factor)` = 64 — the FIRST 64
    /// dims, matching the reference's own `int()` truncation (MOD:96-116).
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "the reference truncates with int(); head_dim is small and positive, and \
                  `validate` has already refused a non-positive factor"
    )]
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }

    /// `(ngram_size - 1) x heads_per_ngram` = 16 — bigram heads 0..8, trigram heads 8..16.
    pub fn ngram_heads(&self) -> usize {
        self.ngram_size.saturating_sub(1) * self.heads_per_ngram
    }

    /// `ple_embed_dim / ngram_heads` = 160 — one gathered row's width, in fp8 bytes.
    pub fn ngram_head_dim(&self) -> Result<usize> {
        let heads = self.ngram_heads();
        ensure!(heads > 0, "ngram_heads is 0");
        Ok(self.ple_embed_dim / heads)
    }

    /// **The ONE place `ple_layer_ids`' one-based id becomes a zero-based `layer_idx`.**
    ///
    /// The reference tests `layer_idx + 1 in config.ple_layer_ids` (MOD:1202) and CFG:240-246
    /// rejects ids outside `[1, num_hidden_layers]`. Getting this wrong emits `layers.2.ple.*`
    /// against a checkpoint whose PLE tensors are under `layers.1.` — trap T12, and it fails the
    /// census rather than producing a wrong artifact, which is the ordering this port wants.
    pub fn ple_host_layer(&self) -> Result<usize> {
        let &id = self
            .ple_layer_ids
            .first()
            .ok_or_else(|| anyhow::anyhow!("ple_layer_ids is empty"))?;
        ensure!(
            id >= 1 && id <= self.n_layers,
            "ple_layer_ids entry {id} is outside [1, num_hidden_layers = {}] — the array is \
             ONE-INDEXED (CFG:240-246)",
            self.n_layers
        );
        Ok(id - 1)
    }

    /// How many layers run QSA (the `full_attention`-spelled, indexer-running ones) — 12.
    pub fn sparse_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|k| *k == FULL_ATTENTION_ALIAS)
            .count()
    }

    /// Does zero-based `layer` run QSA and its indexer? (Otherwise GatedDeltaNet.)
    ///
    /// Read off `layer_types`, which `validate` has already confronted with
    /// `full_attention_interval` — so this is a lookup rather than a second derivation, and there
    /// is no arithmetic here to disagree with that check. `Result` because an out-of-range layer
    /// would otherwise read as "GDN", which is a positive claim about arithmetic
    /// (`K3Config::layer_is_mla`'s argument, and the same asymmetry with [`Self::layer_hosts_ple`]
    /// below).
    pub fn layer_is_sparse_attention(&self, layer: usize) -> Result<bool> {
        let kind = self.layer_types.get(layer).ok_or_else(|| {
            anyhow::anyhow!("layer {layer} >= num_hidden_layers {}", self.n_layers)
        })?;
        Ok(kind == FULL_ATTENTION_ALIAS)
    }

    /// Does zero-based `layer` host the n-gram table?
    ///
    /// **No bounds check, unlike the sibling above**, and for `K3Config::layer_is_dense`'s reason:
    /// an out-of-range id is not the host and reads as "no", which is the same answer it gives for
    /// every real layer but one. [`Self::ple_host_layer`] is where the id itself is bounded.
    pub fn layer_hosts_ple(&self, layer: usize) -> bool {
        self.ple_host_layer().is_ok_and(|host| host == layer)
    }
}
