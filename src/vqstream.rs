//! The VQ-int3 routed-expert source (M3 streaming): the `.i3` analog of
//! [`crate::snapshot::Snapshot`] for the ONLY tensors that change format under VQ —
//! the routed MoE experts. Everything else (attention, norms, embed/lm_head, router
//! gate, shared expert) still comes from the int4 snapshot; VQ is a hybrid load.
//!
//! Layout the converter (`bin/fp82vq`) writes: one file `L{layer:02}.i3` per MoE
//! layer, `n_experts` experts at a fixed [`vq_expert_stride`], each expert one
//! contiguous `gate‖up‖down` block. Because the stride is `VQ_ALIGN`-padded and
//! each expert offset is `e·stride`, a whole expert is ONE block-aligned O_DIRECT
//! read — simpler than int4's six-tensor coalescing (no read plan/table needed, just
//! `(fd, e·stride, expert_bytes)`). A shared `codebook.f32` is loaded once for the
//! caller to upload resident.

use anyhow::{Context, Result, ensure};
use std::os::fd::{AsRawFd, RawFd};

use crate::quant::{VQ_DIM, VQ_K, read_f32, vq_expert_bytes, vq_expert_stride};

/// Open `.i3` routed-expert files + the shared codebook. Holds the files so the
/// `RawFd`s stay valid for the run (mirrors `Snapshot`'s fd-anchor role).
pub struct VqExperts {
    /// One O_DIRECT file per MoE layer, index `layer - dense_layers`.
    files: Vec<std::fs::File>,
    /// The learned codebook (`VQ_K·VQ_DIM` f32); the caller uploads it resident once.
    pub codebook: Vec<f32>,
    dense_layers: usize,
    n_layers: usize,
    n_experts: usize,
    stride: usize,
    expert_bytes: usize,
}

impl VqExperts {
    /// Open `dir/L{l:02}.i3` for every MoE layer + `dir/codebook.f32`, validating
    /// each file is exactly `n_experts · vq_expert_stride` bytes so a bad dim or a
    /// truncated conversion fails loud here, not as OOB reads on the hot path.
    pub fn open(
        dir: &str,
        dense_layers: usize,
        n_layers: usize,
        n_experts: usize,
        hidden: usize,
        moe_inter: usize,
    ) -> Result<Self> {
        let codebook = read_f32(
            &std::fs::read(format!("{dir}/codebook.f32"))
                .with_context(|| format!("read {dir}/codebook.f32"))?,
        );
        ensure!(
            codebook.len() == VQ_K * VQ_DIM,
            "{dir}/codebook.f32: {} f32, expected VQ_K·VQ_DIM = {}",
            codebook.len(),
            VQ_K * VQ_DIM
        );
        let stride = vq_expert_stride(hidden, moe_inter);
        let expert_bytes = vq_expert_bytes(hidden, moe_inter);
        let want = n_experts * stride;
        let mut files = Vec::with_capacity(n_layers - dense_layers);
        for l in dense_layers..n_layers {
            let path = format!("{dir}/L{l:02}.i3");
            let f = open_direct(&path).with_context(|| format!("open {path}"))?;
            let len = f.metadata().with_context(|| format!("stat {path}"))?.len() as usize;
            ensure!(
                len == want,
                "{path}: {len} bytes, expected n_experts·stride = {want}"
            );
            files.push(f);
        }
        Ok(Self {
            files,
            codebook,
            dense_layers,
            n_layers,
            n_experts,
            stride,
            expert_bytes,
        })
    }

    /// Cold-read spec for one routed expert: `(fd, begin, useful_len)`. `begin` is
    /// `VQ_ALIGN`-aligned (expert offset = `e · stride`, stride a `VQ_ALIGN` multiple);
    /// `useful_len` is the unpadded expert bytes — the streamer reads the aligned
    /// superset, exactly as it does for an int4 `read_spec`.
    pub fn read_spec(&self, layer: usize, expert: usize) -> Result<(RawFd, usize, usize)> {
        ensure!(
            (self.dense_layers..self.n_layers).contains(&layer),
            "layer {layer} out of MoE range {}..{}",
            self.dense_layers,
            self.n_layers
        );
        ensure!(
            expert < self.n_experts,
            "expert {expert} >= {}",
            self.n_experts
        );
        let fd = self.files[layer - self.dense_layers].as_raw_fd();
        Ok((fd, expert * self.stride, self.expert_bytes))
    }

    /// Per-expert slot stride (the O_DIRECT-aligned block a fetch lands in).
    pub fn expert_slot(&self) -> usize {
        self.stride
    }
}

/// Open a file O_DIRECT (page-cache-bypassing NVMe DMA), matching how `Snapshot`
/// opens its cold-read fds.
fn open_direct(path: &str) -> Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    Ok(std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // test setup: panic-on-failure is the readable idiom
mod tests {
    use super::*;
    use crate::quant::{VQ_GROUP, vq_expert_stride};
    use std::io::Write;

    /// Write a synthetic `.i3` layer + codebook so the source's specs/validation are
    /// tested without the (NFS-resident, huge) real conversion output.
    #[test]
    fn read_spec_and_validation() {
        let dir = format!(
            "{}/vqstream_test",
            std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into())
        );
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (hidden, moe_inter) = (VQ_GROUP, VQ_GROUP); // tiny but valid
        let (dense, n_layers, n_experts) = (3usize, 5usize, 4usize);
        let stride = vq_expert_stride(hidden, moe_inter);

        // codebook.f32
        let cb = vec![0.0f32; VQ_K * VQ_DIM];
        let cb_bytes: Vec<u8> = cb.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/codebook.f32"), &cb_bytes).unwrap();
        // one file per MoE layer, correctly sized (zeros — specs don't read content)
        for l in dense..n_layers {
            let mut f = std::fs::File::create(format!("{dir}/L{l:02}.i3")).unwrap();
            f.write_all(&vec![0u8; n_experts * stride]).unwrap();
        }

        let src = VqExperts::open(&dir, dense, n_layers, n_experts, hidden, moe_inter).unwrap();
        assert_eq!(src.expert_slot(), stride);
        // aligned, monotonic, non-overlapping expert offsets
        let (_, b0, l0) = src.read_spec(3, 0).unwrap();
        let (_, b1, _) = src.read_spec(3, 1).unwrap();
        assert_eq!(b0, 0);
        assert_eq!(b1, stride);
        assert!(b0 % crate::quant::VQ_ALIGN == 0 && b1 % crate::quant::VQ_ALIGN == 0);
        assert_eq!(l0, vq_expert_bytes(hidden, moe_inter));
        assert!(l0 <= stride, "useful bytes exceed slot stride");
        // out-of-range guards
        assert!(src.read_spec(2, 0).is_err()); // dense layer
        assert!(src.read_spec(3, n_experts).is_err()); // expert OOB

        // a truncated layer file must fail open()
        std::fs::write(format!("{dir}/L03.i3"), vec![0u8; stride]).unwrap(); // 1 expert, not 4
        assert!(VqExperts::open(&dir, dense, n_layers, n_experts, hidden, moe_inter).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
