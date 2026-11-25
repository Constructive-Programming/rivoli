# Bibliography

The papers, formats and evidence behind rivoli's design. Each entry gives the authors,
the title, the venue and year, and a note on why it matters here.

## Offloading and streaming

- Ying Sheng et al., [FlexGen: High-Throughput Generative Inference of Large Language
  Models with a Single GPU](https://arxiv.org/abs/2303.06865) (ICML 2023): offloading
  weights across GPU, CPU and disk, with scheduling chosen as an optimization problem.
- Keivan Alizadeh et al., [LLM in a flash: Efficient Large Language Model Inference with
  Limited Memory](https://arxiv.org/abs/2312.11514) (2023): reading weights from flash on
  demand, and why the read size and pattern decide the throughput.
- Yixin Song et al., [PowerInfer: Fast Large Language Model Serving with a Consumer-grade
  GPU](https://arxiv.org/abs/2312.12456) (2023): splitting hot and cold neurons between GPU
  and CPU.
- Artyom Eliseev & Denis Mazur, [Fast Inference of Mixture-of-Experts Language Models with
  Offloading](https://arxiv.org/abs/2312.17238) (2023): expert caching and speculative
  expert loading on memory-limited hardware.
- Samyam Rajbhandari et al., [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale
  Deep Learning](https://arxiv.org/abs/2104.07857) (SC 2021): NVMe as a tier in the memory
  hierarchy for model state.
- Reza Yazdani Aminabadi et al., [DeepSpeed Inference: Enabling Efficient Inference of
  Transformer Models at Unprecedented Scale](https://arxiv.org/abs/2207.00032) (SC 2022):
  inference with weights held outside GPU memory.

## Caching and the cost model

- László A. Belady, [A study of replacement algorithms for a virtual-storage
  computer](https://doi.org/10.1147/sj.52.0078) (IBM Systems Journal, 1966): the optimal
  replacement policy. For a dense model read in the same order every token, it reduces to
  a static prefix, which is the shape of rivoli's pin.
- Samuel Williams, Andrew Waterman & David Patterson, [Roofline: An Insightful Visual
  Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785)
  (CACM, 2009): bandwidth-bound versus compute-bound work.
- Reiner Pope et al., [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
  (2022): why decode is limited by memory traffic, the basis for "bytes per token are the
  currency".

## Mixture of experts and attention

- Noam Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (ICLR 2017): the routed
  expert layer that makes most weights cold on any given token.
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (2024):
  multi-head latent attention (used by GLM-5.2) and fine-grained experts.
- Songlin Yang, Jan Kautz & Ali Hatamizadeh, [Gated Delta Networks: Improving Mamba2 with
  Delta Rule](https://arxiv.org/abs/2412.06464) (ICLR 2025): the linear-attention layers in
  the hybrid models rivoli ports.
- Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness](https://arxiv.org/abs/2205.14135) (NeurIPS 2022): attention as an
  IO-scheduling problem.

## Number formats and quantization

- Paulius Micikevicius et al., [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
  (2022): the e4m3 and e5m2 formats behind fp8 artifacts.
- Bita Darvish Rouhani et al., [Microscaling Data Formats for Deep
  Learning](https://arxiv.org/abs/2310.10537) (2023): the MX block formats (MXFP4, MXFP8)
  that several checkpoints ship in.
- Elias Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative
  Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (ICLR 2023): post-training
  integer quantization.
- Vage Egiazarian et al., [Extreme Compression of Large Language Models via Additive
  Quantization](https://arxiv.org/abs/2401.06118) (ICML 2024) and Albert Tseng et al.,
  [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice
  Codebooks](https://arxiv.org/abs/2402.04396) (ICML 2024): codebook (vector)
  quantization at around 3 bits per weight.

## Decoding

- Yaniv Leviathan, Matan Kalman & Yossi Matias, [Fast Inference from Transformers via
  Speculative Decoding](https://arxiv.org/abs/2211.17192) (ICML 2023) and Charlie Chen et
  al., [Accelerating Large Language Model Decoding with Speculative
  Sampling](https://arxiv.org/abs/2302.01318) (2023): drafting tokens cheaply and verifying
  them in one pass, a traffic lever under P5.
- Woosuk Kwon et al., [Efficient Memory Management for Large Language Model Serving with
  PagedAttention](https://arxiv.org/abs/2309.06180) (SOSP 2023): KV-cache memory
  management in serving.

## Measurement

- Donald J. Schuirmann, [A comparison of the Two One-Sided Tests Procedure and the Power
  Approach for assessing the equivalence of average
  bioavailability](https://doi.org/10.1007/BF01068419) (Journal of Pharmacokinetics and
  Biopharmaceutics, 1987): equivalence testing. rivoli's quality gates pass only when the
  interval lies inside a band fixed in advance; an interval that straddles zero is
  inconclusive.

## Repository evidence

- [Principles](https://github.com/Constructive-Programming/rivoli/blob/main/docs/reference/principles.md):
  P1 to P7 as confirmed by the owner.
- [Architecture](https://github.com/Constructive-Programming/rivoli/blob/main/docs/reference/architecture.md):
  the crate layering, the residency contract and the invariant registry.
- [Kernel inventory](https://github.com/Constructive-Programming/rivoli/blob/main/docs/reference/kernel-inventory.md):
  every device kernel and its launcher.
- [How to measure](https://github.com/Constructive-Programming/rivoli/blob/main/docs/measurement/how-to-measure.md):
  the measurement rules every cited number follows.
- [Gate red proofs](https://github.com/Constructive-Programming/rivoli/blob/main/docs/measurement/gate-red-proofs.md):
  the recorded runs in which each gate was shown to fail.
- [Docs index](https://github.com/Constructive-Programming/rivoli/blob/main/docs/00-orientation/INDEX.md):
  every investigation with its status and verdict.
