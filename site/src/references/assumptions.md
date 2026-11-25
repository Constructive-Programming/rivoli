# Assumptions and limits

rivoli's guarantees hold on the machine it is designed for, under the conditions below.
Outside them a run may be refused, slower, or not reproducible. It should never be
silently wrong.

## The machine

- **Unified memory.** The GPU reads system RAM directly through ROCm. A discrete GPU with
  its own VRAM is not a target: the design has no host-to-device copy to hide.
- **ROCm and HIP.** There is one backend. Other vendors and APIs are out of scope by
  choice ([P3](../principles.md#p3--maximize-hardware-features)).
- **Fast local NVMe.** Streamed weights are read from local NVMe. An artifact on a network
  filesystem works, but its speed figures are not meaningful.

## Memory

- **The budget is measured at run time.** What is resident follows from free memory when
  the run starts: other tenants, the KV cache at the configured context, and scratch.
- **Below the floor, the run is refused.** If the budget cannot hold the minimum working
  set, rivoli refuses with the arithmetic rather than degrading.
- **Other GPU tenants change the numbers.** A second process using GPU memory shrinks the
  pin, and speed measurements taken while it runs are discarded.

## Output

- **The memory knob does not change text.** `--max-mem` and cache policy move speed only
  ([P4](../principles.md#p4--the-memory-knob-trades-speed-never-text)). Where a format is
  chosen by residency, that is called out as a defect, not a feature.
- **Not every configuration is bit-reproducible.** Some streaming arms vary slightly
  between runs of the same input. Quality comparisons measure that noise floor first, and
  differences below it are not reported.
- **Parity means greedy decoding.** "Token-identical to the reference" is checked with
  greedy sampling on a pinned prompt, for a stated number of tokens.

## Models

- **The artifact decides.** Each model is converted once into a rivoli artifact, which
  names its architecture. Flags that the architecture does not support are refused at
  startup with the reason.
- **Text only.** Vision towers and multi-token-prediction heads are not run, even when the
  checkpoint includes them.
