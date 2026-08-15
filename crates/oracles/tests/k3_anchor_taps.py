"""The two taps: the values a `register_forward_hook` cannot see, and how each is reached.

Split out of `k3_anchor_driver.py` on 2026-08-15 under the 800-line-per-file gate. Every body
below moved verbatim, comments and measurements with it.

Both write into a `k3_anchor_capture.Capture`, and both exist because the hook mechanism in that
module is blind to them: `_apply_attn_res` READS its `proj` and `norm` modules instead of calling
them, and fla's KDA entry points are free functions on the reference module rather than modules
at all. A `setattr` over each is the only thing that sees the values, and the golden went a day
without the first of them while three comments claimed otherwise.
"""

# The lazily-resolved `torch` the capture side already defines -- imported, never restated, since
# nothing ever rebinds it. `k3_anchor_capture` carries the argument for why it is a proxy.
from k3_anchor_capture import torch


def wrap_attn_res(mod, model, cap, ctx, layers):
    """Capture the AttnRes fold, which no forward hook can see.

    **`_apply_attn_res` never calls its `proj` and `norm` modules** -- it reads
    `proj.weight.squeeze(0)`, `norm.weight` and `norm.variance_epsilon` inline (reference
    `:1075-1088`). A `register_forward_hook` only fires from `Module.__call__`, so the six
    AttnRes modules per layer and the two model-level ones produced NOTHING, and the golden
    contained no `*_res_*` tensor at all while three comments said otherwise. Found by review
    2026-08-11 -- and AttnRes is S2 item 1, so the anchor was missing a fixture for the first
    kernel the port writes.

    It is a module-level free function, and all three call sites (twice per layer, once
    model-level) resolve it from module globals at call time, so one `setattr` catches every
    fold. Folds are named by the identity of the `proj` they were handed, which is what
    distinguishes `self_attention_res` from `mlp_res` without depending on call order.
    """
    owners = {}
    for name, m in model.named_modules():
        if not name.endswith("_res_proj"):
            continue
        base = name[: -len("_proj")]
        if base.startswith("model.output") or any(
            base.startswith(f"model.layers.{i}.") for i in layers
        ):
            owners[id(m)] = base
    orig = mod._apply_attn_res

    def inner(prefix_sum, block_residual, proj, norm):
        if ctx.get("attn_res_mix_normalised"):
            out = _fold_mixing_normalised(prefix_sum, block_residual, proj, norm)
        else:
            out = orig(prefix_sum, block_residual, proj, norm)
        tag = owners.get(id(proj))
        if tag is not None and ctx["capturing"]:
            cap.add(f"{tag}.in.prefix_sum", prefix_sum)
            cap.add(f"{tag}.in.block_residual", block_residual)
            cap.add(f"{tag}.out", out)
        return out

    mod._apply_attn_res = inner
    return orig


def _fold_mixing_normalised(prefix_sum, block_residual, proj, norm):
    """`_apply_attn_res` with `k` mixed instead of `v_float` -- the `AttnResNormalisedValues` body.

    A transcription of the reference's five lines with one substitution, which is the whole point
    of the defect; every other defect here perturbs a weight or a kwarg instead.
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    k = v_float * torch.rsqrt(v_float.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    scores = (k * (norm.weight.float() * proj.weight.squeeze(0).float())).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, k).squeeze(1).to(v.dtype)


def wrap_kda_ops(mod, cap, ctx, layers):
    """Record the KDA operator's own inputs and outputs, which no module hook can see.

    `q`/`k`/`v` after the short convolutions, the log-space gate, `beta`, `A_log`, `dt_bias` in;
    `o` and the recurrent state out. This IS the S2 KDA fixture: everything between these two
    points lives in fla's triton kernel and in no document.

    Installed BEFORE the warm prefill, and capture is gated on `ctx["capturing"]` instead.
    Review 2026-08-11: with the wrapper installed after the warm pass, a `--defect` that only
    sets a kernel kwarg perturbed exactly one call per layer, on a recurrent state the defect had
    never touched -- so `in.initial_state` was bit-identical between the runs BY CONSTRUCTION and
    the state-propagation claim was the one thing those defects could not test.
    """

    def wrap(fn, tag):
        def inner(**kw):
            kw = dict(kw, **ctx["kda_kwargs"])
            if ctx["kda_fp32_island"]:
                # **fla's KDA kernel cannot be compiled for fp64 at all** — it raises
                # `fp_downcast_rounding should be set only for truncating fp conversions` on an
                # internal fp32->fp64 conversion (measured 2026-08-11). So `--dtype float64` runs
                # the whole model in double EXCEPT this operator, which stays fp32 with its inputs
                # cast down and its outputs cast back. That makes the fp64 run a rounding-floor
                # reference for every operator but this one; KDA's own floor comes from
                # `--mode kda-equiv` instead, and would have needed a separate measurement anyway
                # because the kernel returns an fp32 state whatever the input dtype.
                # `q` by name, not "the first kwarg with a dtype". That took whatever the reference
                # happened to pass first -- today `q`, but a revision putting `cu_seqlens` or a
                # position index earlier would cast every KDA output to an INTEGER dtype, and the
                # only symptom would be a wrong floor in a measurement nobody re-derives.
                q = kw.get("q")
                assert q is not None and q.is_floating_point(), (
                    f"the fp32 island reads its dtype from `q`; got {type(q).__name__}. The "
                    f"reference calls this op with all-keyword arguments, so a rename upstream "
                    f"lands here rather than silently picking another kwarg's dtype."
                )
                dt = q.dtype
                kw = {
                    k: (v.float() if hasattr(v, "dtype") and v.is_floating_point() else v)
                    for k, v in kw.items()
                }
                out = fn(**kw)
                out = tuple(o.to(dt) if o is not None else None for o in out)
            else:
                out = fn(**kw)
            if ctx.get("kda_transpose_out_state") and out[1] is not None:
                out = (out[0], out[1].transpose(-1, -2).contiguous())
            sink = ctx.get("equiv_sink")
            if sink is not None:
                which_all = ctx["kda_layer_ids"][ctx["kda_calls"]]
                ctx["kda_calls"] += 1
                sink.setdefault((tag, which_all), []).append(out[0].detach().float().cpu())
                return out
            if ctx["capturing"]:
                # The nth KDA call of the capture pass is the nth KDA layer in index order, which
                # the model guarantees by iterating `self.layers` in order.
                which = ctx["kda_layer_ids"][ctx["kda_calls"]]
                ctx["kda_calls"] += 1
                if which in layers:
                    for k in ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "initial_state"):
                        if kw.get(k) is not None:
                            cap.add(f"model.layers.{which}.kda.{tag}.in.{k}", kw[k])
                    cap.add(f"model.layers.{which}.kda.{tag}.out.o", out[0])
                    if out[1] is not None:
                        cap.add(f"model.layers.{which}.kda.{tag}.out.state", out[1])
            return out

        return inner

    for tag in ("chunk_kda", "fused_recurrent_kda"):
        setattr(mod, tag, wrap(getattr(mod, tag), tag))
