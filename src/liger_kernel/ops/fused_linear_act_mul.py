"""
Fused linear + SiLU-multiplier + up-gating for SwiGLU MLPs.

Computes y = silu(x @ Wg.T * gate_multiplier) * up_states in a single
Triton matmul kernel whose epilogue applies the activation and the up-gating
multiply while the accumulator is still in fp32. This eliminates the HBM
round-trip of the (M, N) gate pre-activation that LigerSiLUMulFunction
otherwise requires.

Shapes (standard M, K, N matmul convention):
    x:          (M, K)    activation dtype       M = T (tokens), K = H (hidden)
    gate_weight:(N, K)    nn.Linear.weight shape N = I (intermediate)
    up_states:  (M, N)    activation dtype
    out:        (M, N)    activation dtype

Scope of this module (PR #1):
    - hidden_act="silu" only
    - no DTensor (assert against DTensor inputs)
    - no bias on gate_weight
    - no @triton.autotune (deferred to follow-up PR)
    - backward computes (dg, d_up) in one Triton kernel, delegates
      (dx, dWg) to torch.matmul
"""

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous

# HF-exact SwiGLU numerics require downcasting silu(gate * m) to the activation
# dtype BEFORE multiplying by up_states. This matches the pattern Unsloth uses
# in fast_lora.py (kernels/swiglu.py line 51 "Exact copy from HF"). Diverging
# from this causes visible drift in bf16 tests.


@triton.jit
def _fused_linear_silu_mul_fwd_kernel(
    x_ptr,
    w_ptr,
    up_ptr,
    g_ptr,
    out_ptr,
    M,
    K,
    N,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_um,
    stride_un,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    gate_multiplier,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_bn[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        x_blk = tl.load(x_ptrs, mask=k_mask, other=0.0)
        w_blk = tl.load(w_ptrs, mask=k_mask, other=0.0)
        acc = tl.dot(x_blk, tl.trans(w_blk), acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    g = acc * gate_multiplier
    sig = tl.sigmoid(g)
    silu_fp32 = g * sig

    store_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    store_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (store_m[:, None] < M) & (store_n[None, :] < N)

    g_ptrs = g_ptr + store_m[:, None] * stride_gm + store_n[None, :] * stride_gn
    tl.store(g_ptrs, g.to(g_ptr.dtype.element_ty), mask=out_mask)

    up_ptrs = up_ptr + store_m[:, None] * stride_um + store_n[None, :] * stride_un
    up_blk = tl.load(up_ptrs, mask=out_mask, other=0.0)

    silu_cast = silu_fp32.to(up_blk.dtype)
    out = silu_cast * up_blk

    out_ptrs = out_ptr + store_m[:, None] * stride_om + store_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=out_mask)


@triton.jit
def _fused_silu_mul_bwd_kernel(
    dout_ptr,
    g_ptr,
    up_ptr,
    dg_ptr,
    dup_ptr,
    M,
    N,
    stride_dom,
    stride_don,
    stride_gm,
    stride_gn,
    stride_um,
    stride_un,
    stride_dgm,
    stride_dgn,
    stride_dum,
    stride_dun,
    gate_multiplier,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    dout = tl.load(
        dout_ptr + offs_m[:, None] * stride_dom + offs_n[None, :] * stride_don,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    g = tl.load(
        g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        up_ptr + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    sig = tl.sigmoid(g)
    d_silu_dg = sig * (1.0 + g * (1.0 - sig))
    silu = g * sig

    dup = dout * silu
    dg = dout * up * d_silu_dg * gate_multiplier

    tl.store(
        dup_ptr + offs_m[:, None] * stride_dum + offs_n[None, :] * stride_dun,
        dup.to(dup_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        dg_ptr + offs_m[:, None] * stride_dgm + offs_n[None, :] * stride_dgn,
        dg.to(dg_ptr.dtype.element_ty),
        mask=mask,
    )


def fused_linear_silu_mul_forward(x, gate_weight, up_states, gate_multiplier: float = 1.0):
    assert x.ndim == 2, f"x must be 2D (M, K), got {x.shape}"
    assert gate_weight.ndim == 2, f"gate_weight must be 2D (N, K), got {gate_weight.shape}"
    assert up_states.ndim == 2, f"up_states must be 2D (M, N), got {up_states.shape}"
    assert x.shape[1] == gate_weight.shape[1], "x[K] must match gate_weight[K]"
    assert up_states.shape[0] == x.shape[0], "up_states[M] must match x[M]"
    assert up_states.shape[1] == gate_weight.shape[0], "up_states[N] must match gate_weight[N]"
    assert x.dtype == gate_weight.dtype == up_states.dtype, "dtypes must match"

    M, K = x.shape
    N = gate_weight.shape[0]

    g = torch.empty((M, N), dtype=x.dtype, device=x.device)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 32, 8
    num_warps, num_stages = 4, 3

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _fused_linear_silu_mul_fwd_kernel[grid](
        x,
        gate_weight,
        up_states,
        g,
        out,
        M,
        K,
        N,
        x.stride(0),
        x.stride(1),
        gate_weight.stride(0),
        gate_weight.stride(1),
        up_states.stride(0),
        up_states.stride(1),
        g.stride(0),
        g.stride(1),
        out.stride(0),
        out.stride(1),
        float(gate_multiplier),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, g


def fused_linear_silu_mul_backward(dout, x, gate_weight, up_states, g, gate_multiplier: float = 1.0):
    M, N = g.shape

    dg = torch.empty((M, N), dtype=dout.dtype, device=dout.device)
    d_up = torch.empty((M, N), dtype=dout.dtype, device=dout.device)

    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fused_silu_mul_bwd_kernel[grid](
        dout,
        g,
        up_states,
        dg,
        d_up,
        M,
        N,
        dout.stride(0),
        dout.stride(1),
        g.stride(0),
        g.stride(1),
        up_states.stride(0),
        up_states.stride(1),
        dg.stride(0),
        dg.stride(1),
        d_up.stride(0),
        d_up.stride(1),
        float(gate_multiplier),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    dx = torch.matmul(dg, gate_weight)
    d_gate_weight = torch.matmul(dg.transpose(0, 1), x)

    return dx, d_gate_weight, d_up


class LigerFusedLinearActMulFunction(torch.autograd.Function):
    """
    Fused linear + SiLU + multiplier + up-gating.

    Forward:  y = silu(x @ gate_weight.T * gate_multiplier) * up_states
    Backward: returns (dx, d_gate_weight, d_up_states, None)

    See module docstring for scope limits in this PR.
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, gate_weight, up_states, gate_multiplier: float = 1.0):
        assert not isinstance(x, torch.distributed.tensor.DTensor), (
            "DTensor inputs not supported in LigerFusedLinearActMulFunction; "
            "use LigerSiLUMulFunction for DTensor paths."
        )
        assert not isinstance(gate_weight, torch.distributed.tensor.DTensor)
        assert not isinstance(up_states, torch.distributed.tensor.DTensor)

        gate_multiplier = float(gate_multiplier)
        ctx.gate_multiplier = gate_multiplier
        ctx.input_shape = x.shape

        x_2d = x.reshape(-1, x.shape[-1])
        up_2d = up_states.reshape(-1, up_states.shape[-1])

        out, g = fused_linear_silu_mul_forward(x_2d, gate_weight, up_2d, gate_multiplier)

        ctx.save_for_backward(x_2d, gate_weight, up_2d, g)
        return out.view(*x.shape[:-1], gate_weight.shape[0])

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dout):
        x_2d, gate_weight, up_2d, g = ctx.saved_tensors
        gate_multiplier = ctx.gate_multiplier

        dout_2d = dout.reshape(-1, dout.shape[-1])
        dx_2d, d_gate_weight, d_up_2d = fused_linear_silu_mul_backward(
            dout_2d,
            x_2d,
            gate_weight,
            up_2d,
            g,
            gate_multiplier,
        )

        dx = dx_2d.view(*ctx.input_shape)
        d_up = d_up_2d.view(*ctx.input_shape[:-1], gate_weight.shape[0])
        return dx, d_gate_weight, d_up, None
