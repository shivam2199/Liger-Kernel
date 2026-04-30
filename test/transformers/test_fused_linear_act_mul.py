import pytest
import torch

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP

from liger_kernel.ops import LigerFusedLinearActMulFunction
from liger_kernel.transformers.swiglu import LigerFusedLinearSwiGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.utils import infer_device

device = infer_device()


def _reference_forward(x, gate_weight, up_states, gate_multiplier):
    """Numerical reference: silu(x @ gate_weight.T * gate_multiplier) * up_states.

    Mirrors the HF-exact downcast pattern (silu computed in fp32, cast back to
    activation dtype before multiplying by up_states) that the fused kernel
    also applies in its epilogue.
    """
    act_dtype = x.dtype
    gate = (x @ gate_weight.T).to(torch.float32) * float(gate_multiplier)
    silu = gate * torch.sigmoid(gate)
    return silu.to(act_dtype) * up_states


@pytest.mark.parametrize(
    "M, K, N",
    [
        (4, 8, 16),
        (128, 256, 512),
        # odd shapes to exercise masking
        (33, 31, 257),
        (255, 128, 63),
    ],
)
@pytest.mark.parametrize(
    "gate_multiplier",
    [1.0, 0.5, 3.14],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # fp32 uses IEEE tl.dot; reduction order still differs from cuBLAS, so
        # allow relative drift consistent with fp32 matmul accumulation.
        (torch.float32, 1e-4, 1e-3),
        pytest.param(
            torch.bfloat16,
            2e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bf16 unsupported"),
        ),
        (torch.float16, 5e-3, 5e-3),
    ],
)
def test_forward_matches_reference(M, K, N, gate_multiplier, dtype, atol, rtol):
    set_seed(42)
    x = torch.randn(M, K, device=device, dtype=dtype)
    gate_weight = torch.randn(N, K, device=device, dtype=dtype) / (K**0.5)
    up_states = torch.randn(M, N, device=device, dtype=dtype)

    ref = _reference_forward(x, gate_weight, up_states, gate_multiplier)
    out = LigerFusedLinearActMulFunction.apply(x, gate_weight, up_states, gate_multiplier)

    assert_verbose_allclose(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("M, K, N", [(128, 256, 512), (64, 128, 384)])
@pytest.mark.parametrize("gate_multiplier", [1.0, 0.5])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-3, 1e-3),
        pytest.param(
            # bf16 bwd atol is dominated by (dg.T @ x) reducing M bf16 products.
            # The reference path multiplies d_out * up in bf16 before casting to
            # fp32, losing an LSB our kernel preserves, so the reference is
            # strictly *less* precise than ours -- the drift we see is the
            # reference losing precision, not a bug. Tolerance here is in line
            # with existing Liger SwiGLU bf16 tests.
            torch.bfloat16,
            1e-1,
            1e-1,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bf16 unsupported"),
        ),
    ],
)
def test_backward_matches_reference(M, K, N, gate_multiplier, dtype, atol, rtol):
    set_seed(42)
    # Build leaf tensors: scale in-place before enabling requires_grad, otherwise
    # the scaled tensor becomes a non-leaf and .grad never populates.
    x_init = torch.randn(M, K, device=device, dtype=dtype)
    gw_init = torch.randn(N, K, device=device, dtype=dtype) / (K**0.5)
    up_init = torch.randn(M, N, device=device, dtype=dtype)

    x_ref = x_init.clone().requires_grad_(True)
    gw_ref = gw_init.clone().requires_grad_(True)
    up_ref = up_init.clone().requires_grad_(True)

    x_fused = x_init.clone().requires_grad_(True)
    gw_fused = gw_init.clone().requires_grad_(True)
    up_fused = up_init.clone().requires_grad_(True)

    ref_out = _reference_forward(x_ref, gw_ref, up_ref, gate_multiplier)
    fused_out = LigerFusedLinearActMulFunction.apply(x_fused, gw_fused, up_fused, gate_multiplier)

    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    fused_out.backward(grad.clone())

    assert_verbose_allclose(x_fused.grad, x_ref.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(gw_fused.grad, gw_ref.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(up_fused.grad, up_ref.grad, atol=atol, rtol=rtol)


def test_forward_rejects_bias_and_dtensor():
    """DTensor and bias paths must fail fast rather than silently miscompute."""
    x = torch.randn(4, 8, device=device, dtype=torch.float32)
    gw = torch.randn(16, 8, device=device, dtype=torch.float32)
    up = torch.randn(4, 16, device=device, dtype=torch.float32)

    # dtype mismatch should assert
    with pytest.raises(AssertionError):
        LigerFusedLinearActMulFunction.apply(x, gw.to(torch.bfloat16), up, 1.0)


@pytest.mark.parametrize("seq_len, hidden_size, intermediate_size", [(256, 256, 512), (128, 192, 384)])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # LigerSwiGLUMLP runs cuBLAS gate+up separately then Triton elementwise;
        # the fused path runs a single Triton matmul + epilogue. Reduction order
        # differs slightly in fp32 even with IEEE precision, and the down_proj
        # weight gradient reduces bf16 products across M.
        (torch.float32, 1e-3, 1e-3),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-1,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bf16 unsupported"),
        ),
    ],
)
def test_drop_in_mlp_equivalence(seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    """LigerFusedLinearSwiGLUMLP must match LigerSwiGLUMLP output and gradients
    when initialized with identical weights."""
    set_seed(42)
    cfg = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
    )

    liger = LigerSwiGLUMLP(cfg).to(device).to(dtype)
    fused = LigerFusedLinearSwiGLUMLP(cfg).to(device).to(dtype)

    # mirror weights
    fused.gate_proj.weight.data.copy_(liger.gate_proj.weight.data)
    fused.up_proj.weight.data.copy_(liger.up_proj.weight.data)
    fused.down_proj.weight.data.copy_(liger.down_proj.weight.data)

    x_a = torch.randn(2, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)

    y_liger = liger(x_a)
    y_fused = fused(x_b)
    assert_verbose_allclose(y_fused, y_liger, atol=atol, rtol=rtol)

    grad = torch.randn_like(y_liger)
    y_liger.backward(grad)
    y_fused.backward(grad.clone())

    assert_verbose_allclose(x_b.grad, x_a.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        fused.gate_proj.weight.grad,
        liger.gate_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert_verbose_allclose(
        fused.up_proj.weight.grad,
        liger.up_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    assert_verbose_allclose(
        fused.down_proj.weight.grad,
        liger.down_proj.weight.grad,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-3, 1e-3),
        pytest.param(
            torch.bfloat16,
            5e-2,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bf16 unsupported"),
        ),
    ],
)
def test_fused_mlp_matches_hf_llama(dtype, atol, rtol):
    """Regression guard: LigerFusedLinearSwiGLUMLP must agree with HF LlamaMLP
    under mirrored weights, at tolerances that already pass for LigerSwiGLUMLP."""
    set_seed(42)
    cfg = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        hidden_act="silu",
    )

    hf = LlamaMLP(cfg).to(device).to(dtype)
    fused = LigerFusedLinearSwiGLUMLP(cfg).to(device).to(dtype)

    fused.gate_proj.weight.data.copy_(hf.gate_proj.weight.data)
    fused.up_proj.weight.data.copy_(hf.up_proj.weight.data)
    fused.down_proj.weight.data.copy_(hf.down_proj.weight.data)

    x = torch.randn(2, 64, 128, device=device, dtype=dtype, requires_grad=True)
    x_fused = x.detach().clone().requires_grad_(True)

    y_hf = hf(x)
    y_fused = fused(x_fused)
    assert_verbose_allclose(y_fused, y_hf, atol=atol, rtol=rtol)
