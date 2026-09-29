"""Muon (MomentUm Orthogonalized by Newton-schulz), pure torch.

Implemented here rather than pip-installed so the backends keep the same
dependency set as the SD1.5/SDXL runs.

Muon only applies to matrix-shaped parameters: it orthogonalizes the momentum
buffer with a quintic Newton-Schulz iteration before stepping, which makes the
update spectrally normalized rather than element-wise scaled. Everything that
is not a 2D weight matrix -- biases, norm gains, embeddings, the output head --
has no meaningful singular-value structure, so those are stepped with ordinary
AdamW in the same optimizer. That split is part of the method, not a shortcut:
running Muon on 1D tensors is ill-defined, and running it on embeddings and the
head is what the reference implementation explicitly avoids.

Reference: Keller Jordan et al., "Muon: An optimizer for hidden layers in
neural networks" (2024).
"""

from __future__ import annotations

import torch


# Quintic coefficients tuned so the iteration's fixed point is close to the
# sign of the singular values, converging fast for the first few steps.
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


@torch.no_grad()
def zeropower_via_newtonschulz5(matrix, steps=5, eps=1e-7):
    """Approximate the orthogonal factor of ``matrix`` (i.e. U @ V^T of its SVD).

    Runs in bfloat16 on purpose -- the iteration is numerically forgiving and
    this keeps the extra memory traffic small on a 1B+ parameter model.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Newton-Schulz needs a 2D tensor, got shape {tuple(matrix.shape)}")
    x = matrix.bfloat16()
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a = x @ x.T
        b = _NS_B * a + _NS_C * (a @ a)
        x = _NS_A * x + b @ x
    if transposed:
        x = x.T
    return x.to(matrix.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2D parameters with an AdamW fallback for everything else.

    Args:
        params: iterable of parameters, or of param groups.
        lr: Muon learning rate for the matrix parameters.
        momentum: heavy-ball coefficient on the pre-orthogonalization buffer.
        nesterov: use the Nesterov form of the momentum lookahead.
        ns_steps: Newton-Schulz iterations per step.
        weight_decay: decoupled weight decay, applied in both branches.
        adamw_lr: learning rate for the non-matrix parameters. Defaults to
            ``lr`` when not given, but a smaller value is common because the
            Muon branch is spectrally normalized and the AdamW branch is not.
        adamw_betas, adamw_eps: standard AdamW hyperparameters.
    """

    def __init__(
        self,
        params,
        lr=2e-5,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,
        adamw_lr=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            adamw_lr=lr if adamw_lr is None else adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _use_muon(param):
        # Conv kernels are folded to (out, -1); anything 1D is AdamW's job.
        return param.ndim >= 2

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            adamw_lr = group["adamw_lr"]
            beta1, beta2 = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[param]

                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)

                if self._use_muon(param):
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.lerp_(grad, 1.0 - momentum)
                    update = grad.lerp(buf, momentum) if nesterov else buf

                    original_shape = update.shape
                    flat = update.reshape(original_shape[0], -1)
                    flat = zeropower_via_newtonschulz5(flat, steps=ns_steps)
                    # Keep the update's RMS comparable across shapes, so one lr
                    # works for tall and wide matrices alike.
                    scale = max(1.0, flat.size(0) / flat.size(1)) ** 0.5
                    param.add_(flat.reshape(original_shape), alpha=-lr * scale)
                else:
                    if "exp_avg" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(grad)
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.lerp_(grad, 1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    bias1 = 1.0 - beta1 ** state["step"]
                    bias2 = 1.0 - beta2 ** state["step"]
                    denom = (exp_avg_sq / bias2).sqrt_().add_(adamw_eps)
                    param.addcdiv_(exp_avg / bias1, denom, value=-adamw_lr)

        return loss


def build_optimizer(name, params, learning_rate, weight_decay=0.0, **muon_kwargs):
    """Factory shared by the training scripts.

    Returns (optimizer, description) so the run can record which optimizer and
    which parameter split actually applied.
    """
    params = [p for p in params if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters to optimize")

    if name == "adamw":
        return (
            torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay),
            {"optimizer": "adamw", "lr": learning_rate, "weight_decay": weight_decay},
        )
    if name == "muon":
        matrix = sum(1 for p in params if Muon._use_muon(p))
        optimizer = Muon(
            params, lr=learning_rate, weight_decay=weight_decay, **muon_kwargs
        )
        return optimizer, {
            "optimizer": "muon",
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "muon_params": matrix,
            "adamw_fallback_params": len(params) - matrix,
            **{k: v for k, v in muon_kwargs.items()},
        }
    raise ValueError(f"Unknown optimizer: {name!r} (expected 'adamw' or 'muon')")
