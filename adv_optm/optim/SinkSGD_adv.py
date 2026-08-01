import torch

import math

from ..util import param_update
from ..util.factorization_util import _get_effective_shape, _reconstruct_state, _factorize_state
from ..util.OrthoGrad import _orthogonalize_gradient
from ..util.scaled_optm import scale_update, is_spectral, init_spectral_norm
from ..util.centered_decay import _init_anchor, dequantize_anchor
from ..util.state_util import init_state_tensor, get_state, set_state, upcast_grad_for_precision
from ..util.sinkhorn import apply_sr_sinkhorn, get_sinkhorn_wd_scaler
from ..util.signed_util import get_signsgd_wd_target

class SinkSGD_adv(torch.optim.Optimizer):
    """
    Implements an advanced Stochastic Gradient Descent (SGD) with Sinkhorn Iterative
    Normalization (SinkSGD). The gradient/update matrix is driven toward a fixed
    point where every row and column has uniform L2 magnitude (multi-normalization).
    This advanced version adds optional low-rank factorization of optimizer states
    (SMMF, state_precision='factored'), OrthoGrad, Nesterov momentum, SNR
    preconditioning, centered/cautious/geometric weight decay, spectral
    normalization, and torch.compile support.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): learning rate (default: 1e-3).
        momentum (float): momentum factor in [0.0, 1.0] (default: 0.0).
        weight_decay (float): weight decay (L2 penalty or decoupled) (default: 0.0).
        sinkhorn_iterations (int): number of alternating row/column L2
            normalization iterations applied by SR-Sinkhorn (default: 5).
        orthogonal_sinkhorn (bool): projects the update orthogonal to the
            parameter during each Sinkhorn normalization step (default: False).
        normed_momentum (bool): normalize the gradient *before* it enters the
            momentum buffer ("Normalization then Momentum") instead of after
            (default: False).
        snr_cond (bool): SNR preconditioning. Scales the update by the empirical
            signal-to-noise ratio of the momentum buffer before applying the
            atan bounding. Requires normed_momentum and momentum > 0
            (default: False).
        nesterov (bool): enables Nesterov momentum. Only applicable when momentum
            is non-zero (default: False).
        nesterov_coef (float | None): Nesterov lookahead coefficient. Defaults to
            the momentum value when None (default: None).
        geometric_wd (bool): uses a structural weight-decay scaler that penalizes
            dominant rows/columns more heavily (default: False).
        cautious_wd (bool): weight decay is applied only to coordinates where the
            sign of the parameter and the sign of the optimizer update align
            (default: False).
        stochastic_rounding (bool): stochastic rounding for BF16 parameter updates
            (default: True).
        orthogonal_gradient (str): OrthoGrad mode. One of 'disabled', 'flattened'
            (vectorized projection), or 'iterative' (alternating row/column
            projection) (default: 'disabled').
        spectral_normalization (bool): scales the update so the parameter's
            spectral norm is preserved (default: False).
        centered_wd (float): Centered Weight Decay coefficient. Weights decay
            toward their initial values (anchors) instead of zero. Can be combined
            with standard weight decay (default: 0.0).
        centered_wd_mode (str): quantization format for anchor weights to save
            VRAM. One of 'full', 'float8', 'int8', 'int4' (default: 'float8').
        state_precision (str): precision for optimizer states. One of 'auto'
            (parameter precision), 'fp32', 'factored' (SMMF low-rank FP32),
            'bf16_sr' (BF16 with stochastic rounding), 'fp16', 'int8_sr'
            (blockwise INT8 with stochastic rounding) (default: 'auto').
        nnmf_factor (bool): legacy alias; forces state_precision='factored' when
            True (default: False).
        vector_reshape (bool): treat 1D vectors as 2D matrices so low-rank
            factorization / Sinkhorn normalization can be applied (default: False).
        compiled_optimizer (bool): compile the core step function with torch.compile
            for faster execution (default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        # Sinkhorn Iterative Normalization
        sinkhorn_iterations: int = 5,
        orthogonal_sinkhorn: bool = False,
        # Normalization then Momentum
        normed_momentum: bool = False,
        # SNR Precondition (requires normed_momentum)
        snr_cond: bool = False,
        # Nesterov Momentum
        nesterov: bool = False,
        nesterov_coef: float | None = None,
        # weight decay features
        geometric_wd: bool = False,
        cautious_wd: bool = False,
        # Stochastic Rounding for BF16
        stochastic_rounding: bool = True,
        # OrthoGrad
        orthogonal_gradient: str = 'disabled', # 'flattened', 'iterative'
        # Spectral Normed Optimizer
        spectral_normalization: bool = False,
        # Centered WD
        centered_wd: float = 0.0,
        centered_wd_mode: str = 'float8',
        # States precision
        state_precision: str = "auto",
        # SMMF factorization
        nnmf_factor: bool = False,
        vector_reshape: bool = False,
        # torch.compile
        compiled_optimizer: bool = False,
    ):
        if not (lr >= 0.0):
            raise ValueError(f"Learning-rate should be >= 0.0. Got {lr}")
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(f"Momentum should be in [0.0, 1.0]. Got {momentum}")
        if not (weight_decay >= 0.0):
            raise ValueError(f"Weight-decay should be >= 0.0. Got {weight_decay}")
        if nesterov_coef is not None and not (0.0 <= nesterov_coef <= 1.0):
            raise ValueError(f"nesterov_coef should be in [0.0, 1.0] or None. Got {nesterov_coef}")
        if sinkhorn_iterations < 0:
            raise ValueError(f"sinkhorn_iterations should be >= 0. Got {sinkhorn_iterations}")
        if orthogonal_gradient not in ('disabled', 'flattened', 'iterative'):
            raise ValueError(f"orthogonal_gradient must be one of 'disabled', 'flattened', 'iterative'. Got {orthogonal_gradient}")
        if snr_cond and (not normed_momentum or momentum <= 0.0):
            raise NotImplementedError("snr_cond is intended to be used with normed_momentum and momentum > 0.")

        state_precision = state_precision.lower()
        valid_precisions = {"auto", "fp32", "factored", "bf16_sr", "fp16", "int8_sr"}
        if state_precision not in valid_precisions:
            raise ValueError(f"state_precision must be one of {valid_precisions}. Got {state_precision}")

        if nnmf_factor:
            state_precision = "factored"

        defaults = {
            "lr": lr, "momentum": momentum,
            "weight_decay": weight_decay, "nesterov": nesterov, "nesterov_coef": nesterov_coef, "normed_momentum": normed_momentum, "snr_cond": snr_cond,
            "geometric_wd": geometric_wd, "cautious_wd": cautious_wd,
            "orthogonal_gradient": orthogonal_gradient, 
            "compiled_optimizer": compiled_optimizer,
            "sinkhorn_iterations": sinkhorn_iterations,
            "orthogonal_sinkhorn": orthogonal_sinkhorn,
            "spectral_normalization": spectral_normalization,
            "centered_wd": centered_wd, "centered_wd_mode": centered_wd_mode,
            "state_precision": state_precision,
            "nnmf_factor": nnmf_factor, "vector_reshape": vector_reshape
        }
        self.stochastic_rounding = stochastic_rounding
        self._init_lr = lr if lr > 0 else 1
        super().__init__(params, defaults)

        if self.stochastic_rounding:
            devices = {p.device for group in self.param_groups for p in group['params'] if p.dtype == torch.bfloat16}
            for device in devices:
                param_update.set_seed(device)

        self.init_step()

        # Initialize compiled function (by parameter shape)
        self._compiled_step_fns = {}

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        param_update.post_process_loaded_state(self)
        self.init_step()

    @property
    def supports_fused_back_pass(self):
        return True

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def init_step(self):
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                self.__init_state(p, group)

    @torch.no_grad()
    def __init_state(self, p, group):
        state = self.state[p]
        # State Initialization
        if 'step' not in state:
            state['step'] = 0

            req_precision = group['state_precision']
            # Keep the is_vector definition identical to the one used at step time
            # (see _step_parameter) so state allocation matches the runtime branches.
            # Note: 0-dim (scalar) tensors are ALWAYS treated as vectors since a
            # meaningful 2D reshape / Sinkhorn normalization does not exist for them.
            is_vector = (
                len(p.shape) < 2 and not group['vector_reshape']
            ) or len(p.shape) == 0 or getattr(p, '_is_dora_scale', False) or getattr(p, 'is_vector', False)

            state['factored'] = req_precision == 'factored' and not is_vector

            actual_precision = 'auto' if req_precision == 'factored' else req_precision
            group['actual_state_precision'] = actual_precision

            dtype = torch.float32 if (state['factored'] or req_precision == 'factored') else p.dtype
            device = p.device

            if state['factored']:
                # effective_shape is needed even when momentum == 0 because the
                # factored step path reshapes the gradient with it before cloning.
                state['effective_shape'] = _get_effective_shape(p.numel())

                if group['momentum'] != 0:
                    d1, d2 = state['effective_shape']

                    state['mu_b_nmf'] = torch.zeros(d1, device=device, dtype=torch.float32)
                    state['mv_b_nmf'] = torch.zeros(d2, device=device, dtype=torch.float32)
                    packed_d2 = (d2 + 7) // 8
                    state['sign'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=device)
                    state['shifter'] = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device, dtype=torch.uint8)
            elif group['momentum'] != 0:
                init_state_tensor(state, 'momentum_buffer', p.shape, actual_precision, p.device, dtype)

            if group.get('spectral_normalization', False) and is_spectral(p):
                init_spectral_norm(state, p)

            _init_anchor(p, state, group)

    @torch.no_grad()
    def step_parameter(self, p: torch.Tensor, group: dict, i: int | None = None):
        if p.grad is None:
            return

        grad = p.grad
        # Defensive copy: the step function applies in-place normalization/sign to
        # the gradient. For fp32 grads upcast_grad_for_precision returns the same
        # buffer, so clone here to avoid ever mutating the user's p.grad.
        if grad.dtype == torch.float32:
            grad = grad.clone()
        state = self.state[p]
        self.__init_state(p, group)

        step_size = group['lr']

        random_int_tensor = None
        random_int_state_tensor = None

        if group.get('compiled_optimizer', False):
            step_size = torch.as_tensor(step_size)
            # Only momentum states need stochastic-rounded storage; gate the RNG
            # draws on the exact conditions that consume them so the compiled and
            # uncompiled paths keep identical (deterministic) RNG streams.
            # Draw the state SR tensor FIRST and the parameter SR tensor SECOND,
            # mirroring the uncompiled path where set_state() draws before
            # apply_parameter_update(). The state and parameter must also use
            # INDEPENDENT noise tensors (the uncompiled path draws them
            # separately); reusing one tensor for both would both corrupt the
            # stream parity and correlate the two rounding operations.
            has_state = group.get('momentum', 0) > 0 and not state.get('factored', False)
            if has_state:
                if group['actual_state_precision'] == 'bf16_sr':
                    random_int_state_tensor = param_update._get_random_int_for_sr(p)
                elif group['actual_state_precision'] == 'int8_sr':
                    random_int_state_tensor = param_update._get_random_int_for_8bit_sr(p)
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                random_int_tensor = param_update._get_random_int_for_sr(p)
            # Cache compiled function per-shape/dtype/device
            cache_key = (p.shape, p.dtype, p.device, state.get('factored', False))
            if cache_key not in self._compiled_step_fns:
                self._compiled_step_fns[cache_key] = torch.compile(
                    self._step_parameter,
                    fullgraph=True,
                    dynamic=False
                )
            step_param_fn = self._compiled_step_fns[cache_key]
        else:
            step_param_fn = self._step_parameter

        step_param_fn(p, grad, state, group, step_size, random_int_tensor, random_int_state_tensor)

        state['step'] += 1

    def _step_parameter(self, p, grad, state, group, step_size, random_int_tensor, random_int_state_tensor):
        grad = upcast_grad_for_precision(grad, state, group['state_precision'])
        # NOTE: fp32 gradients were already cloned in step_parameter, so the
        # in-place normalization/sign operations below never touch the user's p.grad.
        # Keep the is_vector definition identical to the one used in __init_state
        # so state allocation matches the runtime branches. 0-dim (scalar)
        # tensors are always treated as vectors (see __init_state).
        is_vector = (
            grad.ndim < 2 and not group.get('vector_reshape', False)
        ) or grad.ndim == 0 or getattr(p, '_is_dora_scale', False) or getattr(p, 'is_vector', False)
        sinkhorn_iterations = group['sinkhorn_iterations']
        orthogonal_sinkhorn = group['orthogonal_sinkhorn']

        momentum = group['momentum']
        normed_mt = group.get('normed_momentum', False)
        nesterov = group['nesterov']
        nesterov_coef = group.get('nesterov_coef', None)
        # snr_cond requires normalized momentum; the momentum > 0 guard guarantees
        # vt_row/vt_col/denom are always populated when snr_cond is active.
        snr_cond = group.get('snr_cond', False) and normed_mt and momentum > 0

        vt_row = None
        vt_col = None
        denom = None

        wd_scaler = None
        wd_target = None
        cwd_target = None

        grad = _orthogonalize_gradient(p, grad, group["orthogonal_gradient"])

        if normed_mt:
            if not is_vector:
                # Sinkhorn iterative normalization
                grad = apply_sr_sinkhorn(grad, iters=sinkhorn_iterations, p=p, ortho_project=orthogonal_sinkhorn)
            else:
                # For vectors, apply sign operation
                grad = grad.sign_()

        if state['factored']:
            d1, d2 = state['effective_shape']
            grad_reshaped = grad.view(d1, d2)

            if momentum != 0:
                buf = _reconstruct_state((state['mu_b_nmf'], state['mv_b_nmf'], state['sign'], d2), signed=True, shifter=state['shifter'])

                if snr_cond:
                    if not is_vector:
                        buf_2d_sq = buf.view(grad.shape[0], -1).square()
                        vt_row = (1 - buf_2d_sq.mean(dim=-1)).clamp_min_(1e-30)
                        vt_col = (1 - buf_2d_sq.mean(dim=-2)).clamp_min_(1e-30)
                        del buf_2d_sq
                    else:
                        denom = (1.0 - buf.square()).clamp_min_(1e-30).sqrt_().view_as(p)

                if nesterov and normed_mt:
                    # Scale the normalized gradient using empirical buffer magnitude (SNR recovery)
                    normed_grad = buf.abs().mul_(grad_reshaped)

                buf.lerp_(grad_reshaped, 1 - momentum)

                if nesterov:
                    nv_coef = momentum if nesterov_coef is None else nesterov_coef
                    if normed_mt:
                        update = normed_grad.lerp_(buf, nv_coef)
                    else:
                        update = grad_reshaped.lerp(buf, nv_coef)
                else:
                    update = buf.clone()

                # Factorize updated buffer
                for key, val in zip(('mu_b_nmf', 'mv_b_nmf', 'sign'), _factorize_state(buf, signed=True, shifter=state['shifter'])):
                    state[key].copy_(val)

            else:
                update = grad_reshaped.clone()

            update = update.view(p.shape)

        else:  # Standard logic for non-factored tensors
            actual_precision = group['actual_state_precision']

            if momentum != 0:
                buf = get_state(state, 'momentum_buffer', actual_precision)

                if snr_cond:
                    if not is_vector:
                        buf_2d_sq = buf.view(grad.shape[0], -1).square()
                        vt_row = (1 - buf_2d_sq.mean(dim=-1)).clamp_min_(1e-30)
                        vt_col = (1 - buf_2d_sq.mean(dim=-2)).clamp_min_(1e-30)
                        del buf_2d_sq
                    else:
                        denom = (1.0 - buf.square()).clamp_min_(1e-30).sqrt_()

                if nesterov and normed_mt:
                    # Scale the normalized gradient using empirical buffer magnitude (SNR recovery)
                    normed_grad = buf.abs().mul_(grad)

                buf.lerp_(grad, 1 - momentum)

                set_state(state, 'momentum_buffer', buf, actual_precision, random_int_state_tensor)

                if nesterov:
                    nv_coef = momentum if nesterov_coef is None else nesterov_coef
                    if normed_mt:
                        update = normed_grad.lerp_(buf, nv_coef)
                    else:
                        update = grad.lerp(buf, nv_coef)
                else:
                    update = buf.clone()
            else:
                update = grad.clone()

            del random_int_state_tensor

        if snr_cond:
            if not is_vector:
                # Align with Sinkhorn: Alternate row/col preconditioning
                update_2d = update.view(update.shape[0], -1)
                update_2d.mul_(vt_row.rsqrt().unsqueeze(1))
                update_2d.mul_(vt_col.rsqrt().unsqueeze(0))
                update = update_2d.atan_().view_as(p)
            else:
                update.atan2_(denom)

        if not group.get('normed_momentum', False):
            if not is_vector:
                # Sinkhorn iterative normalization
                update = apply_sr_sinkhorn(update, iters=sinkhorn_iterations, p=p, ortho_project=orthogonal_sinkhorn)
            else:
                # For vectors, apply sign operation
                update = update.sign_()

        if group.get('geometric_wd', False):
            if group["weight_decay"] > 0:
                if not is_vector:
                    wd_scaler = get_sinkhorn_wd_scaler(p, row_denom=vt_row, col_denom=vt_col)
                else:
                    wd_target = get_signsgd_wd_target(p, denom=denom)
            if is_vector and group.get('centered_wd', 0.0) > 0 and 'anchor_data' in state:
                anchor = dequantize_anchor(p, state, group, p.dtype)
                cwd_target = get_signsgd_wd_target(p.sub(anchor), denom=denom)
                del anchor

        update_scaling = step_size
        if snr_cond:
            # Compensate for the bounded atan/atan2 preconditioning in both paths.
            update_scaling = update_scaling * (4/math.pi)
        if group.get('spectral_normalization', False):
            update = scale_update(p, update, update_scaling, state=state)
        else:
            update.mul_(update_scaling)

        param_update.apply_parameter_update(self, p, group, update, step_size, random_int_tensor=random_int_tensor, wd_scaler=wd_scaler, wd_target=wd_target, cwd_target=cwd_target)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                self.step_parameter(p, group, i)

        return loss