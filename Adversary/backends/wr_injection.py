import torch

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


def _module_functional_call(module, params_override, *args, **kwargs):
    if not params_override:
        return module(*args, **kwargs)
    try:
        return functional_call(
            module,
            params_override,
            args=args,
            kwargs=kwargs,
            strict=False,
        )
    except TypeError:
        return functional_call(module, params_override, args, kwargs, strict=False)


def build_unet_mr_overrides(unet_override_specs, mr_weights):
    overrides = {}
    for full_name, local_name, base_param in unet_override_specs:
        target = base_param + mr_weights[full_name]
        overrides[local_name] = target.to(
            device=base_param.device,
            dtype=base_param.dtype,
        )
    return overrides


class MRInjectedUNet(torch.nn.Module):
    """UNet wrapper that performs virtual forward with W = W0 + M_R."""

    def __init__(self, base_unet, unet_override_specs, mr_weights):
        super().__init__()
        self.base_unet = base_unet
        self.unet_override_specs = unet_override_specs
        self.mr_weights = mr_weights
        self._cached_overrides = None

    def refresh_overrides(self):
        self._cached_overrides = build_unet_mr_overrides(
            unet_override_specs=self.unet_override_specs,
            mr_weights=self.mr_weights,
        )

    def clear_overrides_cache(self):
        self._cached_overrides = None

    def forward(self, *args, **kwargs):
        if self._cached_overrides is None:
            self.refresh_overrides()
        return _module_functional_call(self.base_unet, self._cached_overrides, *args, **kwargs)

    def _apply(self, fn):
        super()._apply(fn)
        self.mr_weights = {key: fn(value) for key, value in self.mr_weights.items()}
        self.clear_overrides_cache()
        return self

    @property
    def config(self):
        return self.base_unet.config

    @property
    def dtype(self):
        return next(self.base_unet.parameters()).dtype

    @property
    def device(self):
        return next(self.base_unet.parameters()).device

    def rebind_for_peft(self):
        """Call after get_peft_model wraps base_unet layers.

        PEFT replaces to_q/to_k/to_v/to_out.0 with LoraLinear whose .weight is a
        read-only property, which breaks functional_call. This method updates
        unet_override_specs to target base_layer.weight inside each LoRA-wrapped layer.
        """
        param_lookup = dict(self.base_unet.named_parameters())
        new_specs = []
        for full_name, local_name, base_param in self.unet_override_specs:
            if local_name in param_lookup:
                new_specs.append((full_name, local_name, param_lookup[local_name]))
            else:
                # e.g. to_q.weight -> to_q.base_layer.weight
                head, attr = local_name.rsplit(".", 1)
                rebound = f"{head}.base_layer.{attr}"
                if rebound in param_lookup:
                    new_specs.append((full_name, rebound, param_lookup[rebound]))
                else:
                    new_specs.append((full_name, local_name, base_param))
        self.unet_override_specs = new_specs
        self.clear_overrides_cache()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_unet, name)


# Backward-compatibility alias
WRInjectedUNet = MRInjectedUNet
