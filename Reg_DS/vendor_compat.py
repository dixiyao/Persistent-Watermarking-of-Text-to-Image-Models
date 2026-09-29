"""Compatibility shims for the vendored upstream trainers.

``flux/vendor/train_dreambooth_lora_flux2_klein.py`` and
``muse/vendor/train_amused.py`` are kept byte-identical to the diffusers
examples they came from, so anything that has to bend to the installed
library versions is patched here instead of in those files.

Two things need bending:

* the FLUX trainer imports twelve names from ``diffusers.training_utils``,
  several of which only exist in recent releases;
* both trainers hand ``accelerator.init_trackers`` a raw ``vars(args)``
  dict, and the TensorBoard tracker rejects any value that is not a scalar
  or a string.

NOTE: this module was reconstructed for the public release - the private
repository imports it but does not contain it.  The fallbacks below are
faithful for the helpers whose behaviour is unambiguous; the rest raise
rather than guess, so an unsupported path fails loudly instead of training
something subtly wrong.
"""

import gc


def _free_memory():
    gc.collect()
    try:
        import torch
    except ImportError:                                  # pragma: no cover
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        torch.mps.empty_cache()


def _to_cpu_contiguous(state_dict):
    return {k: v.detach().to("cpu").contiguous() if hasattr(v, "detach") else v
            for k, v in state_dict.items()}


def _collate_lora_metadata(modules_to_save):
    """Upstream returns {f"{name}_lora_adapter_metadata": peft_config_dict}."""
    metadata = {}
    for name, module in (modules_to_save or {}).items():
        if module is None:
            continue
        cfg = getattr(module, "peft_config", None)
        if isinstance(cfg, dict):
            cfg = cfg.get("default")
        if cfg is not None:
            metadata[f"{name}_lora_adapter_metadata"] = (
                cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg))
    return metadata


def _unsupported(name, since):
    def _raise(*_args, **_kwargs):
        raise NotImplementedError(
            "%s() is used by the vendored FLUX trainer but is missing from the "
            "installed diffusers. It has no safe fallback, so install "
            "diffusers >= %s, or avoid the option that reaches this code path "
            "(aspect-ratio bucketing, FSDP, or CPU offload)." % (name, since))
    return _raise


def patch_diffusers_training_utils():
    """Fill in whatever the installed diffusers.training_utils is missing."""
    import diffusers.training_utils as tu

    faithful = {
        "free_memory": _free_memory,
        "_to_cpu_contiguous": _to_cpu_contiguous,
        "_collate_lora_metadata": _collate_lora_metadata,
    }
    # No sane fallback: these change what is trained or how it is sharded.
    guarded = {
        "generate_aspect_ratio_buckets": "0.33",
        "parse_buckets_string": "0.33",
        "find_nearest_bucket": "0.33",
        "get_fsdp_kwargs_from_accelerator": "0.35",
        "wrap_with_fsdp": "0.35",
        "offload_models": "0.33",
    }
    added = []
    for name, impl in faithful.items():
        if not hasattr(tu, name):
            setattr(tu, name, impl); added.append(name)
    for name, since in guarded.items():
        if not hasattr(tu, name):
            setattr(tu, name, _unsupported(name, since)); added.append(name)
    if added:
        print("vendor_compat: patched diffusers.training_utils -> %s"
              % ", ".join(sorted(added)), flush=True)
    return added


def patch_tensorboard_hparams():
    """Let init_trackers(config=vars(args)) survive non-scalar values.

    TensorBoard's hparams plugin accepts only bool / int / float / str.  The
    vendored trainers pass argparse namespaces straight through, which carry
    lists and None.  Coerce instead of crashing.
    """
    try:
        from accelerate.tracking import TensorBoardTracker
    except ImportError:                                  # pragma: no cover
        return False
    if getattr(TensorBoardTracker, "_vendor_compat_patched", False):
        return True

    original = TensorBoardTracker.store_init_configuration

    def store_init_configuration(self, values):
        clean = {}
        for key, value in (values or {}).items():
            if isinstance(value, bool) or isinstance(value, (int, float, str)):
                clean[key] = value
            elif value is None:
                clean[key] = "None"
            else:
                clean[key] = str(value)
        return original(self, clean)

    TensorBoardTracker.store_init_configuration = store_init_configuration
    TensorBoardTracker._vendor_compat_patched = True
    print("vendor_compat: TensorBoard hparams coerced to scalars", flush=True)
    return True
