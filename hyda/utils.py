import torch
import argparse
from importlib import import_module

def load_submodule_from_checkpoint(ckpt, submodule_name, map_location='cuda:0'):
    if isinstance(ckpt, str):
        ckpt = torch.load(ckpt, map_location=map_location)
    state_dict = {k.replace(f"{submodule_name}.", ""): v for k, v in ckpt['state_dict'].items() if k.startswith(f"{submodule_name}.")}
    module_name, class_name = ckpt['hyper_parameters']['init_args'][submodule_name]['class_path'].rsplit('.', 1)
    module_cls = getattr(import_module(module_name), class_name)
    module_kwargs = ckpt['hyper_parameters']['init_args'][submodule_name].get('init_args', {})
    module_instance = module_cls(**module_kwargs)
    module_instance.load_state_dict(state_dict, strict=True)
    return module_instance

def namespace_to_dict(namespace):
    """Recursively converts an argparse.Namespace to a dictionary."""
    return {key: namespace_to_dict(value) if isinstance(value, argparse.Namespace) else value
            for key, value in vars(namespace).items()}