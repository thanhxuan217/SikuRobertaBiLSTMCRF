import pickle
from argparse import Namespace

import torch

from parsering.config import Config


def load_checkpoint(path, map_location=None):
    """Load checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def _is_plain_data(value):
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return True

    if hasattr(value, "__fspath__"):
        return True

    if isinstance(value, list):
        return all(_is_plain_data(item) for item in value)

    if isinstance(value, tuple):
        return all(_is_plain_data(item) for item in value)

    if isinstance(value, dict):
        return all(_is_plain_data(key) and _is_plain_data(item)
                   for key, item in value.items())

    return False


def serialize_args(args):
    """Save args as plain data for better checkpoint portability."""
    if isinstance(args, dict):
        raw_args = dict(args)
    else:
        namespace = getattr(args, "namespace", None)
        if namespace is not None:
            raw_args = vars(namespace).copy()
        elif isinstance(args, Namespace):
            raw_args = vars(args).copy()
        elif hasattr(args, "__dict__"):
            raw_args = vars(args).copy()
        else:
            return args

    serialized_args = {}
    for key, value in raw_args.items():
        if hasattr(value, "__fspath__"):
            serialized_args[key] = str(value)
        elif _is_plain_data(value):
            serialized_args[key] = value

    return serialized_args


def restore_args(saved_args):
    """Normalize checkpoint args into attribute-style access."""
    if isinstance(saved_args, Namespace):
        return saved_args

    if isinstance(saved_args, dict):
        return Namespace(**saved_args)

    if isinstance(saved_args, Config):
        namespace = getattr(saved_args, "namespace", None)
        return namespace if namespace is not None else saved_args

    return saved_args
