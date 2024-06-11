from omegaconf import OmegaConf
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    k, v = 'config_path', 'baseline.yaml'
    kwargs = {
        'default': v,
        'type': type(v),
        'action': 'store_true' if type(v) is bool else None
    }
    if type(v) is bool:
        kwargs.pop('type')
    parser.add_argument(
        f"--{k}",
        **kwargs
    )
    args = parser.parse_args()
    return args

args = parse_args()
config = OmegaConf.load(args.config_path)
