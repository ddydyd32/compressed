from omegaconf import OmegaConf
from argparse import ArgumentParser
import ast


def parse_args():
    parser = ArgumentParser()
    config = OmegaConf.load('baseline.yaml')
    for k, v in config.items():
        kwargs = {
            'default': v,
            'type': ast.literal_eval if type(v) is bool else type(v),
        }
        parser.add_argument(
            f"--{k}",
            **kwargs
        )
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(config, k, v)
    return config
    # parser.add_argument('--config', type=str, default='baseline.yaml')
    # args = parser.parse_args()
    # return args


# args = parse_args()
# config = OmegaConf.load(args.config)
# config.args.update(vars(args))

config = parse_args()
print(OmegaConf.to_yaml(config))


