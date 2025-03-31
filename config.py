from omegaconf import OmegaConf
from argparse import ArgumentParser
import ast


def parse_args():
    parser = ArgumentParser()
    # config = OmegaConf.load('baseline.yaml')
    # classes = {}
    # for major, _item in config.items():
    #     for k, v in _item.items():
    #         classes[k] = major
    #         kwargs = {
    #             'default': v,
    #             'type': ast.literal_eval if type(v) is bool else type(v),
    #         }
    #         parser.add_argument(
    #             f"--{k}",
    #             **kwargs
    #         )
    # args = parser.parse_args()
    # for k, v in vars(args).items():
    #     major = classes[k]
    #     setattr(getattr(config, major), k, v)
    # return config
    parser.add_argument('--config', type=str, default='baseline.yaml')
    args = parser.parse_args()
    return args


args = parse_args()
config = OmegaConf.load(args.config)
# config.args.update(vars(args))

print(OmegaConf.to_yaml(config))


# config = parse_args()
