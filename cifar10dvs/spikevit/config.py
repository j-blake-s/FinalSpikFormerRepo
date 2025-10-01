import os
from yacs.config import CfgNode as CN
import yaml



# Base
_C = CN()
_C.BASE = ['']
_C.TOKENS = [6, 384]

# STEM
_C.STEM = CN()
_C.STEM.OUT_CHANNELS = 48
_C.STEM.HIDDEN_CHANNELS = 72
_C.STEM.STRIDE = 2
_C.STEM.KERNEL_SIZE = 3
_C.STEM.PADDING = 1


# BLOCK
_C.BLOCK = CN()
_C.BLOCK.IN_CHANNELS =      [48,  64,  96,  96, 128, 128,  256]
_C.BLOCK.HIDDEN_CHANNELS =  [96, 128, 196, 196, 256, 256, 1024]
_C.BLOCK.OUT_CHANNELS =     [64,  96,  96, 128, 128, 256,  512]
_C.BLOCK.MLP_RATIO =        [6,    6,   6,   6,   6,   6,    6]
_C.BLOCK.STRIDE =           [2,    2,   1,   2,   1,   2,    2]
_C.BLOCK.KERNEL_SIZE =      [3,    3,   3,   3,   3,   3,    3]
_C.BLOCK.PADDING =          [1,    1,   1,   1,   1,   1,    1]

# CHANNEL_CONV
_C.CHANNEL_CONV = CN()
_C.CHANNEL_CONV.OUT_CHANNELS = 1024

def _update_config_from_file(config, cfg_file):
  config.defrost()
  with open(cfg_file, 'r') as infile:
    yaml_cfg = yaml.load(infile, Loader=yaml.FullLoader)
  for cfg in yaml_cfg.setdefault('BASE', ['']):
    if cfg:
      _update_config_from_file(
        config, os.path.join(os.path.dirname(cfg_file), cfg)
      )
  config.merge_from_file(cfg_file)
  config.freeze()

def get_config(cfg_file=None):
  config = _C.clone()
  if cfg_file:
      _update_config_from_file(config, cfg_file)
  return config