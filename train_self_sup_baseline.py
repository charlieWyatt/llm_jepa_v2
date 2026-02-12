"""
Self-supervised I-JEPA baseline on NL-RX-Synth.

Swaps in the flat-dataloader config, then delegates to train_accelerate.
"""

import train_accelerate
from config_self_sup_baseline import STRATEGY_CONSTS

train_accelerate.STRATEGY_CONSTS = STRATEGY_CONSTS

if __name__ == "__main__":
    train_accelerate.run_train_jepa()
