#!/usr/bin/env python3
"""Train a LlamaGen copyright LoRA (robust study) with our own training loop."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parti.train_core import add_common_args, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LlamaGen copyright LoRA")
    add_common_args(parser, "robust")
    run(parser.parse_args(), "robust")
