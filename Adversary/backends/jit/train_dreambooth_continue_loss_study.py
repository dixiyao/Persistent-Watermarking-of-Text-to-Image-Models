#!/usr/bin/env python3
"""Continue-tune a PixelDiT copyright LoRA and log CP/ORG gradient geometry."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jit.train_core import add_common_args, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue PixelDiT copyright LoRA")
    add_common_args(parser, "continue")
    run(parser.parse_args(), "continue")
