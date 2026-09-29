#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from muse.train_core import add_common_args, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue aMUSEd training with gradient geometry")
    add_common_args(parser, "continue")
    run(parser.parse_args(), "continue")
