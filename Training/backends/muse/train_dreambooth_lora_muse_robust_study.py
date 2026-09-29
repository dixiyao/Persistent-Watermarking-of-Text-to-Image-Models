#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from muse.personalization import resolve_training_mode, run_personalization

    mode, argv = resolve_training_mode(sys.argv[1:])
    if mode == "personalization":
        run_personalization(argv)
        raise SystemExit(0)
    sys.argv[1:] = argv
    from muse.train_core import add_common_args, run

    parser = argparse.ArgumentParser(description="Train aMUSEd copyright LoRA")
    add_common_args(parser, "robust")
    run(parser.parse_args(), "robust")
