import os
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # keep track of amulet stuff
    parser.add_argument(
        "--data_location", type=str, default='/datadrive/dump'
    )
    parser.add_argument(
        "--method", type=str, default='drcn', choices=['drcn', 'drcn-st', 'drcn-s', 'base']
    )
    parser.add_argument("--source_dataset", type=str, default='mnist')
    parser.add_argument("--target_dataset", type=str, default='usps')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=28)

    return parser