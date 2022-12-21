import os
import argparse
from datasets import SUPPORTED_DATASETS

def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_location", type=str, default="/datadrive/dump")
    parser.add_argument(
        "--source_dataset", type=str, default="mnist", choices=SUPPORTED_DATASETS
    )
    parser.add_argument(
        "--target_dataset", type=str, default="usps", choices=SUPPORTED_DATASETS
    )

    # experiment
    parser.add_argument(
        "--method",
        type=str,
        default="drcn",
        choices=["drcn", "drcn-st", "drcn-s", "base"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="/datadrive/dump/")

    # model architecture
    parser.add_argument("--fc4", type=int, default=500)
    parser.add_argument("--fc5", type=int, default=500)

    # optimization
    parser.add_argument("--lamb", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--no_mix_src_and_tgt",
        action="store_true",
        default=False,
        help="if used, we follow Alg. 1 and go through src dataset, then tgt dataset",
    )

    args = parser.parse_args()

    # Make sure we are attempting a valid transfer experiment
    ds1, ds2 = sorted([args.source_dataset, args.target_dataset])
    if ds1 == "mnist" and ds2 == "usps":
        args.img_size = 28
        args.in_channels = 1
        args.n_classes = 10
    elif ds1 == "mnist" and ds2 == "svhn":
        args.img_size = 32
        args.in_channels = 1
        args.n_classes = 10
    elif ds1 == "cifar10" and ds2 == "stl":
        args.img_size = 32
        args.in_channels = 1
        args.n_classes = 8
    else:
        raise ValueError("invalid combination of datasets")

    return args