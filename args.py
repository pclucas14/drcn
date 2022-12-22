import os
import argparse
from datasets import SUPPORTED_DATASETS


def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        "--data_location",
        type=str,
        default="/datadrive/dump",
        help="path to the directory of datasets",
    )
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/datadrive/dump/",
        help="directory where to store model checkpoints and results",
    )

    # model architecture
    parser.add_argument("--fc4", type=int, default=1000)
    parser.add_argument("--fc5", type=int, default=1000)

    # optimization
    parser.add_argument(
        "--lamb", type=float, default=0.7, help="lambda parameter in the paper"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-6)
    parser.add_argument("--learned_tgt_norm", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--noise_std",
        type=float,
        default=1e-2,
        help="std of gaussian noised in denoising objective",
    )
    parser.add_argument(
        "--noise_p_drop",
        type=float,
        default=0.2,
        help="prob. of masking to 0 in denoising objective",
    )
    parser.add_argument(
        "--no_mix_src_and_tgt",
        action="store_true",
        default=False,
        help="if used, we follow Alg. 1 and go through src dataset, then tgt dataset",
    )
    parser.add_argument(
        "--optim", type=str, default="adam", choices=["adam", "rmsprop"]
    )

    args = parser.parse_args()

    # Make sure we are attempting a valid transfer experiment
    ds1, ds2 = sorted([args.source_dataset, args.target_dataset])
    if ds1 == "mnist" and ds2 == "usps":
        args.img_size = 32  # 28
        args.in_channels = 1
        args.n_classes = 10
    elif ds1 == "mnist" and ds2 == "svhn":
        args.img_size = 32
        args.in_channels = 1
        args.n_classes = 10
    elif ds1 == "cifar10" and ds2 == "stl10":
        args.img_size = 32
        args.in_channels = 3
        args.n_classes = 8
    else:
        raise ValueError("invalid combination of datasets")

    return args
