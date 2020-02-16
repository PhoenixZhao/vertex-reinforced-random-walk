import copy
import itertools
import random
import time
from collections import defaultdict, namedtuple

import numpy as np
import torch
import torch.multiprocessing as mp
from tabulate import tabulate

from cogdl import options
from cogdl.tasks import build_task


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task = build_task(args)
    result = task.train()
    print("result:", result)
    return result


def gen_variants(**items):
    Variant = namedtuple("Variant", items.keys())
    return itertools.starmap(Variant, itertools.product(*items.values()))


def getpid(_):
    # HACK to get different pids
    time.sleep(1)
    return mp.current_process().pid


if __name__ == "__main__":
    # Magic for making multiprocessing work for PyTorch
    mp.set_start_method("spawn")

    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)
    variants = list(
        gen_variants(dataset=args.dataset, model=args.model, seed=args.seed)
    )
    results_dict = defaultdict(list)

    def variant_args_generator():
        """Form variants as group with size of num_workers"""
        for _item in variants:
            args.dataset, args.model, args.seed = _item
            yield copy.deepcopy(args)

    # Collect results
    results = []
    for item in variant_args_generator():
        results.append(main(item))
    for variant, result in zip(variants, results):
        results_dict[variant[:-1]].append(result)

    # Average for different seeds
    col_names = ["Variant"] + list(results_dict[variant[:-1]][-1].keys())

    tab_data = []
    for variant in results_dict:
        results = np.array([list(res.values()) for res in results_dict[variant]])
        tab_data.append(
            [variant]
            + list(
                itertools.starmap(
                    lambda x, y: f"{x:.4f}±{y:.4f}",
                    zip(
                        np.mean(results, axis=0).tolist(),
                        np.std(results, axis=0).tolist(),
                    ),
                )
            )
        )
    print(tabulate(tab_data, headers=col_names, tablefmt="github"))
