import argparse
import os
import numpy as np
import torch
from typing import Dict, Sequence


class DPSolver:

    def __init__(self, costs: Dict[str, Sequence[float]]):
        """
        Args:
            costs: dict of cost options for each layer
        """
        self.costs = costs

    def solve(self, errors: Dict[str, Sequence[float]], max_cost: int) -> np.ndarray:
        """
        Args:
            errors: dict of errors for each layer and option
            max_cost: target cost

        Note:
            Notation follows Algorithm 1 from SPDY
        """
        costs = self.costs
        num_layers = len(costs)
        layer_names = list(costs.keys())

        D = np.full((num_layers, max_cost + 1), float("inf"))
        P = np.full((num_layers, max_cost + 1), -1)

        # Fill values for 0-th layer
        layer_name = layer_names[0]
        num_options = len(costs[layer_name])
        for i in range(num_options):
            if errors[layer_name][i] < D[0][costs[layer_name][i]]:
                D[0][costs[layer_name][i]] = errors[layer_name][i]
                P[0][costs[layer_name][i]] = i
        # Process layers
        for l in range(1, num_layers):
            layer_name = layer_names[l]
            num_options = len(self.costs[layer_name])
            for i in range(num_options):
                cost = costs[layer_name][i]
                score = errors[layer_name][i]
                if cost == 0:
                    tmp = D[l - 1] + score
                    better = tmp < D[l]
                    if np.sum(better):
                        D[l][better] = tmp[better]
                        P[l][better] = i
                    continue
                if cost > max_cost:
                    continue
                try:
                    tmp = D[l - 1][:-cost] + score
                except:
                    print("tmp", tmp)
                    print("score", score)
                better = tmp < D[l][cost:]
                if np.sum(better):
                    D[l][cost:][better] = tmp[better]
                    P[l][cost:][better] = i
        # Determine best configuration
        score = np.min(D[-1, :])
        cost = np.argmin(D[-1, :])
        solution = {}
        for l in range(len(D) - 1, -1, -1):
            layer_name = layer_names[l]
            chosen_option = P[l][cost]
            solution[layer_name] = chosen_option
            cost -= costs[layer_name][chosen_option]
        # Reverse dict and return
        return dict(reversed(solution.items()))


def parse_args():
    parser = argparse.ArgumentParser(description="DP Search.")
    # Compression params
    parser.add_argument(
        "--compressed_weights_path",
        type=str,
        required=True,
        help="The name or path to the model being pruned",
    )
    parser.add_argument("--target_cost", type=int, required=True, help="Target average cost per layer.")
    parser.add_argument("--is_sparsity", action="store_true", help="Whether the compression is sparsity.")
    parser.add_argument(
        "--configuration_name",
        type=str,
        default="dp_configuration.txt",
        help="Name of final configuration",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    errors = torch.load(f"{args.compressed_weights_path}/errors.pth")

    sign = -1 if args.is_sparsity else 1

    solution = {}
    for _, group_errors in errors.items():
        # Get costs
        costs = {}
        # Group cost offset
        min_cost = 0
        for layer_name in group_errors:
            costs[layer_name] = sorted(
                [int(x.split(".")[0]) for x in os.listdir(os.path.join(args.compressed_weights_path, layer_name))]
            )
            costs[layer_name] = [sign * x for x in costs[layer_name]]
            min_cost = min(min_cost, min(costs[layer_name]))
        if min_cost < 0:
            for layer_name, layer_costs in costs.items():
                for i in range(len(layer_costs)):
                    layer_costs[i] -= min_cost
        solver = DPSolver(costs)
        group_solution = solver.solve(group_errors, (args.target_cost - min_cost) * len(costs))
        # Process solution
        group_solution = {
            layer_name: sign * (costs[layer_name][v] + min_cost) for layer_name, v in group_solution.items()
        }
        solution.update(group_solution)

    # Sort solution
    def key_sort_fn(key):
        split_key = key.split(".")
        block_id = int(split_key[2])
        misc = split_key[3:]
        return (block_id, *misc)

    solution = dict(sorted(solution.items(), key=lambda item: key_sort_fn(item[0])))

    print(os.path.join(args.compressed_weights_path, args.configuration_name))
    # Save final configuration
    with open(os.path.join(args.compressed_weights_path, args.configuration_name), "w") as f:
        f.write("\n".join([f"{layer_name}: {level}" for layer_name, level in solution.items()]))


if __name__ == "__main__":
    main()
