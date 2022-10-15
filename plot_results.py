import os
import glob
import argparse
import json

import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import OffsetImage
from matplotlib.offsetbox import AnnotationBbox


sns.set(style="whitegrid", palette="Paired")
params = {
    "figure.constrained_layout.use": True,
    "axes.labelsize": 14,
    # "xtick.labelsize": 18,
    # "ytick.labelsize": 18,
    "legend.fontsize": 8,
    "legend.title_fontsize": 10,
    # "font.size": 24,
}
plt.rcParams.update(params)


def parse_args():
    parser = argparse.ArgumentParser(description="Causal Experimental Design plot")
    parser.add_argument(
        "--num_nodes", type=int, default=20, help="Number of nodes of the graph",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="results/",
        help="Path to load results and store plots.",
    )
    parser.add_argument(
        "--plots_path",
        type=str,
        default="plots/",
        help="Path to load results and store plots.",
    )
    parser.add_argument(
        "--exp_edges",
        type=int,
        default=1,
        help="Expected Number of edges in random graph",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="isotropic-gaussian",
        help="Type of noise variables in SCM",
    )
    parser.add_argument(
        "--nonlinear",
        action="store_true",
        help="NonLinear model or linear model",
    )
    parser.add_argument("--env", type=str, default="erdos", help="SCM to use")
    args = parser.parse_args()
    args.model_type = "nonlinear" if args.nonlinear else "linear"

    return args


if __name__ == "__main__":
    args = parse_args()

    plt.rcParams.update(
        {
            "legend.fontsize": 8,
            "legend.title_fontsize": 9,
            "axes.labelsize": 15,
            "xtick.labelsize": 15,
        }
    )

    # these keys are used to create different plots
    keys = ["strategy"]  # , "model", "env", "id"]
    keys_value = ["value_strategy"]
    plot_metrics = ["eshd", "auroc", "eshd_double", "auprc", "sid", "nll_held_out"]
    data = []

    for f in glob.glob(
        os.path.join(
            args.results_path,
            f"{args.env}_*_{args.num_nodes}_*_{args.model_type}_*"
        )
    ):
        if not os.path.exists(os.path.join(f, "config.json")):
            continue
        if not os.path.exists(os.path.join(f, "metrics.jsonl")):
            continue
        config = json.load(open(os.path.join(f, "config.json")))
        if config["strategy"] == "softbald":
            if not config["bald_temperature"] == 2.0:
                continue 
        metrics = pd.read_json(open(os.path.join(f, "metrics.jsonl")), lines=True)
        if not metrics["interventional_samples"].max() == 100:
            continue
        metrics["model"] = ""
        metrics["env"] = ""
        metrics["strategy"] = ""
        metrics["value_strategy"] = ""
        metrics["id"] = ""
        for (k, v) in config.items():
            if k == "node_range":
                continue
            metrics[k] = v
        data.append(metrics)
    data = pd.concat(data)
    # keep only the seeds that are common to all runs
    seeds_all = set(data.data_seed.unique())

    #for name, group in data.groupby(keys):
        #for name1, group1 in group.groupby(keys_value):
            #print(name, name1, len(set(group1.data_seed.unique())))
            #seeds = seeds_all.intersection(set(group1.data_seed.unique()))
            #data = group[group["data_seed"].isin(seeds)]

    plots = [plt.subplots(dpi=150) for _ in plot_metrics]

    colors = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13"]
    i = 0

    for name1, group1 in data.groupby(keys):
        if name1 == "f-score":
            name1 = "ait"
        for name, group in group1.groupby(keys_value):
            if name == "sample-dist":
                continue
            if name == "gp-ucb":
                if name1 == "random" or name1 == "randombatch":
                    continue
            print("plotting", name, name1)
            stats = group.groupby(["interventional_samples"]).agg(["mean", "sem", "count"])
            for m, gm in enumerate(plots):
                interventions, means, sems, n = (
                    stats[plot_metrics[m]].reset_index().to_numpy().T
                )
                print(n.min())
                gm[1].plot(
                    interventions,
                    means,
                    color=colors[i],
                    lw=6,
                    ls=":",
                    label="-".join(map(str, [name1.upper(), name.upper()])) + f" ({n.min()})",
                )

                gm[1].fill_between(
                    x=interventions,
                    y1=means - 1.96 * sems,
                    y2=means + 1.96 * sems,
                    color=colors[i],
                    alpha=0.3,
                )

            i += 1

    for m, gm in enumerate(plots):
        gm[1].set_ylabel(plot_metrics[m].upper())
        gm[1].set_xlabel("Interventional Samples")
        gm[1].set_title(
            "Env: {}-{} Seeds: {}".format(
                metrics["env"][0], args.exp_edges, int(n.min())
            )
        )
        gm[1].legend(loc="best")

        gm[0].savefig(
            os.path.join(
                args.plots_path,
                f"results_{args.num_nodes}_{args.exp_edges}_{args.env}_{args.model_type}_{plot_metrics[m]}.pdf"
            ),
            dpi=300,
        )
        plt.close()
