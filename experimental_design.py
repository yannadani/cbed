import os
import argparse
import random
import json
import numpy as np

from utils.logger import Logger

from envs import ErdosRenyi, ScaleFree, BifEnvironment, Dream4Environment
from strategies import (
    RandomBatchAcquisitionStrategy,
    RandomAcquisitionStrategy,
    ABCDStrategy,
    CBEDStrategy,
    GreedyCBEDStrategy,
    SoftCBEDStrategy,
    ReplayStrategy,
    FScoreBatchStrategy
)
from models import DagBootstrap, DiBS_BGe, DiBS_Linear, DiBS_NonLinear
from replay_buffer import ReplayBuffer

wandb = None
if 'WANDB_API_KEY' in os.environ:
    import wandb

import torch
import warnings

def parse_args():
    parser = argparse.ArgumentParser(description="Causal Experimental Design")
    parser.add_argument(
        "--save_path", type=str, default="results/", help="Path to save result files"
    )
    parser.add_argument(
        "--id", type=str, default=None, help="ID for the run"
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=20,
        help="random seed for generating data (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--num_nodes", type=int, default=20, help="Number of nodes in the causal model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dag_bootstrap",
        help="Posterior model to use {vcn, dibs, dag_bootstrap}",
    )
    parser.add_argument("--env", type=str, default="erdos", help="SCM to use")
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        help="Acqusition strategy to use {abcd, random}",
    )
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument(
        "--sparsity_factor",
        type=float,
        default=0.0,
        help="Hyperparameter for sparsity regulariser",
    )
    parser.add_argument(
        "--exp_edges",
        type=int,
        default=1,
        help="Number of expected edges in random graphs",
    )
    parser.add_argument(
        "--alpha_lambd", type=int, default=10.0, help="Hyperparameter for the bge score"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Total number of samples in the synthetic data",
    )
    parser.add_argument(
        "--num_starting_samples",
        type=int,
        default=100,
        help="Total number of samples in the synthetic data to start with",
    )
    parser.add_argument(
        "--dibs_steps",
        type=int,
        default=20000,
        help="Total number of steps DiBs to run per training iteration.",
    )
    parser.add_argument(
        "--dibs_graph_prior",
        type=str,
        default='er',
        help="DiBs graph prior (only applicable for bif env).",
    )

    parser.add_argument(
        "--exploration_steps",
        type=int,
        default=3,
        help="Total number of exploration steps in gp-ucb",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="isotropic-gaussian",
        help="Type of noise of causal model",
    )
    parser.add_argument(
        "--bald_temperature", type=float, default=2.0, help="Temperature of soft bald"
    )
    parser.add_argument(
        "--noise_sigma", type=float, default=0.1, help="Std of Noise Variables"
    )
    parser.add_argument(
        "--theta_mu", type=float, default=2.0, help="Mean of Parameter Variables"
    )
    parser.add_argument(
        "--theta_sigma", type=float, default=1.0, help="Std of Parameter Variables"
    )
    parser.add_argument(
        "--gibbs_temp", type=float, default=1000.0, help="Temperature of Gibbs factor"
    )

    # TODO: improve names
    parser.add_argument('--num_intervention_values', type=int, default=5, help="Number of interventional values to consider.")
    parser.add_argument('--intervention_values', type=float, nargs="+", help='Interventioanl values to set in `grid` value_strategy, else ignored.')
    parser.add_argument('--intervention_value', type=float, default=0.0, help="Interventional value to set in `fixed` value_strategy, else ingored.")

    parser.add_argument(
        "--group_interventions", action='store_true'
    )
    parser.add_argument(
        "--plot_graphs", action='store_true'
    )
    parser.add_argument('--no_sid', action='store_true', default=False)
    parser.set_defaults(group_interventions=False)
    parser.add_argument(
        "--nonlinear", action='store_true'
    )
    parser.add_argument(
        "--value_strategy",
        type=str,
        default="fixed",
        help="Possible strategies: gp-ucb, grid, fixed, sample-dist",
    )
    parser.add_argument(
        "--bif_file", type=str, default="bif/sachs.bif", help="Path of BIF file to load"
    )
    parser.add_argument(
        "--dream4_path", type=str, default="envs/dream4/configurations/", help="Path of DREAM4 files."
    )
    parser.add_argument(
        "--dream4_name", type=str, default="insilico_size10_1", help="Name of DREAM4 experiment to load."
    )
    parser.add_argument(
        "--bif_mapping", type=str, default="{\"LOW\": 0, \"AVG\": 1, \"HIGH\": 2}", help="BIF states mapping"
    )

    parser.set_defaults(nonlinear=False)

    args = parser.parse_args()
    args.node_range = (-10, 10)

    if args.env== "sf":
        args.dibs_graph_prior = "sf"

    return args


STRATEGIES = {
    "abcd": ABCDStrategy,
    "softcbed": SoftCBEDStrategy,
    "greedycbed": GreedyCBEDStrategy,
    "cbed": CBEDStrategy,
    "random": RandomAcquisitionStrategy,
    "randombatch": RandomBatchAcquisitionStrategy,
    "replay": ReplayStrategy,
    "ait": FScoreBatchStrategy
}
MODELS = {"dag_bootstrap": DagBootstrap, "dibs": DiBS_BGe, "dibs_linear": DiBS_Linear, "dibs_nonlinear": DiBS_NonLinear}
ENVS = {"erdos": ErdosRenyi, "sf": ScaleFree, "bif": BifEnvironment, "dream4": Dream4Environment}


def causal_experimental_design_loop(args):

    # prepare save path
    args.save_path = os.path.join(
        args.save_path,
        "_".join(
            map(
                str,
                [
                    args.env,
                    args.data_seed,
                    args.seed,
                    args.num_nodes,
                    args.num_starting_samples,
                    args.model,
                    args.strategy,
                    args.value_strategy,
                    args.exp_edges,
                    args.noise_type,
                    args.noise_sigma,
                    args.bald_temperature,
                    args.intervention_value,
                    'nonlinear' if args.nonlinear else 'linear',
                    args.dream4_name if args.env == 'dream4' else '',
                    args.id
                ],
            )
        ),
    )

    if wandb is not None:
        wandb.init(project="CED", name=args.id)
        wandb.config.update(args, allow_val_change=True)

    os.makedirs(args.save_path, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.save_path, "config.json"), "w"))
    logger = Logger(args.save_path, resume=False, wandb=wandb)

    # set the seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.env == 'bif':
        env = ENVS[args.env](args.bif_file, args.bif_mapping, logger=logger)
        args.num_nodes = env.num_nodes
        args.noise_sigma = [args.noise_sigma] * args.num_nodes
    if args.env == 'dream4':
        env = ENVS[args.env](args.data_seed, args.dream4_path, args.dream4_name, logger=logger)
        args.num_nodes = env.num_nodes
        args.noise_sigma = [args.noise_sigma] * args.num_nodes
    else:
        env = ENVS[args.env](
            num_nodes=args.num_nodes,
            exp_edges=args.exp_edges,
            noise_type=args.noise_type,
            noise_sigma=args.noise_sigma,
            num_samples=args.num_samples,
            mu_prior=args.theta_mu,
            sigma_prior=args.theta_sigma,
            seed=args.data_seed,
            nonlinear = args.nonlinear,
            logger=logger
        )
        if args.env == "erdos":
            args.dibs_graph_prior = "er"
        else:
            args.dibs_graph_prior = args.env
        args.noise_sigma = env._noise_std

    model = MODELS[args.model](args)

    env.plot_graph(os.path.join(args.save_path, "graph.png"))

    buffer = ReplayBuffer()
    # sample num_starting_samples initially - not num_samples
    buffer.update(env.sample(args.num_starting_samples))

    # if DAG_BOOTSTRAP:
    samples = buffer.data().samples
    args.sample_mean = samples.mean(0)
    args.sample_std = samples.std(0, ddof=1)

    precision_matrix = np.linalg.inv(samples.T @ samples / len(samples))
    model.precision_matrix = precision_matrix
    model.update(buffer.data())

    strategy = STRATEGIES[args.strategy](model, env, args)

    # evaluate
    logger.log_metrics(
        {
            "eshd": env.eshd(model, 1000, double_for_anticausal=False),
            "sid": -1 if args.no_sid else env.sid(model, 1000, force_ensemble=True),
            "auroc":  env.auroc(model, 1000),
            "auprc":  env.auprc(model, 1000),
            "observational_samples": buffer.n_obs,
            "interventional_samples": buffer.n_int,
            "ensemble_size": len(model.dags)
        }
    )

    warnings.warn(
            "Assuming value sampler corresponds to `Constant` distribution. Code can break/could lead to wrong results if using any other sampler"
        )
    for i in range(args.num_batches):
        print(f"====== Experiment {i+1} =======")

        # example of information based strategy
        valid_interventions = list(range(args.num_nodes))
        interventions = strategy.acquire(valid_interventions, i)

        for node, samplers in interventions.items():
            for sampler in samplers:
                buffer.update(env.intervene(i, 1, node, sampler))

        model.update(buffer.data())
        logger.log_metrics(
            {
                "eshd": env.eshd(model, 1000, double_for_anticausal=False),
                "sid": -1 if args.no_sid else env.sid(model, 1000, force_ensemble=True),
                "auroc":  env.auroc(model, 1000),
                "auprc":  env.auprc(model, 1000),
                "observational_samples": buffer.n_obs,
                "interventional_samples": buffer.n_int,
                "ensemble_size": len(model.dags)
            }
        )


if __name__ == "__main__":
    args = parse_args()
    causal_experimental_design_loop(args)
