import numpy as np
from envs.samplers import Constant
from scipy.special import logsumexp

from .acquisition_strategy import AcquisitionStrategy
from collections import defaultdict


def entropy(p):
    return -sum(p * np.log(p))


def logmeanexp(A, axis):
    return logsumexp(A, axis=axis)-np.log(A.shape[axis])

class CBEDStrategy(AcquisitionStrategy):
    def _score_for_value(self, nodes, value_samplers):

        # DAGs x Interventions x Samples x Nodes - y[t][m]
        n_samples = self.num_samples
        datapoints = self.model.sample_interventions(
            nodes, value_samplers, n_samples
        )
        logpdfs = self.model._update_likelihood(
            nodes, n_samples, value_samplers, datapoints
        )
        logpdfs_val = logpdfs.values

        total_uncertainty = -logmeanexp(logpdfs_val, axis = 2).mean((1,2))
        aleatoric_uncertainty = -np.diagonal(logpdfs_val, axis1=1, axis2=2).mean((1, 2))

        MI = total_uncertainty - aleatoric_uncertainty

        return MI, {}


class GreedyCBEDStrategy(CBEDStrategy):
    def _score_for_value(self, nodes, value_samplers, pnm1):

        # DAGs x Interventions x Samples x Nodes - y[t][m]
        n_samples = self.num_samples
        datapoints = self.model.sample_interventions(
            nodes, value_samplers, n_samples
        )

        logpdfs = self.model._update_likelihood(
            nodes, n_samples, value_samplers, datapoints
        )

        # intervention x inner dag x outer dag x samples
        logpdfs_val = logpdfs.values

        if pnm1 is not None:
            # the recursive equation (13) in batchbald paper
            logpdfs_val += pnm1[None, ...]

        total_uncertainty = -logmeanexp(logpdfs_val, axis = 2).mean((1,2))

        aleatoric_uncertainty = -np.diagonal(logpdfs_val, axis1=1, axis2=2).mean((1, 2))

        MI = total_uncertainty - aleatoric_uncertainty

        return MI, {'logpdfs': logpdfs_val}

    def acquire(self, nodes, iteration):
        n_boot = len(self.model.dags)
        current_logpdfs = np.zeros([n_boot, n_boot])

        # DAGs x Interventions x Samples x Nodes - y[t][m]
        strategy = self.get_value_strategy(nodes)

        targets = []
        values = []
        pnm1 = None
        for _ in range(self.args.batch_size):
            strategy(
                self._score_for_value,
                n_iters =self.args.num_intervention_values,
                pnm1=pnm1)


            pnm1 = strategy.extra[strategy.max_iter]['logpdfs'][strategy.max_j]

            targets.append(strategy.max_j)
            values.append(strategy.max_x)

        selected_interventions = defaultdict(list)
        for value, target in zip(values, targets):
            selected_interventions[target].append(Constant(value))

        return selected_interventions


class SoftCBEDStrategy(CBEDStrategy):
    def __init__(self, model, env, args):
        super().__init__(model, env, args)
        self.temperature = args.bald_temperature

    def acquire(self, nodes, iteration):

        # DAGs x Interventions x Samples x Nodes - y[t][m]
        strategy = self.get_value_strategy(nodes)
        strategy(self._score_for_value, n_iters = self.args.num_intervention_values)

        probs = (np.exp(strategy.target / self.temperature) / np.exp(strategy.target / self.temperature).sum())

        assert self.batch_size < self.args.num_nodes, 'Batch size need to be smaller than the number of nodes'
        try:
            interventions = np.random.choice(range(len(probs.flatten())), p=probs.flatten(), replace=False, size=self.batch_size)
            value_ids, node_ids = np.unravel_index(interventions, shape=probs.shape)
        except ValueError:
            value_ids, node_ids = np.unravel_index(np.nonzero(probs.flatten())[0], shape = probs.shape)
            interventions = np.random.choice(range(len(probs.flatten())), p=probs.flatten(), replace=True, size=self.batch_size - len(value_ids))
            value_ids_, node_ids_ = np.unravel_index(interventions, shape=probs.shape)

            value_ids = np.concatenate([value_ids, value_ids_])
            node_ids = np.concatenate([node_ids, node_ids_])

        selected_interventions = defaultdict(list)
        for value_id, node in zip(value_ids, node_ids):
            selected_interventions[nodes[node]].append(Constant(strategy.values[value_id][nodes[node]]))

        return selected_interventions