import numpy as np

from .acquisition_strategy import AcquisitionStrategy
from collections import defaultdict
from envs.samplers import Constant


class ACDStrategy(AcquisitionStrategy):

    def _score_for_value(self, nodes, value_samplers, current_logpdfs=None):
        n_boot = len(self.model.dags)

        if current_logpdfs is None:
            current_logpdfs = np.zeros([n_boot, n_boot, self.num_samples])

        # DAGs x Interventions x Samples x Nodes - y[t][m]
        datapoints = self.model.sample_interventions(
            nodes, value_samplers, self.num_samples
        )

        logpdfs = self.model._update_likelihood(
            nodes, self.num_samples, value_samplers, datapoints
        )

        posterior_entropy, intervention_logpdfs = self.model.graph_entropy(
            nodes,
            value_samplers,
            self.num_samples,
            logpdfs=logpdfs,
            current_logpdfs=current_logpdfs,
        )

        _extras = {
            'intervention_logpdfs': intervention_logpdfs
        }

        return posterior_entropy, _extras


class ABCDStrategy(ACDStrategy):
    def acquire(self, nodes, iteration):
        n_boot = len(self.model.dags)
        current_logpdfs = np.zeros([n_boot, n_boot])

        # DAGs x Interventions x Samples x Nodes - y[t][m]
        strategy = self.get_value_strategy(nodes)

        targets = []
        values = []
        for _ in range(self.args.batch_size):
            strategy(
                self._score_for_value,
                n_iters =self.args.num_intervention_values,
                current_logpdfs=current_logpdfs)

            interventional_logpdf = strategy.extra[strategy.max_iter]['intervention_logpdfs']
            current_logpdfs += interventional_logpdf[strategy.max_j]

            targets.append(strategy.max_j)
            values.append(strategy.max_x)

        selected_interventions = defaultdict(list)
        for value, target in zip(values, targets):
            selected_interventions[target].append(Constant(value))

        return selected_interventions