import numpy as np

from envs.samplers import Constant
from .acquisition_strategy import AcquisitionStrategy
from collections import defaultdict


class RandomAcquisitionStrategy(AcquisitionStrategy):
    def acquire(self, nodes, iteration):
        strategy = self.get_value_strategy(nodes)
        values = strategy.intervention_value_prior(self.args.batch_size)

        idx = np.random.choice(range(len(nodes)*len(values)), size=self.args.batch_size)
        value_ids, nodes = np.unravel_index(idx, shape=(len(values), len(nodes)))
        selected_interventions = defaultdict(list)
        for i, node in enumerate(nodes):
            selected_interventions[node].append(Constant(values[value_ids[i]][node]))

        return selected_interventions


class RandomBatchAcquisitionStrategy(AcquisitionStrategy):
    def acquire(self, nodes, iteration):
        strategy = self.get_value_strategy(nodes)
        print(strategy.intervention_value_prior)
        values = strategy.intervention_value_prior(1).ravel()
        idx = np.random.choice(range(len(nodes)*len(values)))
        value_id, node = np.unravel_index(idx, shape=(len(values), len(nodes)))

        selected_interventions = defaultdict(list)
        selected_interventions[node].extend([Constant(values[value_id])]*self.batch_size)

        return selected_interventions
