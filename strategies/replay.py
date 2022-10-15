from .acquisition_strategy import AcquisitionStrategy
from collections import defaultdict


class ReplayStrategy(AcquisitionStrategy):
    def set_replay(self, replay):
        self.replay = replay

    def acquire(self, nodes, iteration):
        selected_interventions = defaultdict(int)
        idx = self.replay[iteration]
        selected_interventions[idx] = self.nsamples

        return selected_interventions
