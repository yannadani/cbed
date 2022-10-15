import json
import numpy as np
from collections import namedtuple

from .causal_environment import CausalEnvironment
import networkx as nx
import graphical_models
from pgmpy.readwrite import BIFReader

from .utils import expm_np, num_mec


Data = namedtuple("Data", ["samples", "intervention_node"])


class BifEnvironment(CausalEnvironment):

    def __init__(self, bif_file, mapping, logger):
        self.logger = logger
        self.reader = BIFReader(bif_file)
        self.model = self.reader.get_model()
        self.mapping = json.loads(mapping)

        self.num_nodes = len(self.reader.get_variables())

        self.var2id = {}
        self.id2var = {}
        id = 0
        for var in self.reader.get_variables():
            if var not in self.var2id:
                self.var2id[var] = id
                self.id2var[id] = var
                id += 1

        A = np.zeros((id, id))
        for child, parents in self.reader.get_parents().items():
            for parent in parents:
                A[self.var2id[parent], self.var2id[child]] = 1

        self.adjacency_matrix = A

        G = nx.convert_matrix.from_numpy_matrix(A, create_using=nx.DiGraph)
        self.graph = nx.relabel_nodes(G, self.id2var)

    def __getitem__(self, index):
        return self.samples[index]

    def dag(self):
        return graphical_models.DAG.from_nx(self.graph)

    def reseed(self, seed=None):
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def intervene(self, iteration, num_samples, node, value_sampler):
        """Perform intervention to obtain a mutilated graph"""

        value = value_sampler.sample(num_samples).item()

        assert num_samples == 1, 'only sample size 1 is supported'

        inverse_mapping = {}
        for key, val in self.mapping.items():
            inverse_mapping[val] = key

        value = inverse_mapping[value]
        node_name = self.id2var[node]

        data = self.model.simulate(n_samples=num_samples, do={node_name: value})
        data = data[self.reader.get_variables()]
        data = data.replace(self.mapping).to_numpy()

        self.logger.log_interventions(iteration, [node]*num_samples, data[:, node])

        return Data(samples=data, intervention_node=node)

    def sample(self, num_samples):
        data = self.model.simulate(n_samples=num_samples)
        data = data[self.reader.get_variables()]

        inverse_mapping = {}
        for key, val in self.mapping.items():
            inverse_mapping[val] = key
        data = data.replace(self.mapping).to_numpy()

        return Data(samples=data, intervention_node=-1)