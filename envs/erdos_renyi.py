import numpy as np
from .causal_environment import CausalEnvironment
import networkx as nx
import graphical_models
from .utils import expm_np, num_mec


class ErdosRenyi(CausalEnvironment):
	"""Generate erdos renyi random graphs using networkx's native random graph builder
		Args:
		num_nodes - Number of Nodes in the graph
		exp_edges - Expected Number of edges in Erdos Renyi graph
		noise_type - Type of exogenous variables
		noise_sigma - Std of the noise type
		num_sampels - number of observations
		mu_prior - prior of weights mean(gaussian)
		sigma_prior - prior of weights sigma (gaussian)
		seed - random seed for data
	"""


	def __init__(self, num_nodes, exp_edges = 1, noise_type='isotropic-gaussian', noise_sigma = 1.0, num_samples=1000, mu_prior = 2.0, sigma_prior = 1.0, seed = 10, nonlinear = False, logger = None):
		self.noise_sigma = noise_sigma
		p = float(exp_edges)/ (num_nodes-1)
		acyclic = 0
		mmec = 0
		count = 1
		while not (acyclic and mmec):
			if exp_edges <= 2:
				self.graph = nx.generators.random_graphs.fast_gnp_random_graph(num_nodes, p, directed = True, seed = seed*count)
			else:
				self.graph = nx.generators.random_graphs.gnp_random_graph(num_nodes, p, directed = True, seed = seed*count)
			acyclic = expm_np(nx.to_numpy_matrix(self.graph), num_nodes) == 0
			if acyclic:
				mmec = num_mec(self.graph) >=2
			count += 1

		print(f"MEC SIZE: {num_mec(self.graph)}")

		super().__init__(num_nodes, len(self.graph.edges), noise_type, num_samples, mu_prior = mu_prior , sigma_prior = sigma_prior, seed = seed, nonlinear = nonlinear, logger = logger)

		self.reseed(self.seed)
		self.init_sampler()

		self.dag = graphical_models.DAG.from_nx(self.graph)
		self.nodes = self.dag.nodes
		self.arcs = self.dag.arcs

	def __getitem__(self, index):
		return self.samples[index]

	def dag(self):
		return graphical_models.DAG.from_nx(self.graph)

