import numpy as np
from .causal_environment import CausalEnvironment
import networkx as nx
import igraph as ig
import graphical_models
from .utils import expm_np, num_mec


class ScaleFree(CausalEnvironment):
	"""Generate scale free (Barbasi-Albert) random graphs
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
		acyclic = 0
		mmec = 0
		count = 1
		while not (acyclic and mmec):
			random = np.random
			random.seed(seed*count)
			perm = random.permutation(num_nodes).tolist()
			self.graph = nx.DiGraph(np.array(ig.Graph.Barabasi(n=num_nodes, m=exp_edges, directed = True).permute_vertices(perm).get_adjacency().data))
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

	def albert_barbasi(self, d, m, random=np.random):
		B = np.zeros([d, d])
		bag = [0]
		for ii in range(1, d):
			dest = random.choice(bag, size=m)
			for jj in dest:
				B[ii, jj] = 1
			bag.append(ii)
			bag.extend(dest)
		P = np.random.permutation(np.eye(d, d))  # permutes first axis only
		B_perm = P.T.dot(B).dot(P)
		return nx.DiGraph(B_perm)