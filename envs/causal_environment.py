import os
from functools import partial
from collections import namedtuple
from sklearn import metrics

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import torch
import networkx as nx
import igraph as ig
import causaldag as cd

try:
    from jax import random
except:
    print('jax is not installed')

import cdt

from envs.samplers import D
from utils import shd, auroc, sid

try:
    from models.dibs.models.nonlinearGaussian import DenseNonlinearGaussianJAX
except:
    print('jax is not installed')

PRESETS = ["chain", "collider", "fork", "random"]
NOISE_TYPES = ["gaussian", "isotropic-gaussian", "exponential", "gumbel"]
VARIABLE_TYPES = ["gaussian", "non-gaussian", "categorical"]

Data = namedtuple("Data", ["samples", "intervention_node"])


class CausalEnvironment(torch.utils.data.Dataset):

    """ Base class for generating different graphs and performing ancestral sampling"""

    def __init__(
        self,
        num_nodes,
        num_edges,
        noise_type,
        num_samples,
        mu_prior=None,
        sigma_prior=None,
        seed=None,
        nonlinear = False,
        logger=None
    ):
        self.allow_cycles = False
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        assert (
            noise_type in NOISE_TYPES
        ), "Noise types must correspond to {} but got {}".format(
            NOISE_TYPES, noise_type
        )
        self.noise_type = noise_type
        self.num_samples = num_samples
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.seed = seed
        self.nonlinear = nonlinear
        self.logger = logger

        if seed is not None:
            self.reseed(seed)

        self.init_sampler()

        if self.nonlinear:
            self.conditionals = DenseNonlinearGaussianJAX(obs_noise = self._noise_std, sig_param = 1.0, hidden_layers=[5,])

        self.sample_weights()
        self.build_graph()

        self.held_out_data = self.sample(1000).samples

    def reseed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        try:
            self.rng_jax = random.PRNGKey(seed)
        except:
            print("No JAX")

    def __getitem__(self, index):
        raise NotImplementedError

    def build_graph(self):
        """ Initilises the adjacency matrix and the weighted adjacency matrix"""

        self.adjacency_matrix = nx.to_numpy_matrix(self.graph)

        if self.nonlinear:
            self.weighted_adjacency_matrix = None
        else:
            self.weighted_adjacency_matrix = self.adjacency_matrix.copy()
            edge_pointer = 0
            for i in nx.topological_sort(self.graph):
                parents = list(self.graph.predecessors(i))
                if len(parents) == 0:
                    continue
                else:
                    for j in parents:
                        self.weighted_adjacency_matrix[j, i] = self.weights[edge_pointer]
                        edge_pointer += 1

        print("GT causal graph")
        print(self.adjacency_matrix.astype(np.uint8))


    def init_sampler(self, graph=None):
        if graph is None:
            graph = self.graph

        if self.noise_type.endswith("gaussian"):
            # Identifiable
            if self.noise_type == "isotropic-gaussian":
                self._noise_std = [self.noise_sigma] * self.num_nodes
            elif self.noise_type == "gaussian":
                self._noise_std = np.linspace(0.1, 1.0, self.num_nodes)
            for i in range(self.num_nodes):
                graph.nodes[i]["sampler"] = D(
                    self.rng.normal, loc=0.0, scale=self._noise_std[i]
                )

        elif self.noise_type == "exponential":
            noise_std = [self.noise_sigma] * self.num_nodes
            for i in range(self.num_nodes):
                graph.nodes[i]["sampler"] = D(self.rng.exponential, scale=noise_std[i])

        return graph

    def sample_weights(self):
        """Sample the edge weights"""
        if self.nonlinear:
            self.rng_jax, subk = random.split(self.rng_jax)
            self.weights = self.conditionals.sample_parameters(key = subk, n_vars = self.num_nodes)
        else:
            if self.mu_prior is not None:
                # self.weights = torch.distributions.normal.Normal(self.mu_prior, self.sigma_prior).sample([self.num_edges])
                self.weights = D(self.rng.normal, self.mu_prior, self.sigma_prior).sample(
                    size=self.num_edges
                )
            else:
                dist = D(self.rng.uniform, -5, 5)
                self.weights = torch.zeros(self.num_edges)
                for k in range(self.num_edges):
                    sample = 0.0
                    while sample > -0.5 and sample < 0.5:
                        sample = dist.sample(size=1)
                        self.weights[k] = sample

    def sample_linear(self, num_samples, graph=None, node=None, value_sampler=None):
        """Sample observations given a graph
        num_samples: Scalar
        graph: networkx DiGraph
        node: If intervention is performed, specify which node
        value: value set to node after intervention

        Outputs: Observations [num_samples x num_nodes]
        """

        if graph is None:
            graph = self.graph

        samples = np.zeros((num_samples, self.num_nodes))
        edge_pointer = 0
        for i in nx.topological_sort(graph):
            if i == node:
                noise = value_sampler.sample(num_samples)
            else:
                noise = self.graph.nodes[i]["sampler"].sample(num_samples)
            parents = list(graph.predecessors(i))
            if len(parents) == 0:
                samples[:, i] = noise
            else:
                curr = 0.0
                for j in parents:
                    curr += self.weighted_adjacency_matrix[j, i] * samples[:, j]
                    edge_pointer += 1
                curr += noise
                samples[:, i] = curr

        return Data(samples=samples, intervention_node=-1)

    def intervene(self, iteration, num_samples, node, value_sampler, _log = True):
        """Perform intervention to obtain a mutilated graph"""

        mutated_graph = self.adjacency_matrix.copy()
        mutated_graph[:, node] = 0.0  # Cut off all the parents

        if self.nonlinear:
            samples = self.sample_nonlinear(
            num_samples,
            self.init_sampler(nx.DiGraph(mutated_graph)),
            node,
            value_sampler,
        ).samples
        else:
            samples = self.sample_linear(
            num_samples,
            self.init_sampler(nx.DiGraph(mutated_graph)),
            node,
            value_sampler,
        ).samples

        if _log:
            self.logger.log_interventions(iteration, [node]*num_samples, samples[:, node])

        return Data(samples=samples, intervention_node=node)

    def sample_nonlinear(self, num_samples, graph = None, node = None, value_sampler = None):
        self.rng_jax, subk = random.split(self.rng_jax)
        if graph is None:
            graph = self.graph
        mat = nx.to_numpy_matrix(graph)
        g = ig.Graph.Weighted_Adjacency(mat.tolist())
        samples = self.conditionals.sample_obs(key=subk, n_samples=num_samples, g=g, theta=self.weights, node=node, value_sampler = value_sampler)
        return Data(samples=samples, intervention_node=-1)

    def sample(self, num_samples):
        if self.nonlinear:
            return self.sample_nonlinear(num_samples)
        else:
            return self.sample_linear(num_samples)


    def __len__(self):
        return self.num_samples

    def plot_graph(self, path, A=None, scores=None, dashed_cpdag=True, ax=None, legend=True, save=True):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        if A is None:
            A = self.adjacency_matrix

        try:
            cpdag = cd.DAG().from_amat(amat=A).cpdag().to_amat()[0]
        except:
            cpdag = None

        G = nx.DiGraph(A)
        g = nx.convert_matrix.from_numpy_matrix(A, create_using=nx.DiGraph)

        options = {
            "font_size": 28,
            "font_color": "white",
            "node_size": 3000,
            # "node_color": "white",
            "edgecolors": "black",
            "linewidths": 5,
            "width": 5,
        }

        pos = {}
        labels = {}
        r = 1
        for i, n in enumerate(range(A.shape[0])):
            theta = np.deg2rad(i * 360 / A.shape[0])
            x, y = r * np.sin(theta), r * np.cos(theta)
            pos[n] = (x, y)
            labels[n] = f"{n+1}"

        edges = G.edges()
        CPDAG_A = np.zeros(A.shape)
        NON_CPDAG_A = np.zeros(A.shape)
        for (i, j) in edges:
            if cpdag is not None and cpdag[i, j] == cpdag[j, i]:
                CPDAG_A[i][j] = 1
            else:
                NON_CPDAG_A[i][j] = 1

        cmap = plt.cm.plasma

        nodes = nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=g.nodes(),
            node_color=scores,
            node_size=2000,
            edgecolors="black",
            linewidths=5,
            cmap="coolwarm",
        )
        nx.draw_networkx_labels(g, pos, labels, font_color="white")

        nx.draw_networkx_edges(
            nx.convert_matrix.from_numpy_matrix(NON_CPDAG_A, create_using=nx.DiGraph),
            pos,
            style="solid",
            node_size=1000,
            width=5,
            arrowsize=20,
            connectionstyle="arc3, rad = 0.08",
        )

        collection = nx.draw_networkx_edges(
            nx.convert_matrix.from_numpy_matrix(CPDAG_A, create_using=nx.DiGraph),
            pos,
            style="dashed",
            node_size=1000,
            width=5,
            arrowsize=20,
            connectionstyle="arc3, rad = 0.08",
        )

        if dashed_cpdag and collection is not None:
            for patch in collection:
                patch.set_linestyle('--')

        ax.set_axis_off()
        if scores is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            cb = plt.colorbar(nodes, cax=cax)
            cb.outline.set_visible(False)

            cax.get_yaxis().labelpad = -60
            cax.set_ylabel("score", rotation=270)

        if legend:
            ax.legend(
                [
                    Line2D([0, 1], [0, 1], linewidth=3, linestyle="-", color="black"),
                    Line2D([0, 1], [0, 1], linewidth=3, linestyle="--", color="black"),
                ],
                [r"$\notin$ CPDAG", r"$\in$ CPDAG"],
                frameon=False,
            )

        if save:
            plt.savefig(path)

    def eshd(self, model, samples, double_for_anticausal=True, force_ensemble = False):
        shds = []

        if model.ensemble or force_ensemble:
            for i in range(len(model.all_graphs)):
                if getattr(model.all_graphs[0], 'to_amat', False):
                    G = (model.all_graphs[i].to_amat() != 0).astype(np.uint8)
                else:
                    G = np.array(model.all_graphs[i])
                shds.append(cdt.metrics.SHD(self.adjacency_matrix.copy(), np.array(G), double_for_anticausal=double_for_anticausal))
        else:
            Gs = model.sample(samples)
            for G in Gs:
                shds.append(cdt.metrics.SHD(self.adjacency_matrix.copy(), np.array(G), double_for_anticausal=double_for_anticausal))
        return np.array(shds).mean()

    def sid(self, model, samples, force_ensemble = False):
        sids = []

        if model.ensemble or force_ensemble:
            for i in range(len(model.all_graphs)):
                if getattr(model.all_graphs[0], 'to_amat', False):
                    G = (model.all_graphs[i].to_amat() != 0).astype(np.uint8)
                else:
                    G = np.array(model.all_graphs[i])
                sids.append(sid(self.adjacency_matrix.copy(), np.array(G)))
        else:
            Gs = model.sample(samples)
            for G in Gs:
                sids.append(sid(self.adjacency_matrix.copy(), np.array(G)))
        return np.array(sids).mean()

    def auprc(self, model, samples=1000, force_ensemble = False):
        auprcs = []

        if model.ensemble or force_ensemble:
            for i in range(len(model.all_graphs)):
                if getattr(model.all_graphs[0], 'to_amat', False):
                    G = (model.all_graphs[i].to_amat() != 0).astype(np.uint8)
                else:
                    G = np.array(model.all_graphs[i])
                auprcs.append(cdt.metrics.precision_recall(np.array(self.adjacency_matrix.copy()), np.array(G))[0])
        else:
            Gs = model.sample(samples)
            for G in Gs:
                auprcs.append(cdt.metrics.precision_recall(np.array(self.adjacency_matrix.copy()), np.array(G))[0])
        return np.array(auprcs).mean()

    def auroc(self, model, samples=1000, force_ensemble = False):
        if model.ensemble or force_ensemble:
            Gs = []
            for i in range(len(model.all_graphs)):
                if getattr(model.all_graphs[0], 'to_amat', False):
                    G = (model.all_graphs[i].to_amat() != 0).astype(np.uint8)
                else:
                    G = np.array(model.all_graphs[i])
                Gs.append(G)
            Gs = np.array(Gs)
        else:
            Gs = np.array(model.sample(samples))

        return auroc(Gs, self.adjacency_matrix.copy())

    def neg_log_likelihood(self, model, data, samples = 1000, force_ensemble = False):

        if len(data) == 0:
            return -np.inf

        liks = []
        if model.ensemble or force_ensemble:
            if getattr(model.dags[0], 'to_amat', False):
                for i in range(len(model.all_graphs)):
                    liks.append(model.interventional_likelihood(graph_ix = i, data=data, interventions = None).sum())
            else:
                liks = model.interventional_likelihood(graph_ix = np.arange(len(model.all_graphs)), data=data, interventions = None, all_graphs = True).sum(axis=1)
        else:
            Gs, thetas = model.sample(samples)
            for G, theta in zip(Gs, thetas):
                liks.append(model.interventional_likelihood(graph = G, theta = theta, data=data, interventions = None).sum())
        return -np.array(liks).mean().astype(np.float64)

