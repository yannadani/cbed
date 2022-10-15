import os

import numpy as np

from .posterior_model import PosteriorModel

from .dibs.eval.target import make_graph_model
from .dibs.models.linearGaussianEquivalent import BGe, BGeJAX
from .dibs.models.linearGaussian import LinearGaussian, LinearGaussianJAX
from .dibs.models.nonlinearGaussian import DenseNonlinearGaussianJAX
from .dibs.inference import MarginalDiBS, JointDiBS
from .dibs.kernel import (
    FrobeniusSquaredExponentialKernel,
    JointAdditiveFrobeniusSEKernel,
)
from .dibs.utils.func import (
    particle_marginal_empirical,
    particle_marginal_mixture,
    particle_joint_empirical,
    particle_joint_mixture,
)
from .dibs.utils.graph import elwise_acyclic_constr_nograd
from .dibs.utils.tree import tree_shapes, tree_select, tree_index

import jax.numpy as jnp
from jax import jit, vmap
from jax import random
from jax.tree_util import tree_map

import igraph as ig
from utils import binary_entropy
from scipy.special import logsumexp
from scipy import special
import causaldag as cd
from tqdm import tqdm
import utils
import xarray as xr
import pickle


class DiBS_BGe(PosteriorModel):
    def __init__(self, args, precision_matrix=None):
        self.key = random.PRNGKey(123)
        self.num_nodes = args.num_nodes
        self.precision_matrix = precision_matrix
        self.dags = None
        self.ensemble = False
        self.reset_after_each_update = True

        graph_model = make_graph_model(
            n_vars=args.num_nodes,
            graph_prior_str=args.dibs_graph_prior,
            edges_per_node=args.exp_edges,
        )

        inference_model = BGeJAX(
            mean_obs=jnp.zeros(args.num_nodes),
            alpha_mu=1.0,
            alpha_lambd=args.num_nodes + 2,
        )

        def log_prior(single_w_prob):
            """log p(G) using edge probabilities as G"""
            return graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

        def log_likelihood(single_w, x, interv_targets):
            log_lik = inference_model.log_marginal_likelihood_given_g(
                w=single_w, data=x, interv_targets=interv_targets
            )
            return log_lik

        self.eltwise_log_prob = vmap(
            lambda g, x, interv_targets: log_likelihood(g, x, interv_targets),
            (0, None, None),
            0,
        )

        # SVGD + DiBS hyperparams
        self.n_particles = 20
        self.n_steps = lambda t: 3000 #int(100*t/15)

        # initialize kernel and algorithm
        kernel = FrobeniusSquaredExponentialKernel(h=5.0)

        self.model = MarginalDiBS(
            kernel=kernel,
            target_log_prior=log_prior,
            target_log_marginal_prob=log_likelihood,
            alpha_linear=0.1,
        )

        self.key, subk = random.split(self.key)
        self.particles_z = self.model.sample_initial_random_particles(
            key=subk, n_particles=self.n_particles, n_vars=args.num_nodes
        )

    def update(self, data):
        data_samples = jnp.array(data.samples)
        interv_targets = jnp.zeros((data_samples.shape[0], self.num_nodes)).astype(bool)
        int_idx = np.argwhere(data.nodes >= 0)
        interv_targets = interv_targets.at[int_idx, data.nodes[int_idx]].set(True)

        if self.reset_after_each_update:
            self.key, subk = random.split(self.key)
            self.particles_z = self.model.sample_initial_random_particles(
                key=subk, n_particles=self.n_particles, n_vars=self.num_nodes
            )

        self.key, subk = random.split(self.key)
        self.particles_z = self.model.sample_particles(
            key=subk,
            n_steps=self.n_steps(data_samples.shape[0]),
            init_particles_z=self.particles_z,
            data=data_samples,
            interv_targets=interv_targets,
        )
        self.update_dist(data_samples, interv_targets)

    def sample_interventions(self, nodes, value_samplers, nsamples):

        # Collect interventional samples
        # Bootstraps x Interventions x Samples x Nodes
        cov_mat = np.linalg.inv(self.precision_matrix)
        dags = [
            utils.cov2dag(cov_mat, cd.DAG.from_amat(dag)) for dag in self.posterior[0]
        ]
        datapoints = np.array(
            [
                [
                    dag.sample_interventional({node: sampler}, nsamples=nsamples)
                    for node, sampler in zip(nodes, value_samplers)
                ]
                for dag in dags
            ]
        )

        return datapoints

    def update_dist(self, data, interv_targets):
        particles_g = self.model.particle_to_g_lim(self.particles_z)
        self.posterior = particle_marginal_empirical(particles_g)
        #self.posterior = particle_marginal_mixture(
        #    particles_g, self.eltwise_log_prob, data, interv_targets
        #)
        self.dags = self.posterior[0]


    def sample(self, num_samples):
        self.key, subk = random.split(self.key)
        sampled_particles = random.categorical(
            key=subk, logits=self.posterior[1], shape=[num_samples]
        )
        return self.model.particle_to_g_lim(self.particles_z)[sampled_particles]

    def log_prob_single(self, graph):
        particles, log_prob = self.posterior
        equal = jnp.sum(not jnp.equal(particles, graph), axis=0) == 0
        index = jnp.nonzero(equal, size=1, fill_value=-1)

        if index == -1:
            return jnp.inf
        else:
            return log_prob[index]

    def log_prob(self, graphs):
        return vmap(self.log_prob_single, 0, 0)(graphs)

    def interventional_likelihood(self, graph, data, interventions):
        graph = jnp.array(graph)
        data = jnp.array(data)
        interv_targets = jnp.zeros(data.shape[-1]).astype(bool)
        nodes = interventions.keys(0)
        interv_targets[nodes] = True
        return self.eltwise_log_prob(graph, x, interv_targets)


class DiBS_Linear(PosteriorModel):
    def __init__(self, args, precision_matrix = None):
        self.key = random.PRNGKey(123)
        self.num_nodes = args.num_nodes
        self.precision_matrix = precision_matrix
        self.ensemble = False
        self.reset_after_each_update = True

        graph_model = make_graph_model(
            n_vars=args.num_nodes,
            graph_prior_str=args.dibs_graph_prior,
            edges_per_node=args.exp_edges,
        )

        self.inference_model = LinearGaussianJAX(
            obs_noise=args.noise_sigma,
            mean_edge=0.0,
            sig_edge=2.0,
        )

        def log_prior(single_w_prob):
            """log p(G) using edge probabilities as G"""
            return graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

        def log_likelihood(single_w, single_theta, x, interv_targets, rng):
            log_prob_theta = self.inference_model.log_prob_parameters(theta=single_theta, w=single_w)
            log_lik = self.inference_model.log_likelihood(
                w=single_w, theta=single_theta, data=x, interv_targets=interv_targets
            )
            return log_prob_theta + log_lik

        self.eltwise_log_prob = vmap(
            lambda g, theta, x, interv_targets: log_likelihood(
                g, theta, x, interv_targets, None
            ),
            (0, 0, None, None),
            0,
        )

        # SVGD + DiBS hyperparams
        self.n_particles = 20
        self.n_steps = lambda t: 3000 #int(100*t/15)

        # initialize kernel and algorithm
        kernel = JointAdditiveFrobeniusSEKernel(
            h_latent=5.0, h_theta=500.0
        )

        self.model = JointDiBS(
            kernel=kernel,
            target_log_prior=log_prior,
            target_log_joint_prob=log_likelihood,
            alpha_linear=0.1,
        )

        self.key, subk = random.split(self.key)
        self.particles_z, self.particles_w = self.model.sample_initial_random_particles(
            key=subk, n_particles=self.n_particles, model = self.inference_model, n_vars=self.num_nodes
        )

    def update(self, data):
        data_samples = jnp.array(data.samples)
        interv_targets = jnp.zeros((data_samples.shape[0], self.num_nodes)).astype(bool)
        int_idx = np.argwhere(data.nodes >= 0)
        interv_targets = interv_targets.at[int_idx, data.nodes[int_idx]].set(True)

        if self.reset_after_each_update:
            self.key, subk = random.split(self.key)
            self.particles_z, self.particles_w = self.model.sample_initial_random_particles(
            key=subk, n_particles=self.n_particles, model = self.inference_model, n_vars=self.num_nodes
            )

        self.key, subk = random.split(self.key)

        self.particles_z, self.particles_w = self.model.sample_particles(
            key=subk,
            n_steps=self.n_steps(data_samples.shape[0]),
            init_particles_z=self.particles_z,
            init_particles_theta=self.particles_w,
            data=data_samples,
            interv_targets=interv_targets,
        )
        self.update_dist(data_samples, interv_targets)

    def update_dist(self, data, interv_targets):
        particles_g = self.model.particle_to_g_lim(self.particles_z)
        # self.dibs_empirical = particle_marginal_empirical(particles_g)
        self.posterior = particle_joint_mixture(
            particles_g, self.particles_w, self.eltwise_log_prob, data, interv_targets
        )
        self.dags = self.posterior[0]

    def sample(self, num_samples):
        self.key, subk = random.split(self.key)
        sampled_particles = random.categorical(
            key=subk, logits=self.posterior[2], shape=[num_samples]
        )
        return self.model.particle_to_g_lim(self.particles_z)[sampled_particles]

    def log_prob_single(self, graph):
        particles, log_prob = self.posterior
        equal = jnp.sum(not jnp.equal(particles, graph), axis=0) == 0
        index = jnp.nonzero(equal, size=1, fill_value=-1)

        if index == -1:
            return jnp.inf
        else:
            return log_prob[index]

    def log_prob(self, graphs):
        return vmap(self.log_prob_single, 0, 0)(graphs)

class DiBS_NonLinear(PosteriorModel):
    def __init__(self, args):
        self.key = random.PRNGKey(123)
        self.num_nodes = args.num_nodes
        self.ensemble = True
        self.reset_after_each_update = False

        graph_model = make_graph_model(
            n_vars=args.num_nodes,
            graph_prior_str=args.dibs_graph_prior,
            edges_per_node=args.exp_edges,
        )

        self.inference_model = DenseNonlinearGaussianJAX(obs_noise = args.noise_sigma,
            sig_param = 1.0, hidden_layers=[5,]
        )

        def log_prior(single_w_prob):
            """log p(G) using edge probabilities as G"""
            return graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

        def log_likelihood(single_w, single_theta, x, interv_targets, rng):
            log_prob_theta = self.inference_model.log_prob_parameters(theta=single_theta, w=single_w)
            log_lik = self.inference_model.log_likelihood(
                w=single_w, theta=single_theta, data=x, interv_targets=interv_targets
            )
            return log_lik + log_prob_theta


        self.eltwise_log_prob_single = vmap(
            lambda g, theta, x, interv_targets: self.inference_model.log_likelihood_single(
                w=g, theta=theta, data=x, interv_targets=interv_targets
            ),
            (0, 0, None, None),
            0,
        )

        # SVGD + DiBS hyperparams
        self.n_particles = 20
        self.n_steps = args.dibs_steps

        # initialize kernel and algorithm
        kernel = JointAdditiveFrobeniusSEKernel(
            h_latent=5.0, h_theta=500.0
        )

        self.model = JointDiBS(
            kernel=kernel,
            target_log_prior=log_prior,
            target_log_joint_prob=log_likelihood,
            alpha_linear=0.05,
        )

        self.key, subk = random.split(self.key)
        self.particles_z, self.particles_w = self.model.sample_initial_random_particles(
            key=subk, n_particles=self.n_particles, model = self.inference_model, n_vars=self.num_nodes
        )

    def update(self, data):
        data_samples = jnp.array(data.samples)

        interv_targets = jnp.zeros((data_samples.shape[0], self.num_nodes)).astype(bool)
        int_idx = np.argwhere(data.nodes >= 0)
        interv_targets = interv_targets.at[int_idx, data.nodes[int_idx]].set(True)

        if self.reset_after_each_update:
            self.key, subk = random.split(self.key)
            self.particles_z, self.particles_w = self.model.sample_initial_random_particles(
            key=subk, n_particles=self.n_particles, model = self.inference_model, n_vars=self.num_nodes
            )

        self.key, subk = random.split(self.key)

        self.particles_z, self.particles_w = self.model.sample_particles(
            key=subk,
            n_steps=self.n_steps,
            init_particles_z=self.particles_z,
            init_particles_theta=self.particles_w,
            data=data_samples,
            interv_targets=interv_targets,
        )
        self.update_dist()

    def update_dist(self):
        particles_g = self.model.particle_to_g_lim(self.particles_z)
        _posterior = particle_joint_empirical(particles_g, self.particles_w)

        is_dag = elwise_acyclic_constr_nograd(_posterior[0], self.num_nodes) == 0
        self.all_graphs = _posterior[0]
        self.dags = _posterior[0][is_dag, :, :]

        self.posterior = self.dags, tree_select(_posterior[1], is_dag), _posterior[2][is_dag] - logsumexp(_posterior[2][is_dag])
        self.full_posterior = _posterior

    def sample(self, num_samples):
        self.key, subk = random.split(self.key)
        sampled_particles = random.categorical(
            key=subk, logits=self.posterior[2], shape=[num_samples]
        )
        return self.model.particle_to_g_lim(self.particles_z)[sampled_particles], self.particles_w[sampled_particles]

    def sample_interventions(self, nodes, value_samplers, nsamples):

        # Collect interventional samples
        # Bootstraps x Interventions x Samples x Nodes
        thetas = self.posterior[1]
        #self.key, subk = random.split(self.key)
        all_dags = []
        for i, dag in enumerate(self.dags):
            theta = tree_index(thetas, i)
            all_interventions = []
            for node, sampler in zip(nodes, value_samplers):
                self.key, subk = random.split(self.key)
                all_interventions.append(self.inference_model.sample_obs(key = subk, n_samples=nsamples, g = ig.Graph.Weighted_Adjacency(dag.tolist()), theta = theta, node = node, value_sampler = sampler))
            all_dags.append(all_interventions)
        return np.array(all_dags)

    def log_prob_single(self, graph):
        particles, particles_w, log_prob = self.posterior
        equal = jnp.sum(not jnp.equal(particles, graph), axis=0) == 0
        index = jnp.nonzero(equal, size=1, fill_value=-1)

        if index == -1:
            return jnp.inf
        else:
            return log_prob[index]

    def log_prob(self, graphs):
        return vmap(self.log_prob_single, 0, 0)(graphs)

    def interventional_likelihood(self, graph_ix, data, interventions, all_graphs = False):
        if all_graphs:
            posterior = self.full_posterior
        else:
            posterior = self.posterior

        graph = posterior[0][graph_ix]
        theta = tree_index(posterior[1], graph_ix)

        data = jnp.array(data)
        interv_targets = jnp.zeros(data.shape[-1]).astype(bool)
        if interventions is not None:
            nodes = list(interventions.keys())[0]
            interv_targets = interv_targets.at[nodes].set(True)
        return self.eltwise_log_prob_single(graph, theta, data, interv_targets)

    def _update_likelihood(self, nodes, nsamples, value_samplers, datapoints):
        matrix = np.stack([
                    self.interventional_likelihood(
                        graph_ix=jnp.arange(len(self.dags)),
                        data=datapoints[:, intv_ix].reshape(-1, len(nodes)),
                        interventions={nodes[intv_ix]: intervention}
                    ).reshape(len(self.dags), len(self.dags), nsamples)
            for intv_ix, intervention in tqdm(enumerate(value_samplers), total=len(value_samplers))])
        logpdfs = xr.DataArray(
            matrix,
            dims=['intervention_ix', 'inner_dag', 'outer_dag', 'datapoint'],
            coords={
                'intervention_ix': list(range(len(nodes))),
                'inner_dag': list(range(len(self.dags))),
                'outer_dag': list(range(len(self.dags))),
                'datapoint': list(range(nsamples)),
            })

        return logpdfs

    def save(self, path):
        with open(os.path.join(path, "particles_z.pkl"), "wb") as b:
            pickle.dump(self.particles_z, b)
            b.close()
        with open(os.path.join(path, "particles_w.pkl"), "wb") as f:
            pickle.dump(self.particles_w, f)
            f.close()

    def load(self, path):
        with open(os.path.join(path, "particles_z.pkl"), "rb") as b:
            self.particles_z = pickle.load(b)
            b.close()
        self.particles_z = jnp.array(self.particles_z)
        with open(os.path.join(path, "particles_w.pkl"), "rb") as f:
            self.particles_w = pickle.load(f)
            f.close()
        self.particles_w = tree_map(lambda arr: jnp.array(arr), self.particles_w)
        self.update_dist()