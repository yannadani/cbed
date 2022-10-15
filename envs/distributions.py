import time
import numpy as np
import tqdm
import scipy
from scipy.stats import multivariate_normal
import itertools
import networkx as nx

from utils import expm_np, all_combinations
import torch

class GraphDistribution:
    """
    Class to represent distributions over graphs.
    """

    def __init__(self, n_vars, verbose=False):
        self.n_vars = n_vars
        self.verbose = verbose

    def sample_G(self, return_mat=False):
        """
        Samples graph according to distribution

        n: number of vertices
        Returns:
            g: igraph.Graph 
        """
        raise NotImplementedError

    def unnormalized_log_prob(self, g):
        """
        g: igraph.Graph object
        Returns:
            float   log p(G) + const, i.e. unnormalized
        """
        raise NotImplementedError

    def log_normalization_constant(self, all_g):
        """
        Computes normalization constant for log p(G), i.e. `Z = log(sum_G p(g))`
        all_g: list of igraph.Graph objects
        Returns:
            float
        """
        log_prob_g_unn = np.zeros(len(all_g))
        for i, g in enumerate(tqdm.tqdm(all_g, desc='p(G) log_normalization_constant', disable=not self.verbose)):
            log_prob_g_unn[i] = self.unnormalized_log_prob(g=g)
        log_prob_sum_g = scipy.special.logsumexp(log_prob_g_unn)
        return log_prob_sum_g






class UniformDAGDistributionRejection(GraphDistribution):
    """
    Uniform distribution over DAGs
    """

    def __init__(self, n_vars, verbose=False):
        super(UniformDAGDistributionRejection, self).__init__(n_vars=n_vars, verbose=verbose)
        self.n_vars = n_vars 
        self.verbose = verbose

    def sample_G(self, return_mat=False):
        """Samples uniformly random DAG"""
        while True:
            mat = np.random.choice(2, size=self.n_vars * self.n_vars).reshape(self.n_vars, self.n_vars)
            if expm_np(mat) == 0:
                if return_mat:
                    return mat
                else:
                    return nx.DiGraph(mat)

    def unnormalized_log_prob(self, g):
        """
        p(G) ~ 1
        """

        return 0.0

class GibbsUniformDAGDistribution(GraphDistribution):
    """
    Almost Uniform distribution over DAGs based on the DAG constraint
    """

    def __init__(self, n_vars, gibbs_temp=10., sparsity_factor = 0.0, verbose=False):
        super(GibbsUniformDAGDistribution, self).__init__(n_vars=n_vars, verbose=verbose)
        self.n_vars = n_vars 
        self.verbose = verbose
        self.gibbs_temp = gibbs_temp
        self.sparsity_factor = sparsity_factor
        self.z_g = None

    def sample_G(self, return_mat=False):
        """Samples almost uniformly random DAG"""
        raise NotImplementedError

    def unnormalized_log_prob(self, g):
        """
        p(G) ~ 1
        """
        mat = g
        dagness = expm_np(mat, self.n_vars)
        return -self.gibbs_temp*dagness - self.sparsity_factor*np.sum(mat)

class GibbsDAGDistributionFull(GraphDistribution):
    """
    Almost Uniform distribution over DAGs based on the DAG constraint
    """

    def __init__(self, n_vars, gibbs_temp=10., sparsity_factor = 0.0, verbose=False):
        super(GibbsDAGDistributionFull, self).__init__(n_vars=n_vars, verbose=verbose)
        assert n_vars<=4, 'Cannot use this for higher dimensional variables, Try UniformDAGDistributionRejection instead'
        self.n_vars = n_vars 
        self.verbose = verbose
        self.gibbs_temp = gibbs_temp
        self.sparsity_factor = sparsity_factor
        all_g = all_combinations(n_vars, return_adj = True) #Do not store this in interest of memory
        dagness = np.zeros(len(all_g))
        for i, j in enumerate(all_g):
            dagness[i] = expm_np(j, self.n_vars)
        self.logits = -gibbs_temp*dagness - sparsity_factor*np.sum(all_g, axis = (-1, -2))
        self.z_g = scipy.special.logsumexp(self.logits)

    def sample_G(self, return_mat=False):
        """Samples almost uniformly random DAG"""
        all_g = all_combinations(self.n_vars, return_adj = True)
        mat_id = torch.distributions.Categorical(logits = torch.tensor(self.logits)).sample()
        mat = all_g[mat_id]
        if return_mat:
            return mat
        else:
            return nx.DiGraph(mat)

    def unnormalized_log_prob(self, g):
        """
        p(G) ~ 1
        """
        mat = g
        dagness = expm_np(mat, self.n_vars)
        return -self.gibbs_temp*dagness - self.sparsity_factor*np.sum(mat)