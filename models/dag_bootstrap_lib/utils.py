from __future__ import division  # in case python2 is used

import os
import shutil
import numpy as np
#import config
import pandas as pd
import networkx as nx
from networkx.utils import powerlaw_sequence
# from sksparse.cholmod import cholesky  # this has to be used instead of scipy's because it doesn't permute the matrix
from scipy import sparse
import operator as op
import causaldag as cd
import random
from scipy.special import logsumexp


def bernoulli(p):
    return np.random.binomial(1, p)


def RAND_RANGE():
    return np.random.uniform(.25, 1) * (-1 if bernoulli(.5) else 1)


def run_gies_boot(n_boot, data_path, intervention_path, dags_path, seed, delete=False):
    if delete:
        try:
            shutil.rmtree(dags_path)
            print('All DAGs deleted in ' + dags_path)
        except FileNotFoundError as e:
            pass
    if not os.path.exists(dags_path):
        os.mkdir(dags_path)
    rfile = os.path.join('models', 'dag_bootstrap_lib', 'run_gies.r')
    r_command = 'Rscript {} {} {} {} {} {}'.format(rfile, n_boot, data_path, intervention_path, seed, dags_path)
    os.system(r_command)


def _write_data(data, samples_path, interventions_path):
    """
    Helper function to write interventional data to files so that it can be used by R
    """
    # clear current data
    open(samples_path, 'w').close()
    open(interventions_path, 'w').close()

    iv_nodes = []
    for iv_node, samples in data.items():
        with open(samples_path, 'ab') as f:
            np.savetxt(f, samples)
        iv_nodes.extend([iv_node+1 if iv_node != -1 else -1]*len(samples))
    pd.Series(iv_nodes).to_csv(interventions_path, index=False)


def generate_DAG(p, m=4, prob=0., type_='config_model'):
    if type_ == 'config_model':
        z = [int(e) for e in powerlaw_sequence(p)]
        if np.sum(z) % 2 != 0:
            z[0] += 1
        G = nx.configuration_model(z)
    elif type_ == 'barabasi':
        G = nx.barabasi_albert_graph(p, m)
    elif type_ == 'small_world':
        G = nx.watts_strogatz_graph(p, m, prob)
    elif type_ == 'chain':
        source_node = int(np.ceil(p/2)) - 1
        arcs = {(i+1, i) for i in range(source_node)} | {(i, i+1) for i in range(source_node, p-1)}
        print(source_node, arcs)
        return cd.DAG(nodes=set(range(p)), arcs=arcs)
    elif type_ == 'chain_one_direction':
        return cd.DAG(nodes=set(range(p)), arcs={(i, i+1) for i in range(p-1)})
    else:
        raise Exception('Not a graph type')
    G = nx.Graph(G)
    dag = cd.DAG(nodes=set(range(p)))
    for i, j in G.edges:
        if i != j:
            dag.add_arc(*sorted((i, j)))
    return dag


def get_precision_interventional(gdag, iv_node, iv_variance):
    adj = gdag.weight_mat.copy()
    adj[:, iv_node] = 0
    vars = gdag.variances.copy()
    vars[iv_node] = iv_variance
    id_ = np.identity(adj.shape[0])
    return (id_ - adj) @ np.diag(vars**-1) @ (id_ - adj).T


def get_covariance_interventional(gdag, iv_node, iv_variance):
    adj = gdag.weight_mat.copy()
    adj[:, iv_node] = 0
    vars = gdag.variances.copy()
    vars[iv_node] = iv_variance
    id_ = np.identity(adj.shape[0])

    id_min_adj_inv = np.linalg.inv(id_ - adj)
    return id_min_adj_inv.T @ np.diag(vars) @ id_min_adj_inv


def cross_entropy_interventional(gdag1, gdag2, iv_node, iv_variance):
    precision2 = get_precision_interventional(gdag2, iv_node, iv_variance)
    covariance1 = get_covariance_interventional(gdag1, iv_node, iv_variance)
    p = len(gdag1.nodes)
    kl_term = -p/2
    kl_term += np.trace(precision2 @ covariance1)/2
    logdet2 = np.sum(np.log(gdag2.variances)) - np.log(gdag2.variances[iv_node]) + np.log(iv_variance)
    logdet1 = np.sum(np.log(gdag1.variances)) - np.log(gdag1.variances[iv_node]) + np.log(iv_variance)
    kl_term += (logdet2 - logdet1)/2
    entropy_term = (np.log(2*np.pi*np.e) * p + logdet1) / 2

    return -1 * (kl_term + entropy_term)


def _load_dags(dags_path, delete=True):
    """
    Helper function to load the DAGs generated in R
    """
    adj_mats = []
    paths = os.listdir(dags_path)
    for file_path in paths:
        if 'score' not in file_path and '.DS_Store' not in file_path:
            adj_mat = pd.read_csv(os.path.join(dags_path, file_path))
            adj_mats.append(adj_mat.values)
            if delete:
                os.remove(os.path.join(dags_path, file_path))
    return adj_mats, [cd.DAG.from_amat(adj) for adj in adj_mats]


def probability_shrinkage(prob):
    return 2 * min(1 - prob, prob)


def entropy_shrinkage(prob):
    if prob == 0 or prob == 1:
        return 0
    return (prob * np.log(prob) + (1 - prob) * np.log(1 - prob)) / np.log(2)


# def prec2dag(prec, node_order):
#     p = prec.shape[0]
#
#     # === permute precision matrix into correct order for LDL
#     prec = prec.copy()
#     rev_node_order = list(reversed(node_order))
#     prec = prec[rev_node_order]
#     prec = prec[:, rev_node_order]
#
#     # === perform ldl decomposition and correct for floating point errors
#     factor = cholesky(sparse.csc_matrix(prec))
#     l, d = factor.L_D()
#     l = l.todense()
#     d = d.todense()
#
#     # === permute back
#     inv_rev_node_order = [i for i, j in sorted(enumerate(rev_node_order), key=op.itemgetter(1))]
#     l = l.copy()
#     l = l[inv_rev_node_order]
#     l = l[:, inv_rev_node_order]
#     d = d.copy()
#     d = d[inv_rev_node_order]
#     d = d[:, inv_rev_node_order]
#
#     amat = np.eye(p) - l
#     variances = np.diag(d) ** -1
#
#     return cd.GaussDAG.from_amat(amat, variances=variances)


# def cov2dag(cov_mat, dag):
#     # See formula https://arxiv.org/pdf/1303.3216.pdf pg. 17
#     nodes = dag.nodes
#     p = len(nodes)
#     amat = np.zeros((p, p))
#     variances = np.zeros(p)
#     for node in nodes:
#         node_parents = list(dag.parents[node])
#         if len(node_parents) == 0:
#             variances[node] = cov_mat[node, node]
#         else:
#             S_k_k = cov_mat[node, node]
#             S_k_pa = cov_mat[node, node_parents]
#             S_pa_pa = cov_mat[np.ix_(node_parents, node_parents)]
#             if len(node_parents) > 1:
#                 inv_S_pa_pa = np.linalg.inv(S_pa_pa)
#             else:
#                 inv_S_pa_pa = np.array(1 / S_pa_pa)
#             node_mle_coefficents = S_k_pa.dot(inv_S_pa_pa)
#             error_mle_variance = S_k_k - S_k_pa.dot(inv_S_pa_pa.dot(S_k_pa.T))
#             variances[node] = error_mle_variance
#             amat[node_parents, node] = node_mle_coefficents
#     return cd.GaussDAG.from_amat(amat, variances=variances)


def cov2dag(cov_mat, dag):
    # See formula https://arxiv.org/pdf/1303.3216.pdf pg. 17
    nodes = dag.nodes
    p = len(nodes)
    amat = np.zeros((p, p))
    variances = np.zeros(p)
    for node in nodes:
        node_parents = list(dag.parents[node])
        if len(node_parents) == 0:
            variances[node] = cov_mat[node, node]
        else:
            S_k_k = cov_mat[node, node]
            S_k_pa = cov_mat[node, node_parents]
            S_pa_pa = cov_mat[np.ix_(node_parents, node_parents)]
            if len(node_parents) > 1:
                inv_S_pa_pa = np.linalg.inv(S_pa_pa)
            else:
                inv_S_pa_pa = np.array(1 / S_pa_pa)
            node_mle_coefficents = S_k_pa.dot(inv_S_pa_pa)
            error_mle_variance = S_k_k - S_k_pa.dot(inv_S_pa_pa.dot(S_k_pa.T))
            # clip variances - is this correct?
            variances[node] = np.clip(error_mle_variance, 0, np.abs(error_mle_variance)) + 1e-6
            amat[node_parents, node] = node_mle_coefficents
    return cd.GaussDAG.from_amat(amat, variances=variances)


def dag_posterior(dag_collec, data, intervention_nodes, interventions):
    logpdfs = np.zeros(len(dag_collec))
    for dag_ix, cand_dag in enumerate(dag_collec):
        for iv, samples in data.items():
            if iv == -1:
                logpdfs[dag_ix] += cand_dag.logpdf(samples).sum()
            else:
                iv_ix = intervention_nodes.index(iv)
                logpdfs[dag_ix] += cand_dag.logpdf(samples, interventions={iv: interventions[iv_ix]}).sum()
    return np.exp(logpdfs - logsumexp(logpdfs))


if __name__ == '__main__':
    import numpy as np
    import causaldag as cd
    from utils.graph_utils import cross_entropy_interventional, get_covariance_interventional, get_precision_interventional
    from scipy import stats

    amat1 = np.array([
        [0, 2, 3],
        [0, 0, 5],
        [0, 0, 0]
    ])
    g1 = cd.GaussDAG.from_amat(amat1, variances=[2, 2, 2])

    amat2 = np.array([
        [0, 3, 3],
        [0, 0, 5],
        [0, 0, 0]
    ])
    g2 = cd.GaussDAG.from_amat(amat2)

    iv_variance = .1
    actual = cross_entropy_interventional(g1, g2, 0, iv_variance)
    g1_samples = g1.sample_interventional({0: cd.GaussIntervention(mean=0, variance=iv_variance)}, 1000000)
    g2_logpdfs = g2.logpdf(g1_samples, {0: cd.GaussIntervention(mean=0, variance=iv_variance)})
    print('approx', g2_logpdfs.mean())
    print('actual', actual)

    cov1 = get_covariance_interventional(g1, 0, iv_variance)
    cov2 = get_covariance_interventional(g2, 0, iv_variance)

    p = 3
    .5 * (-p + np.trace(np.linalg.inv(cov2).dot(cov1)) + np.log(np.linalg.det(cov2) - np.log(np.linalg.det(cov1))) + np.log(np.linalg.det(2 * np.pi * np.e * cov1) ))

    samples = stats.multivariate_normal(cov=cov1).rvs(1000000)
    logpdfs = stats.multivariate_normal(cov=cov2).logpdf(samples)
    cd_samples = g1.sample_interventional({0: cd.GaussIntervention(mean=0, variance=iv_variance)}, 1000000)
    logpdfs_cd_samples = stats.multivariate_normal(cov=cov2).logpdf(cd_samples)
    print('scipy approx', logpdfs.mean())
    print('scipy approx of my samples', logpdfs_cd_samples.mean())

    print(np.cov(samples, rowvar=False))
    print(np.cov(cd_samples, rowvar=False))

    print(cross_entropy_interventional(g1, g1, 0, iv_variance))
    # print(entropy_interventional(g1, 0, iv_variance))
    # entropy_us1 = entropy_interventional(g1, 0, iv_variance)
    # entropy_us2 = entropy_interventional(g2, 0, iv_variance)
    # print(entropy_us1 - entropy_us2)

    entropy_scipy1 = stats.multivariate_normal(cov=cov1).entropy()
    entropy_scipy2 = stats.multivariate_normal(cov=cov2).entropy()
    print(entropy_scipy1)
    print(entropy_scipy1 - entropy_scipy2)

    g2.logpdf(samples)

    ############### TEST I MENTIONED ON FACEBOOK ###############
    import numpy as np
    import causaldag as cd
    from utils.graph_utils import cross_entropy_interventional, get_covariance_interventional, get_precision_interventional
    from scipy import stats

    amat1 = np.array([
        [0, 2, 3],
        [0, 0, 5],
        [0, 0, 0]
    ])
    g1 = cd.GaussDAG.from_amat(amat1, variances=[2, 2, 2])

    amat2 = np.array([
        [0, 3, 3],
        [0, 0, 5],
        [0, 0, 0]
    ])
    g2 = cd.GaussDAG.from_amat(amat2)
    iv_variance = .1

    g1_samples = g1.sample_interventional({0: cd.GaussIntervention(mean=0, variance=iv_variance)}, 100000)
    g2_logpdfs = g2.logpdf(g1_samples, {0: cd.GaussIntervention(mean=0, variance=iv_variance)})

    cov1 = get_covariance_interventional(g1, 0, iv_variance)
    cov2 = get_covariance_interventional(g2, 0, iv_variance)

    samples = stats.multivariate_normal(cov=cov1).rvs(100000)
    logpdfs = stats.multivariate_normal(cov=cov2).logpdf(samples)

    avg_using_causal_dag_sampler = np.mean(stats.multivariate_normal(cov=cov2).logpdf(g1_samples))
    avg_using_scipy_sampler = np.mean(stats.multivariate_normal(cov=cov2).logpdf(samples))

    print(avg_using_causal_dag_sampler - avg_using_scipy_sampler)