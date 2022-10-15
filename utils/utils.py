import networkx as nx
import causaldag as cd
from scipy import special
import numpy as np

try:
    from jax import vmap
    import jax.numpy as jnp
except RuntimeError:
    vmap = None
    jnp = None

def binary_entropy(probs):
    probs = probs.copy()
    probs[probs < 0] = 0
    probs[probs > 1] = 1
    return special.entr(probs) - special.xlog1py(1 - probs, -probs)


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
            variances[node] = error_mle_variance
            amat[node_parents, node] = node_mle_coefficents
    return cd.GaussDAG.from_amat(amat, variances=variances)

def adj_mat_to_vec_single_jax(matrix_full):
    num_nodes = np.shape(matrix_full)[-1]
    upper_tria = jnp.triu_indices(n = num_nodes, k =1)
    lower_tria = jnp.tril_indices(n = num_nodes, k = -1)

    upper_tria_el = matrix_full.at[upper_tria].get()
    lower_tria_el = matrix_full.at[lower_tria].get()

    return jnp.concatenate([upper_tria_el, lower_tria_el], axis = -1)

def adj_mat_to_vec_jax(matrix_full):
    return vmap(adj_mat_to_vec_single_jax, 0, 0)(jnp.array(matrix_full))

def adj_mat_to_vec(matrix_full):
        num_nodes = matrix_full.shape[-1]
        for xx in range(num_nodes):
            matrix_full[:,xx] = np.roll(matrix_full[:,xx], -xx, axis = -1)
        matrix = np.reshape(matrix_full[..., 1:], (matrix_full.shape[0], num_nodes*(num_nodes-1)))
        return matrix

