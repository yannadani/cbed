import os
import uuid
from pathlib import Path

from shutil import rmtree

import networkx as nx

import cdt
from cdt.utils.R import launch_R_script, RPackages

import numpy as np
from .utils import adj_mat_to_vec
from sklearn import metrics


def create_tmp(base):
    uid = str(uuid.uuid4())
    tmp_path = Path(base) / uid
    os.makedirs(tmp_path, exist_ok=True)
    return tmp_path


def shd(B_est, B_true):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive

        Taken from https://github.com/xunzheng/notears
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError("B_est should take value in {0,1,-1}")
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError("undirected edge should only appear once")
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError("B_est should take value in {0,1}")
        # if not is_dag(B_est):
        #    raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    shd_wc = shd + len(pred_und)
    prc = float(len(true_pos)) / max(
        float(len(true_pos) + len(reverse) + len(false_pos)), 1.0
    )
    rec = tpr
    return {
        "fdr": fdr,
        "tpr": tpr,
        "fpr": fpr,
        "prc": prc,
        "rec": rec,
        "shd": shd,
        "shd_wc": shd_wc,
        "nnz": pred_size,
    }


def auroc(samples, gt):
    """Compute the AUROC of the model as given in
        https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009202

        Args:
        samples (numpy.ndarray): [n_samples, d, d] or [d,d] in case of a single sample
        gt (numpy.ndarray): [d, d] ground truth adjacency matrix

        Returns:
        auroc (float)
    """
    gt = adj_mat_to_vec(np.expand_dims(gt,0))
    if len(samples.shape) <3 :
        samples = np.expand_dims(samples, 0)
    samples = adj_mat_to_vec(samples)

    samples_mean = np.mean(samples, axis=0)
    sorted_beliefs_index = np.argsort(samples_mean)[::-1]
    fpr = np.zeros((samples_mean.shape[-1]))
    tpr = np.zeros((samples_mean.shape[-1]))
    for i in range(samples_mean.shape[-1]):
        indexes = np.zeros((samples_mean.shape[-1]))
        indexes[sorted_beliefs_index[:i]] = 1
        tp = np.sum(np.logical_and(gt == 1, indexes == 1))
        fn = np.sum(np.logical_and(indexes == 0, gt != indexes))
        tn = np.sum(np.logical_and(gt == 0, indexes == 0))
        fp = np.sum(np.logical_and(indexes == 1, gt != indexes))
        fpr[i] = float(fp) / (fp + tn)
        tpr[i] = float(tp) / (tp + fn)
    auroc = metrics.auc(fpr, tpr)
    return auroc


def sid(target, pred):
    """Compute the Strutural Intervention Distance.
    [R wrapper] The Structural Intervention Distance (SID) is a new distance
    for graphs introduced by Peters and Bühlmann (2013). This distance was
    created to account for the shortcomings of the SHD metric for a causal
    sense.
    It consists in computing the path between all the pairs of variables, and
    checks if the causal relationship between the variables is respected.
    The given graphs have to be DAGs for the SID metric to make sense.
    Args:
        target (numpy.ndarray or networkx.DiGraph): Target graph, must be of
            ones and zeros, and instance of either numpy.ndarray or
            networkx.DiGraph. Must be a DAG.
        prediction (numpy.ndarray or networkx.DiGraph): Prediction made by the
            algorithm to evaluate. Must be a DAG.
    Returns:
        int: Structural Intervention Distance.
            The value tends to zero as the graphs tends to be identical.
    .. note::
        Ref: Structural Intervention Distance (SID) for Evaluating Causal Graphs,
        Jonas Peters, Peter Bühlmann: https://arxiv.org/abs/1306.1043
    Examples:
        >>> from cdt.metrics import SID
        >>> from numpy.random import randint
        >>> tar = np.triu(randint(2, size=(10, 10)))
        >>> pred = np.triu(randint(2, size=(10, 10)))
        >>> SID(tar, pred)
   """
    if not RPackages.SID:
        raise ImportError("SID R package is not available. Please check your installation.")

    true_labels = cdt.metrics.retrieve_adjacency_matrix(target)
    predictions = cdt.metrics.retrieve_adjacency_matrix(pred, target.nodes()
                                            if isinstance(target, nx.DiGraph) else None)

    tmp_path = create_tmp('tmp/')

    def retrieve_result():
        return np.loadtxt(f'{tmp_path}/result.csv')

    try:
        np.savetxt(f'{tmp_path}/target.csv', true_labels, delimiter=',')
        np.savetxt(f'{tmp_path}/pred.csv', predictions, delimiter=',')
        sid_score = launch_R_script("{}/R_templates/sid.R".format(os.path.dirname(os.path.realpath(cdt.utils.__file__))),
                                    {"{target}": f'{tmp_path}/target.csv',
                                     "{prediction}": f'{tmp_path}/pred.csv',
                                     "{result}": f'{tmp_path}/result.csv'},
                                    output_function=retrieve_result)
    # Cleanup
    except Exception as e:
        rmtree(tmp_path)
        raise e
    except KeyboardInterrupt:
        rmtree(tmp_path)
        raise KeyboardInterrupt

    rmtree(tmp_path)
    return sid_score