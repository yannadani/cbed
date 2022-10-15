import os
import uuid
from pathlib import Path
import random
import shutil
from scipy.special import logsumexp
from collections import defaultdict
import xarray as xr
import itertools as itr
import pickle
import numpy as np
import causaldag as cd

from .posterior_model import PosteriorModel
from utils import binary_entropy
from models.dag_bootstrap_lib import utils


def logmeanexp(A, axis):
    return logsumexp(A, axis=axis)-np.log(A.shape[axis])


class DagBootstrap(PosteriorModel):
    def __init__(self, args, precision_matrix=None):
        super().__init__()
        # todo: get from config

        self.ensemble = True

        self.num_nodes = args.num_nodes
        self.num_bootstraps = 100
        self.group_interventions = args.group_interventions
        self.seed = args.seed

        self.dags = None

        # precision matrix is already known
        self.precision_matrix = precision_matrix

    def update(self, data, cleanup=True):
        uid = str(uuid.uuid4())
        tmp_path = Path("tmp/") / uid

        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(tmp_path / "dags", exist_ok=True)

        interventions = np.array(data.nodes)

        if self.group_interventions:
            idx = np.argsort(interventions)
        else:
            idx = np.array(range(len(interventions)))

        # order samples by interventions to group similar interventions together
        np.savetxt(tmp_path / "samples.csv", data.samples[idx], delimiter=" ")

        # order interventions
        interventions = interventions[idx]
        interventions[interventions != -1] += 1
        interventions.tofile(tmp_path / "interventions.csv", sep="\n", format="%d")

        utils.run_gies_boot(
            self.num_bootstraps,
            str(tmp_path / "samples.csv"),
            str(tmp_path / "interventions.csv"),
            str(tmp_path / "dags") + "/",
            self.seed,
            delete=True,
        )
        self.amats, dags = utils._load_dags(str(tmp_path / "dags") + "/", delete=True)

        cov_mat = np.linalg.inv(self.precision_matrix)
        self.dags = [utils.cov2dag(cov_mat, dag) for dag in dags]
        self.all_graphs = self.dags
        self.functional_matrix = np.ones([len(self.dags), self.num_nodes])

        # get w_gd

        # w_gd
        # log_gauss_dag_weights_unnorm = np.zeros(len(self.dags))
        # obs = np.stack([s for (s, n) in zip(data.samples, data.nodes) if n == -1])
        # log_gauss_dag_weights_unnorm = np.array([gdag.logpdf(obs).sum(axis=0) for gdag in self.dags])
        # for sample, node in tqdm(zip(data.samples, data.nodes)):
        #     if node != -1:
        #         logpdfs = np.array([gdag.logpdf(np.array([sample]), interventions={node: sample[node]}).sum(axis=0) for gdag in self.dags])
        #         log_gauss_dag_weights_unnorm += logpdfs
        # self.gauss_dag_weights = np.exp(log_gauss_dag_weights_unnorm - logsumexp(log_gauss_dag_weights_unnorm)) # equation 6

        # clean up
        if cleanup:
            shutil.rmtree(tmp_path)

    def sample(self):
        return self.amats[np.random.randint(len(self.amats))]

    def obs_entropy(self):
        pass

    def sample_interventions(self, nodes, value_samplers, nsamples):
        n_boot = len(self.dags)

        # Collect interventional samples
        # Bootstraps x Interventions x Samples x Nodes
        datapoints = np.array(
            [
                [
                    dag.sample_interventional({node: sampler}, nsamples=nsamples)
                    for node, sampler in zip(nodes, value_samplers)
                ]
                for dag in self.dags
            ]
        )

        return datapoints


    def interventional_likelihood(self, graph_ix, data, interventions):
        return self.dags[graph_ix].logpdf(data, interventions=interventions)

    def save(self, path):
        with open(os.path.join(path, "dags.pkl"), "wb") as b:
            pickle.dump(self.amats, b)
            b.close()

    def load(self, path):
        with open(os.path.join(path, "dags.pkl"), "rb") as b:
            self.amats = pickle.load(b)
            dags = [cd.DAG.from_amat(adj) for adj in self.amats]
            b.close()
        cov_mat = np.linalg.inv(self.precision_matrix)
        self.dags = [utils.cov2dag(cov_mat, dag) for dag in dags]
        self.all_graphs = self.dags
        self.functional_matrix = np.ones([len(self.dags), self.num_nodes])