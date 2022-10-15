from tqdm import tqdm
import numpy as np
import xarray as xr
from scipy.special import logsumexp
from scipy.stats import entropy


class PosteriorModel(object):
    def __init__(self):
        self.ensemble = False

    def update(self, data):
        """
        Data contains both observational
        and interventional data
        """
        # assert data should always be in numpy
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def log_prob(self, graph):
        pass

    def sample(self):
        pass

    def simulate(self, nodes):
        pass

    def params_entropy(self):
        # monte carlo
        pass

    def interventional_likelihood(self, graph_ix, data, interventions):
        pass

    def graph_entropy(self, nodes=[], values=[], nsamples=0, **kargs):
        """
        P(theta | D, xi)
        """
        n_boot = len(self.dags)

        current_logpdfs = kargs.get("current_logpdfs", np.zeros([n_boot, n_boot, nsamples]))
        logpdfs = kargs.get("logpdfs")

        # collecting samples
        datapoint_ixs = list(range(nsamples))

        intervention_scores = np.zeros(len(nodes))
        intervention_logpdfs = np.zeros([len(nodes), n_boot, n_boot, nsamples])
        for intv_ix in range(len(nodes)):
            # current number of times this intervention has already been selected
            #selected_datapoint_ixs = random.choices(datapoint_ixs, k=10)
            selected_datapoint_ixs = datapoint_ixs
            for outer_dag_ix in range(n_boot):

                # import pdb; pdb.set_trace()
                intervention_logpdfs[intv_ix, outer_dag_ix] = logpdfs.sel(
                    outer_dag=outer_dag_ix,
                    intervention_ix=intv_ix
                )

                new_logpdfs = current_logpdfs[outer_dag_ix] + intervention_logpdfs[intv_ix, outer_dag_ix]

                # W_i
                importance_weights = np.exp(new_logpdfs - logsumexp(new_logpdfs, 0)[None])
                intervention_scores[intv_ix] += entropy(importance_weights, axis=0).sum()

        intervention_scores = entropy(np.ones(n_boot)) - intervention_scores/(n_boot*nsamples)

        return intervention_scores, intervention_logpdfs

    def _update_likelihood(self, nodes, nsamples, value_samplers, datapoints):
        matrix = np.stack([
                    np.stack([
                        self.interventional_likelihood(
                            graph_ix=l,
                            data=datapoints[:, intv_ix].reshape(-1, len(nodes)),
                            interventions={nodes[intv_ix]: intervention}
                        ).reshape(len(self.dags), nsamples)
                        for l, outter_dag in enumerate(self.dags)
                    ], axis=1)
                for intv_ix, intervention in tqdm(enumerate(value_samplers), total=len(value_samplers))])

        self.interventional_likelihood(graph_ix=0, data=datapoints[:, 0].reshape(-1, len(nodes)), interventions={nodes[0]: 0}).reshape(len(self.dags), nsamples)

        logpdfs = xr.DataArray(
            matrix,
            dims=['intervention_ix', 'outer_dag', 'inner_dag', 'datapoint'],
            coords={
                'intervention_ix': list(range(len(nodes))),
                'outer_dag': list(range(len(self.dags))),
                'inner_dag': list(range(len(self.dags))),
                'datapoint': list(range(nsamples)),
            })

        return logpdfs
    def save(self, path):
        pass
    def load(self, path):
        pass