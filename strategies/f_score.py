from .acquisition_strategy import AcquisitionStrategy


class FScoreBatchStrategy(AcquisitionStrategy):

    def _score_for_value(self, nodes, value_samplers):
        n_boot = len(self.model.dags)

        # DAGs x Interventions x Samples x Nodes - y[t][m]
        datapoints = self.model.sample_interventions(
            nodes, value_samplers, self.num_samples
        )

        mu_i_k = datapoints.mean(-2, keepdims=True)
        mu_k = mu_i_k.mean(0, keepdims=True)

        vbg_k = ((mu_i_k - mu_k) ** 2).sum((0, -1, -2))
        vwg_k = ((datapoints - mu_i_k) ** 2).sum((0, -1, -2))

        scores = vbg_k / vwg_k

        return scores, {}