from bayes_opt import BayesianOptimization, UtilityFunction
import numpy as np
from envs.samplers import Constant


class ValueAcquisitionStrategy(object):
    def __init__(self, nodes, args):
        self.nodes = nodes
        self.func_max = None
        self.max_x = None
        self.max_j = None
        self.target = None
        self.intervention_value_prior = lambda size: np.zeros((size, args.num_nodes))
        self._force_iters = None

    def __call__(self, func, n_iters=5, **kargs):
        n_iters = self._force_iters if self._force_iters is not None else n_iters

        self.values = self.intervention_value_prior(n_iters)
        self.target = np.zeros((n_iters, len(self.nodes)))
        self.extra = [{}]*len(self.nodes)

        for i in range(n_iters):
            value_samplers = [Constant(self.values[i][node]) for node in self.nodes]
            self.target[i], self.extra[i] = func(self.nodes, value_samplers, **kargs)

        self.func_max = np.amax(self.target)
        _max_x_idx, _max_j_idx = np.unravel_index(
            np.argmax(self.target, axis=None), self.target.shape
        )
        self.max_iter, self.max_x, self.max_j = (
            _max_x_idx,
            self.values[_max_x_idx][self.nodes[_max_j_idx]],
            self.nodes[_max_j_idx],
        )

        return self


class BOValueAcquisitionStrategy(ValueAcquisitionStrategy):
    def __init__(self, nodes, args):
        super().__init__(nodes=nodes, args=args)
        self.num_nodes = args.num_nodes
        self.optimizers = [
            BayesianOptimization(
                f=None,
                pbounds={node: args.node_range},
                verbose=2,
                random_state=hash(node),
            )
            for node in self.nodes
        ]
        self.utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        self.exploration_steps = args.exploration_steps

    def __call__(self, func, n_iters=5, **kargs):
        self.target = np.zeros((n_iters, len(self.nodes)))
        self.values = np.zeros((n_iters, self.num_nodes))
        self.extra = [{}]*len(self.nodes)

        for k in range(n_iters):
            next_point = []
            for i, j in enumerate(self.nodes):
                if k < self.exploration_steps:
                    _next_point = {
                        j: np.random.uniform(
                            low=self.optimizers[i].space.bounds[0][0],
                            high=self.optimizers[i].space.bounds[0][1],
                        )
                    }
                else:
                    _next_point = self.optimizers[i].suggest(self.utility)
                next_point.append(Constant(_next_point[j]))
                self.values[k, j] = _next_point[j]
            self.target[k], self.extra[k] = func(self.nodes, next_point, **kargs)
            for i, j in enumerate(self.nodes):
                try:
                    self.optimizers[i].register(
                        params={j: next_point[i].value}, target=self.target[k, i]
                    )
                except KeyError:
                    continue

        self.func_max = np.amax(self.target)
        _max_x_idx, _max_j_idx = np.unravel_index(
            np.argmax(self.target, axis=None), self.target.shape
        )
        self.max_iter, self.max_x, self.max_j = (
            _max_x_idx,
            self.values[_max_x_idx][self.nodes[_max_j_idx]],
            self.nodes[_max_j_idx],
        )
        return self


class MarginalDistValueAcquisitionStrategy(ValueAcquisitionStrategy):
    def __init__(self, nodes, args):
        super().__init__(nodes=nodes, args=args)
        self.intervention_value_prior = lambda size: np.random.normal(
            args.sample_mean, 1.5 * args.sample_std, size=(size, len(args.sample_mean))
        )


class FixedValueAcquisitionStrategy(ValueAcquisitionStrategy):
    def __init__(self, nodes, args):
        super().__init__(nodes=nodes, args=args)
        self.intervention_value_prior = lambda size: np.array(
            [args.intervention_value] * (args.num_nodes * size)
        ).reshape(size, args.num_nodes)
        self._force_iters = 1


class GridValueAcquisitionStrategy(ValueAcquisitionStrategy):
    def __init__(self, nodes, args):
        super().__init__(nodes=nodes, args=args)
        self.intervention_value_prior = lambda size: np.array(args.intervention_values * args.num_nodes).reshape(args.num_nodes, len(args.intervention_values)).T
        self._force_iters = len(args.intervention_values)

class LinspacePlot(ValueAcquisitionStrategy):
    def __init__(self, nodes, args):
        super().__init__(nodes=nodes, args=args)
        self.intervention_value_prior = np.linspace(-10,10, 200)
        self.save = True

    def __call__(self, func, n_iters = 2000, **kargs):

        self.values = self.intervention_value_prior
        n_iters = len(self.values)
        self.target = np.zeros((n_iters, len(self.nodes)))
        self.extra = [{}]*n_iters

        for i in range(n_iters):
            print(i)
            value_samplers = [Constant(self.values[i])]*len(self.nodes)
            self.target[i], self.extra[i] = func(self.nodes, value_samplers, **kargs)

        if self.save:
            import pickle as pkl
            with open("plots/MI_plot_{}.pkl".format(len(self.nodes)),"wb") as b:
                pkl.dump(self.target, b)

        return self