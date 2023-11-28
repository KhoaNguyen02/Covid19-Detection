import numpy as np

class Optimize:
    """A class to optimize parameters in the model."""

    def __init__(self, alpha, beta1, beta2, lambd, epsilon, size,
                 weights, bias, dw_cache, db_cache):
        """Setup the class with necessary information.
        
        Args:
            alpha (float): learning rate.
            beta1 (float): hyperparameter for momentum optimizer.
            beta2 (float): hyperparameter for root mean squared (RMS) propagation optimizer.
            lambd (float): hyperparameter for L2 regularization.
            epsilon (float): a scailing factor to avoid extremely small value. 
            size (int): number of examples to train the model.
            weights (numpy array): weights of the model.
            bias (numpy array): bias of the model.
            dw_cache (numpy array): stored weights derivative.
            db_cache (numpy array): stored bias derivative.
        """
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambd = lambd
        self.epsilon = epsilon
        self.size = size
        self.weights = weights
        self.bias = bias
        self.dw_cache = dw_cache[::-1]
        self.db_cache = db_cache[::-1]

    def __call__(self, optim_method):
        """A method to run the optimizer.

        Args:
            optim_method (str): type of optimization.
        """
        optimization_dict = {"sgd": self.sgd(), "momentum": self.momentum(),
                             "rms_prop": self.rms_prop()}
        assert optim_method in optimization_dict, "Invalid method. {} is not available".format(
            optim_method)
        optimization_dict[optim_method]

    def sgd(self):
        """Stochastic gradient descent optimizer."""
        for i in range(self.weights.shape[0]):
            self.weights[i] -= self.alpha * \
                (self.dw_cache[i] + (self.lambd / self.size) * self.weights[i])
            self.bias[i] -= self.alpha * self.db_cache[i]

    def initialize_optimizer_params(self):
        """Initialize parameters for momentum and RMS prop optimizer."""
        self.v_dw, self.v_db = [], []
        for i in range(self.weights.shape[0]):
            self.v_dw.append(np.zeros_like(self.weights[i]))
            self.v_db.append(np.zeros_like(self.bias[i]))

    def momentum(self):
        """Momentum optimizer."""
        self.initialize_optimizer_params()
        for i in range(self.weights.shape[0]):
            self.v_dw[i] = self.beta1 * self.v_dw[i] + \
                (1 - self.beta1) * self.dw_cache[i]
            self.v_db[i] = self.beta1 * self.v_db[i] + \
                (1 - self.beta1) * self.db_cache[i]
            self.weights[i] -= self.alpha * self.v_dw[i]
            self.bias[i] -= self.alpha * self.v_db[i]

    def rms_prop(self):
        """Root mean squared propagation optimizer."""
        self.initialize_optimizer_params()
        for i in range(self.weights.shape[0]):
            v_dw = self.beta2 * self.v_dw[i] + \
                (1 - self.beta2) * self.dw_cache[i] ** 2
            v_db = self.beta2 * self.v_db[i] + \
                (1 - self.beta2) * self.db_cache[i] ** 2
            self.weights[i] -= self.alpha * v_dw / np.sqrt(v_dw + self.epsilon)
            self.bias[i] -= self.alpha * v_db / np.sqrt(v_db + self.epsilon)
