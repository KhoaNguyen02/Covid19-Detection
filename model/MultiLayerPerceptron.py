import numpy as np
import matplotlib.pyplot as plt
from model.activation_functions import *
from model.initializer import *
from model.optimizer import *

# ignore UserWarning
import warnings
warnings.filterwarnings("ignore")

class MLP:
    """A class to implement Multi Layer Perceptron model."""

    def __init__(self, layers, n_iterations=10000, alpha=0.003,
                 lambd=0.7, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 initialize_method="xavier", function_type="relu",
                 optim_method="rms_prop"):
        """Setup the class with necessary parameters.

        Args:
            layers (list): a list containing number of nodes per layer in the model.
            n_iterations (int, optional): number of iterations to train the model. 
                    Defaults to 10000.
            alpha (float, optional): learning rate. 
                    Defaults to 0.003.
            lambd (float, optional): hyperparameter for L2 regularization. 
                    Defaults to 0.7.
            beta1 (float, optional): hyperparameter for momentum optimizer. 
                    Defaults to 0.9.
            beta2 (float, optional): hyperparameter for RMS propagation optimizer.
                    Defaults to 0.999.
            epsilon (float, optional): a value to avoid divide by zero. 
                    Defaults to 1e-8.
            initialize_method (str, optional): initialization method. 
                    Support 'zero', 'random' and 'xavier' only. 
                    Defaults to "xavier".
            function_type (str, optional): activation function. 
                    Support 'sigmoid', 'relu', 'tanh', and 'softmax' only. 
                    Defaults to "relu".
            optim_method (str, optional): optimization method. 
                    Support 'sgd', 'momentum', and 'rms_prop' only. 
                    Defaults to "rms_prop".
        """
        self.layers = layers
        self.n_hidden_layers = len(layers) - 2
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.lambd = lambd
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.initialize_method = initialize_method
        self.function_type = function_type
        self.optim_method = optim_method
        self.A_cache, self.Z_cache = [], []

    def initialize_parameters(self):
        """Initialize parameters for training process.

        Returns:
            (tuple): a tuple containing weights and bias for the model.
        """
        initialize = Initialize(self.layers, self.initialize_method)
        self.weights, self.bias = initialize.get_params()
        return self.weights, self.bias

    def forward(self, a_prev, w, b, activation_func):
        """Compute activation value for current layer.

        Args:
            a_prev (numpy array): activation of previous layer.
            w (numpy array): weights.
            b (numpy array)): bias.
            activation (class): type of activation function.

        Returns:
            (numpy array): computed activation.
        """
        # Compute activations
        z = np.dot(w, a_prev) + b
        a = activation_func(z)

        # Store activations for backward propagation process.
        self.A_cache.append(a_prev)
        self.Z_cache.append(z)
        return a

    def forward_propagation(self, X):
        """Run the process of feeding data forward.

        Args:
            X (numpy array): data to be fed into the model.

        Returns:
            (numpy array): output of the feed-forward process.
        """
        # Initialize activation function.
        self.activation_function = Activation(self.function_type)
        a = X

        # Begin forward propagation process.
        for i in range(self.n_hidden_layers):
            a_prev = a
            a = self.forward(a_prev, self.weights[i], self.bias[i],
                             activation_func=self.activation_function)

        # Compute activation for last layer using sigmoid function.
        a = self.forward(a, self.weights[self.n_hidden_layers],
                         self.bias[self.n_hidden_layers],
                         activation_func=Activation("sigmoid"))
        return a

    def compute_cost(self, y, a):
        """Compute cost of the model.

        Args:
            y (numpy array): target data.
            a (numpy array): produced activations.

        Returns:
            (float): computed cost.
        """
        # Compute cross entropy cost.
        cross_entropy_cost = -1/self.size * np.sum(np.dot(y, np.log(a).T) +
                                                   np.dot((1-y), np.log(1-a).T))

        # Compute L2 regularization penalty.
        total_l2 = 0
        for i in range(self.weights.shape[0]):
            total_l2 += np.sum(self.weights[i] ** 2)
        l2_regularization = self.lambd / (2 * self.size) * total_l2

        # Compute cross entropy cost with L2 regularization.
        cost = cross_entropy_cost + l2_regularization
        return cost

    def backward(self, da_prev, a_prev, w, z, activation_func):
        """Compute derivative of activation, weights and bias in current layer 
        after running feed-forward process.
        Reference:
            https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60

        Args:
            da_prev (numpy array): derivative of activation in previous layer.
            a_prev (numpy array): activation of previous layer.
            w (numpy array): weights
            b (numpy array): bias
            Z (numpy array): parameters to feed into activation function.
            activation_func (class): type of activation function.

        Returns:
            (tuple): a tuple containing derivative of activation, weights and bias.
        """
        dz = da_prev * activation_func.grad(z)
        d_weight = (1 / self.size) * np.dot(dz, a_prev.T) + \
            (self.lambd / self.size) * w
        d_bias = (1 / self.size) * np.sum(dz, axis=1, keepdims=True)
        da = np.dot(w.T, dz)
        return da, d_weight, d_bias

    def add_gradient(self, da, d_weights, d_bias):
        """Store computed derivative value for updating parameters.

        Args:
            da (numpy array): derivative of activation.
            d_weights (numpy array): derivative of weights.
            d_bias (numpy array): derivative of bias.
        """
        self.da_cache.append(da)
        self.dw_cache.append(d_weights)
        self.db_cache.append(d_bias)

    def backward_propagation(self, y, a):
        """Run the backward process to update parameters.

        Args:
            y (numpy array): target data.
            a (numpy array): activations produced by forward propagation process.
        """
        # Initialize list to store derivatives.
        self.da_cache, self.dw_cache, self.db_cache = [], [], []

        # Compute and store the backward process for last layer.
        da = - (y/a - (1 - y)/(1 - a))
        da, d_weights, d_bias = self.backward(da, self.A_cache[-1],
                                              self.weights[-1], self.Z_cache[-1],
                                              Activation("sigmoid"))
        self.add_gradient(da, d_weights, d_bias)

        # Begin backward propagation process for the remaining layers.
        for i in reversed(range(self.n_hidden_layers)):
            da, d_weights, d_bias = self.backward(da, self.A_cache[i],
                                                  self.weights[i], self.Z_cache[i],
                                                  self.activation_function)
            self.add_gradient(da, d_weights, d_bias)

    def update_parameters(self):
        """Update weights and bias for the model."""
        optim = Optimize(self.alpha, self.beta1, self.beta2, self.lambd,
                         self.epsilon, self.size, self.weights,
                         self.bias, self.dw_cache, self.db_cache)
        optim(self.optim_method)
        self.A_cache, self.Z_cache = [], []

    def fit(self, X, y):
        """Execute the learning process of Multi Layer Perceptron model.

        Args:
            X (numpy array): data to learn.
            y (numpy array): target data.
        """
        self.size = X.shape[1]
        # Initialize necessary parameters for plotting graph.
        self.cost_list, self.accuracy_list = [], []

        # Initialize weights and bias.
        self.initialize_parameters()

        # Training process
        for i in range(self.n_iterations):
            # Forward propagation.
            activations = self.forward_propagation(X)

            # Compute accuracy of the model at current iteration.
            y_pred = self.threshold_function(activations)
            accuracy = self.accuracy(y_pred, y)

            # Compute cost of the model at current iteration.
            cost = self.compute_cost(y, activations)

            # Backward propagation
            self.backward_propagation(y, activations)

            # Update weights and bias.
            self.update_parameters()

            # Print training information at current iteration.
            print("Epoch {}/{}: --| Training loss: {:.6f} --| Training accuracy: {:.6f}".format(
                i+1, self.n_iterations, cost, accuracy))

            # Store cost and accuracy for plotting graph later.
            if i % 10 == 0:
                self.cost_list.append(cost)
                self.accuracy_list.append(accuracy)

    def threshold_function(self, a):
        """A function to produce predicted labels based on given activations.

        Args:
            a (numpy array): produced activations.

        Returns:
            (numpy array): predicted labels.
        """
        y_pred = np.zeros_like(a)
        y_pred[a >= 0.5] = 1
        return y_pred

    def predict(self, X, activation_func="relu"):
        """Make prediction of a new observation.

        Args:
            X (numpy array): new observation.
            activation_func (str, optional): type of activation function. 
                Defaults to "relu".

        Returns:
            (numpy array): predicted labels.
        """
        self.function_type = activation_func
        activations = self.forward_propagation(X)
        return self.threshold_function(activations)

    def accuracy(self, y_pred, y):
        """Compute accuracy between a predicted labels and actual labels.

        Args:
            y_pred (numpy array): predicted labels.
            y (numpy array): actual labels.

        Returns:
            (float): accuracy
        """
        accuracy = np.mean(y_pred == y)
        return accuracy

    def evaluate(self, X, y):
        """Evaluate the accuracy of current model.

        Args:
            X (numpy array): data to feed in the model.
            y (numpy array): targeted data.

        Returns:
            (float): accuracy of model.
        """
        y_pred = self.predict(X)
        return self.accuracy(y_pred, y)

    def summary(self):
        """Display a summary of the model."""
        print("\t\t\tModel summary: ")
        print("--" * 30)
        print("{:<30}{:>30}".format("Number of nodes input layer:",
                                    self.layers[0]))
        print("{:<30}{:>30}".format("Number of nodes output layer:",
                                    self.layers[-1]))
        print("{:<30}{:>31}".format("Number of nodes hidden layers:",
                                    str(self.layers[1:-1])))
        print("{:<30}{:>30}".format("Number of iterations:",
                                    self.n_iterations))
        print("{:<30}{:>30}".format("Learning rate:", self.alpha))
        print("{:<30}{:>30}".format("L2 regularization parameter:",
                                    self.lambd))
        print("{:<30}{:>30}".format("Initialization method:",
                                    self.initialize_method))
        print("{:<30}{:>30}".format("Activation function:",
                                    self.function_type))
        print("{:<30}{:>30}".format("Optimization function: ",
                                    self.optim_method))

    def print_confusion_matrix(self, y_pred, y):
        """Print confusion matrix of the model.

        Args:
            y_pred (numpy array): predicted labels.
            y (numpy array): actual labels.
        """
        # Compute true positive, true negative, false positive, and false negative.
        true_positive = np.sum((y_pred == 1) & (y == 1))
        true_negative = np.sum((y_pred == 0) & (y == 0))
        false_positive = np.sum((y_pred == 1) & (y == 0))
        false_negative = np.sum((y_pred == 0) & (y == 1))

        # Compute precision, recall, and f1 score.
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = 2 * precision * recall / (precision + recall)
        print("--> Precision: {:.6f}".format(precision))
        print("--> Recall: {:.6f}".format(recall))
        print("--> F1 score: {:.6f}".format(f1_score))

        # Draw confusion matrix.
        _, ax = plt.subplots(figsize=(5, 5))
        ax.matshow([[true_positive, false_positive],
                    [false_negative, true_negative]], cmap=plt.cm.Blues)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["1", "0"])
        ax.set_yticklabels(["1", "0"])
        ax.set_title("Confusion matrix")
        ax.text(0, 0, true_positive, ha="center", va="center", color="red")
        ax.text(0, 1, false_positive, ha="center", va="center", color="red")
        ax.text(1, 0, false_negative, ha="center", va="center", color="red")
        ax.text(1, 1, true_negative, ha="center", va="center", color="red")
        plt.show()

    def plot(self):
        """Plot loss curve and training accuracy."""
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss curve.
        ax[0].plot(self.cost_list)
        ax[0].set_xlabel("Iterations (per hundreds)")
        ax[0].set_ylabel("Loss")
        ax[0].set_title("Loss curve through time")

        # Plot training accuracy.
        ax[1].plot(self.accuracy_list)
        ax[1].set_xlabel("Iterations (per hundreds)")
        ax[1].set_ylabel("Accuracy")
        ax[1].set_title("Training accuracy through time")

        # Show the plot.
        plt.show()
