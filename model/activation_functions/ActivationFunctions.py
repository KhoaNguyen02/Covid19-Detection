import numpy as np

class ActivationFunctions:
    """An interface class to use activation functions."""
    def __init__(self):
        """Initialize current interface."""
        pass

    def __call__(self, x):
        """Compute the activation function.

        Args:
            x (numpy array): an array containing values 
            to pass through the activation function.
        """
        pass

    def grad(self, x):
        """Compute derivative of activation function.

        Args:
            x (numpy array): an array containing values 
            to pass through the activation function.
        """
        pass
    
class Sigmoid(ActivationFunctions):
    """A class to implement sigmoid function.

    Args:
        ActivationFunctions (class): inherited interface.
    """
    def __call__(self, x):
        """Compute sigmoid function.

        Args:
            x (numpy array): an array containing values 
            to pass through sigmoid function.

        Returns:
            (numpy array): an array containing computed values.
        """
        return 1 / (1 + np.exp(-x))
    
    def grad(self, x):
        """Compute derivative of sigmoid function.

        Args:
            x (numpy array): an array containing values 
            to pass through sigmoid function.

        Returns:
            (numpy array): an array containing computed values.
        """
        return self.__call__(x) * (1 - self.__call__(x))

class Relu(ActivationFunctions):
    """A class to implement rectified linear unit (relu) function.

    Args:
        ActivationFunctions (class): inherited interface.
    """
    def __call__(self, x):
        """Compute relu function.

        Args:
            x (numpy array): an array containing values 
            to pass through relu function.

        Returns:
            (numpy array): an array containing computed values.
        """
        return np.maximum(0, x)
    
    def grad(self, x):
        """Compute derivative of relu function.

        Args:
            x (numpy array): an array containing values 
            to pass through relu function.

        Returns:
            (numpy array): an array containing computed values.
        """
        x[x > 0] = 1
        x[x <= 0] = 0
        return x
    
class Tanh(ActivationFunctions):
    """A class to implement tanh function.

    Args:
        ActivationFunctions (class): inherited interface.
    """
    def __call__(self, x):
        """Compute tanh function.

        Args:
            x (numpy array): an array containing values 
            to pass through tanh function.

        Returns:
            (numpy array): an array containing computed values.
        """
        numerator = 1 - np.exp(-2 * x)
        denominator = 1 + np.exp(-2 * x)
        return numerator/denominator
    
    def grad(self, x):
        """Compute derivative of tanh function.

        Args:
            x (numpy array): an array containing values 
            to pass through tanh function.

        Returns:
            (numpy array): an array containing computed values.
        """
        return 1 - self.__call__(x) ** 2

class SoftMax(ActivationFunctions):
    """A class to implement softmax function.

    Args:
        ActivationFunctions (class): inherited interface.
    """
    def __call__(self, x):
        """Compute softmax function.

        Args:
            x (numpy array): an array containing values 
            to pass through softmax function.

        Returns:
            (numpy array): an array containing computed values.
        """
        return np.exp(x) / np.sum(np.exp(x))
    
    def grad(self, x):
        """Compute derivative of softmax function.

        Args:
            x (numpy array): an array containing values 
            to pass through softmax function.

        Returns:
            (numpy array): an array containing computed values.
        """
        return self.__call__(x) * (1 - self.__call__(x))

class Activation(ActivationFunctions):
    """A class to implement activation functions.

    Args:
        ActivationFunctions (class): inherited class.
    """
    def __init__(self, function_type = "relu"):
        """Initialize current class.

        Args:
            function_type (str, optional): chosen activation function. 
            Support following functions:
                - "relu": rectified linear unit function.
                - "sigmoid": sigmoid function.
                - "tanh": tanh function.
                - "softmax": softmax function.
                Defaults to "relu".
        """
        activation_functions_dict = {"sigmoid": Sigmoid, "relu": Relu, 
                                "tanh": Tanh, "softmax": SoftMax}
        assert function_type in activation_functions_dict, \
            "Invalid function. {} is not available".format(function_type)
        self.activation_functions = activation_functions_dict[function_type]()

    def __call__(self, x):
        """Compute the activation function.

        Args:
            x (numpy array): an array containing values 
            to pass through the activation function.

        Returns:
            (numpy array): an array containing computed values.
        """
        return self.activation_functions(x)
    
    def grad(self, x):
        """Compute the activation function.

        Args:
            x (numpy array): an array containing values 
            to pass through the activation function.

        Returns:
            (numpy array): an array containing computed values.
        """
        return self.activation_functions.grad(x)