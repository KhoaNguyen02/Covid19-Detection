import numpy as np

class InitializeParams:
    """An interface class to initialize parameters for the model."""
    def __init__(self, layers):
        """Initialize the interface."""
        self.layers = layers
        self.weights = []
        self.bias = []
        pass
      
    def set_params(self):
        """A method to set parameters for the model."""
        pass

    def get_params(self):
        """A method to get parameters of the model."""
        pass


class ZeroInitialize(InitializeParams):
    """A class to initialize parameters with zeros 
    inherited from InitializeParams interface class.

    Args:
        InitalizeParams (class): class to be inherited.
    """
    def set_params(self):
        """Implement method to initialize parameters."""
        # Initialize weights and bias.
        for i in range(len(self.layers) - 1):
            self.weights.append(np.zeros((self.layers[i + 1], self.layers[i])))
            self.bias.append(np.zeros((self.layers[i + 1], 1)))
        
        # Convert weights and bias into numpy array.
        self.weights = np.array(self.weights, dtype = object)
        self.bias = np.array(self.bias, dtype = object)
    
    def get_params(self):
        """Implement method to get parameters of the model.

        Returns:
            (tuple): weights and bias of the model.
        """
        return self.weights, self.bias

class RandomInitialize(InitializeParams):
    """A class to initialize parameters randomly 
    inherited from InitializeParams interface class.

    Args:
        InitalizeParams (class): class to be inherited.
    """
    def set_params(self):
        """Implement method to initialize parameters."""
        # Initialize weights and bias.
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i + 1], self.layers[i]) * 0.01)
            self.bias.append(np.zeros((self.layers[i + 1], 1)))

        # Convert weights and bias into numpy array.
        self.weights = np.array(self.weights, dtype = object)
        self.bias = np.array(self.bias, dtype = object)
    
    def get_params(self):
        """Implement method to get parameters of the model.

        Returns:
            (tuple): weights and bias of the model.
        """
        return self.weights, self.bias

class XavierInitialize(InitializeParams):
    """A class to initialize parameters with Xavier Glorot method 
    inherited from InitializeParams interface class.

    Args:
        InitializeParams (class): class to be inherited.
    """
    def set_params(self):
        """Implement method to initialize parameters."""
        # Initialize weights and bias.
        for i in range(len(self.layers) - 1):
            upper_bound = np.sqrt(2/(self.layers[i+1] + self.layers[i]))
            self.weights.append(np.random.randn(self.layers[i + 1], self.layers[i]) * upper_bound)
            self.bias.append(np.zeros((self.layers[i + 1], 1)))
        
        # Convert weights and bias into numpy array.
        self.weights = np.array(self.weights, dtype = object)
        self.bias = np.array(self.bias, dtype = object)
    
    def get_params(self):
        """Implement method to get parameters of the model.

        Returns:
            (tuple): weights and bias of the model.
        """
        return self.weights, self.bias

class Initialize(InitializeParams):
    """A main class to initialize parameters for the model.

    Args:
        InitializeParams (class): interface to be implemented.
    """
    def __init__(self, layers, initialize_method = "xavier"):
        """Setup the class with necessary information.

        Args:
            layers (list): number of nodes per layer.
            initialize_method (str, optional): initialization method. 
                    Support 'zero', 'random' and 'xavier'. Defaults to "xavier".
        """
        initialize_dict = {"zero": ZeroInitialize, "random": RandomInitialize, 
                        "xavier": XavierInitialize}
        assert initialize_method in initialize_dict, \
        "Invalid function. {} is not available".format(initialize_method)
        self.initialize = initialize_dict[initialize_method](layers)
        self.set_params()

    def set_params(self):
        """Initialize parameters for the model."""
        self.initialize.set_params()

    def get_params(self):
        """Get parameters of the model.

        Returns:
            (tuple): weights and bias of the model.
        """
        return self.initialize.get_params()
