import os
import joblib
from preprocessing import *
from model import *

class CovidDetection:
    """A class to run the Covid Diagnosis program."""

    def __init__(self):
        """Setup the class."""
        print("Starting the program. Please wait for a moment...")
        self.X_train, self.y_train, self.X_test, \
            self.y_test = DataPreProcess().load_data()
        self.customize = False

    def option(self):
        """Let the user to choose whether using a 
        pre-trained model or custom a new model.

        Returns:
            (class): A class represents for the model.
        """
        # Display the option menu.
        print("Welcome to COVID-19 Diagnosis program.\
            \nPlease choose the folowing options: ")
        print(" "*3 + "1. Using a pre-trained model")
        print(" "*3 + "2. Customize your model")

        # Let the user to choose their preferred option.
        print("--" * 35)
        while (True):
            user_input = input(">> Enter your choice (1 or 2): ")
            try:
                user_input = int(user_input)
                assert 1 <= user_input <= 2
                if user_input == 1:
                    return self.pre_trained_model()
                else:
                    self.customize = True
                    return self.customize_model()
            except (AssertionError, ValueError):
                print("Invalid choice. Please try again.")

    def waiting_for_input(self, dict_to_check):
        """A common method to wait for user input 
        to choose initialization method,
        activation function and optimization method.

        Args:
            dict_to_check (dict): a dictionary containing 
            available options.

        Returns:
            (str): input from the user.
        """
        while (True):
            user_input = input(">> Enter your choice: ")
            try:
                assert user_input in dict_to_check or user_input == ""
                if user_input == "":
                    print("Set to " + dict_to_check[-1] + " by default.")
                    return dict_to_check[-1]
                return user_input
            except AssertionError:
                print("Invalid choice. Please try again.")

    def choose_initialize_method(self):
        """Ask the user to choose initialization method.

        Returns:
            (str): user choice.
        """
        print("1. Initialization methods: ")
        print(" " * 2 + "Support 'zero', 'random', and 'xavier' only." +
              " Press Enter to skip.")
        initialization_methods = ["zero", "random", "xavier"]
        return self.waiting_for_input(initialization_methods)

    def choose_activation_function(self, predict=False):
        """Ask the user to choose activation function.

        Args:
            predict (bool, optional): whether to choose activation function 
                    for training process or predicting process. 
                    Training process if True, predicting process otherwise. 
                    Defaults to False.
        Returns:
            (str): user choice.
        """
        if predict:
            print(">> Choose activation functions to make diagnosis: ")
        else:
            print("2. Activation functions: ")
        print(" " * 2 + "Support 'sigmoid', 'softmax', 'tanh', 'relu' only." +
              " Press Enter to skip.")
        activation_functions = ["sigmoid", "softmax", "tanh", "relu"]
        return self.waiting_for_input(activation_functions)

    def choose_optimizize_method(self):
        """Ask the user to choose optimization method.

        Returns:
            (str): user choice.
        """
        print("3. Optimization methods: ")
        print(" " * 2 + "Support 'sgd', 'momentum', and 'rms_prop' only." +
              " Press Enter to skip.")
        optimization_methods = ["sgd", "momentum", "rms_prop"]
        return self.waiting_for_input(optimization_methods)

    def choose_hidden_layers(self):
        """Ask the user to choose number of hidden layers and number of nodes
        each layer.

        Returns:
            (list): a list containing number of nodes per layer.
        """
        print("4. Hidden layers: ")
        nodes = [self.X_train.shape[0]]

        # Ask the user for number of hidden layers.
        while (True):
            n_layers = input(">> Enter number of hidden layers: ")
            try:
                n_layers = int(n_layers)
                assert n_layers > 0
                break
            except ValueError:
                print("Invalid input. Please try again.")
            except AssertionError:
                print("Number of hidden layers must be larger than 0")

        # Ask the user for number of nodes each layer.
        for i in range(n_layers):
            while (True):
                n_nodes = input(">> Enter number of nodes in hidden layer {}: "
                                .format(i+1))
                try:
                    n_nodes = int(n_nodes)
                    assert n_nodes > 1
                    nodes.append(n_nodes)
                    break
                except ValueError:
                    print("Invalid input. Please try again.")
                except AssertionError:
                    print("Number of nodes must be larger than 1")
        nodes.append(1)
        return nodes

    def choose_n_iterations(self):
        """Ask the user to choose number of iterations for the model.

        Returns:
            (int): number of iterations.
        """
        print("5. Number of iterations: ")
        while (True):
            n_iters = input(">> Enter number of iterations." +
                            " Press Enter to skip: ")
            try:
                if n_iters != "":
                    n_iters = int(n_iters)
                else:
                    print("Set to " + str(10000) + " by default.")
                    return 10000
                return n_iters
            except ValueError:
                print("Invalid input. Please try again.")

    def choose_learning_rate(self):
        """Ask the user to choose learning rate for the model.

        Returns:
            (float): learning rate.
        """
        print("6. Learning rate: ")
        while (True):
            alpha = input(">> Enter learning rate." +
                          " Press Enter to skip: ")
            try:
                if alpha != "":
                    alpha = float(alpha)
                else:
                    print("Set to " + str(0.0001) + " by default.")
                    return 0.0001
                return alpha
            except ValueError:
                print("Invalid input. Please try again.")

    def choose_l2_params(self):
        """Ask the user to choose hyperparameter 
        for L2 regularization.

        Returns:
            (float): hyperparameter for L2.
        """
        print("7. L2 regularization parameter: ")
        while (True):
            alpha = input(">> Enter L2 regularization parameter (recommended from 0.0 to 1.0)." +
                          " Press Enter to skip: ")
            try:
                if alpha != "":
                    alpha = float(alpha)
                else:
                    print("Set to " + str(0.7) + " by default.")
                    return 0.7
                return alpha
            except ValueError:
                print("Invalid input. Please try again.")

    def print_summary(self, model):
        """Print the summary of current model.

        Args:
            model (class): a class represents for the model.
        """
        print("--" * 35 + "\nHere is the information about the model: ")
        model.summary()

    def customize_model(self):
        """Ask the user to customize their preferred model.

        Returns:
            (class): customized model.
        """
        print("Let's get started to create your preferred model")
        initialize = self.choose_initialize_method()
        function = self.choose_activation_function()
        optim = self.choose_optimizize_method()
        layers = self.choose_hidden_layers()
        n_iters = self.choose_n_iterations()
        learning_rate = self.choose_learning_rate()
        lambd = self.choose_l2_params()
        model = MLP(layers, n_iters, alpha=learning_rate, lambd=lambd,
                    initialize_method=initialize, function_type=function,
                    optim_method=optim)
        self.print_summary(model)
        return model

    def pre_trained_model(self):
        """Load the pre-trained model.

        Returns:
            (class): pre-trained model.
        """
        print("Loading pre trained model...")
        model = joblib.load("trained_model/pre_trained.pkl")
        return model

    def print_accuracy(self, model):
        """Display accuracy of the model.

        Args:
            model (class): a class represents for the model.
        """
        # Wait for user input.
        print("--" * 35)
        while (True):
            user_input = input(">> Do you want to print" +
                               " accuracy of the model (yes/no)? ")
            try:
                assert user_input == "yes" or user_input == "no"
                break
            except AssertionError:
                print("Invalid input. Please try again.")

        # Process user input.
        if user_input == "yes":
            accuracy_train = model.evaluate(self.X_train, self.y_train)
            accuracy_test = model.evaluate(self.X_test, self.y_test)
            print("Accuracy on training set: {:.2f}%".format(
                accuracy_train * 100))
            print("Accuracy on test set: {:.2f}%".format(accuracy_test * 100))

    def plot_result(self, model):
        """Plot cost and accuracy of the model as well as confusion matrix.

        Args:
            model (class): a class represents for the model.
        """
        # Wait for user input.
        while (True):
            user_input = input(">> Do you want to plot loss and" +
                               " accuracy of the model (yes/no)? ")
            try:
                assert user_input == "yes" or user_input == "no"
                break
            except AssertionError:
                print("Invalid input. Please try again.")

        # Process user input.
        if user_input == "yes" and model.n_iterations > 10:
            y_pred = model.predict(self.X_test)
            model.print_confusion_matrix(y_pred, self.y_test)
            model.plot()
        elif model.n_iterations <= 10:
            y_pred = model.predict(self.X_test)
            model.print_confusion_matrix(y_pred, self.y_test)
            print("Cannot plot the loss curve because number of iterations is too small.")

    def predicted_summary(self, new_observations, model):
        """Display a summary of predicted results based on new images.

        Args:
            new_observations (numpy array): new images to make a diagnosis.
            model (class): model used to make diagnosis.
        """
        # Make predictions.
        print("\nHere is the prediction from the model: ")
        predicted_labels = model.predict(new_observations)

        # Display predictions.
        for label, file in zip(predicted_labels[0, :], os.listdir("Covid_Predict")):
            diagnosis = "Positive" if label == 1 else "Negative"
            print(" " * 3 + "{} : {}".format("Covid_Predict/" + file, diagnosis))

    def continue_predict(self):
        """Ask the user whether to continue diagnosing process.

        Returns:
            (bool): True if continue, False otherwise.
        """
        while (True):
            try:
                user_input = input("\n>> Do you want to continue (yes/no):  ")
                assert user_input == "yes" or user_input == "no"
                if user_input == "no":
                    return False
                return True
            except AssertionError:
                print("Invalid input. Please try again.")

    def diagnose(self, model):
        """Let the model to make a diagnosis for new COVID cases.

        Args:
            model (class): model used to make diagnosis.
        """
        while (True):
            print("--" * 35 + "\n>> Please put new images into 'new_prediction'" +
                  " folder to make diagnosis.")
            user_input = input("Press any key to continue or 'quit' to quit: ")
            if user_input == "quit":
                print("Thanks for using the program. Bye!")
                break
            self.choose_activation_function(predict=True)
            try:
                # Read new images.
                if os.path.getsize("new_prediction") == 0:
                    raise ValueError
                data = DataPreProcess("new_prediction")
                new_observations = data.read_new_image()

                # Show a diagnosis summary.
                self.predicted_summary(new_observations, model)

                # Ask user to continue the prediction process.
                if not self.continue_predict():
                    print("Thanks for using the program. Bye!")
                    break

            except FileNotFoundError:
                print("Folder not found.")
            except ValueError:
                print("There are no images in the folder. Please try again.")
    
    def save_model(self, model):
        """Save the model to a file.

        Args:
            model (class): a class represents for the model.
        """
        while (True):
            user_input = input(">> Do you want to save the model (yes/no)? ")
            try:
                assert user_input == "yes" or user_input == "no"
                if user_input == "yes":
                    name = input(">> Enter the name of your model: ")
                    joblib.dump("trained_model/{name}.pkl".format(name=name))
                    print("Model saved successfully in trained_model folder.")
                break
            except AssertionError:
                print("Invalid input. Please try again.")

    def run(self):
        """A method to run the program."""
        # Show menu option.
        model = self.option()

        # Process selected option.
        if self.customize == True:
            print("--" * 35 + "\nStarting training the model....")
            model.fit(self.X_train, self.y_train)
        else:
            self.print_summary(model)

        # Show model accuracy.
        self.print_accuracy(model)

        # Plot model result
        self.plot_result(model)

        # Save the model.
        self.save_model(model)

        # Run diagnosing process.
        self.diagnose(model)


if __name__ == "__main__":
    program = CovidDetection()
    program.run()
