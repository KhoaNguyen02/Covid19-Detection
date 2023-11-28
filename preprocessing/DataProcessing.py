import os
import numpy as np
import pandas as pd
import cv2
import pickle

class DataPreProcess:
    """A class to prepare the data for the model."""
    def __init__(self, path = "Radiography dataset", \
                split_probability = 0.8):
        """Setup the class with necessary information.

        Args:
            path (str, optional): location of original data. 
                Defaults to "COVID-19_Radiography_Dataset".
            split_probability (float, optional): probability to split data 
                into training set and test set. 
                For example, if split_probability = 0.6,
                this means that the data will be split into 
                training set with 60% size of the original data, 
                and test set with 40% size of original data. 
                Defaults to 0.8.
        """
        self.data_location = path
        self.split_prob = split_probability

    def read_dataset(self):
        """Read images from dataset.

        Returns:
            (tuple): A tuple with first element is a list of images, 
            and second element is a list of corresponding label 
            (1 if the image shows a COVID positive case, 0 otherwise).
        """
        X, y = [], []
        try:
            covid_data_dir = self.data_location + "/COVID"
            normal_data_dir = self.data_location + "/Normal"
        except FileNotFoundError:
            print("There are no {} available".format(self.data_location))

        # Read images labelled with COVID.
        for file, _ in zip(os.listdir(covid_data_dir), range(3000)):
            image = cv2.imread("{}/{}".format(covid_data_dir, file))
            X.append(self.process_image(image))
            y.append(1.0)

        # Read images labelled with Normal.
        for file, _ in zip(os.listdir(normal_data_dir), range(3000)):
            image = cv2.imread("{}/{}".format(normal_data_dir, file))
            X.append(self.process_image(image))
            y.append(0.0)

        return X, y

    def process_image(self, image):
        """Process an image read from file.
        Reference:
            https://realpython.com/image-processing-with-the-python-pillow-library/
    
        Args:
            image (class): a class represents for an image.

        Returns:
            (numpy array): a numpy array containing processed image.
        """
        # Convert image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Reduce noise
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Resize image
        image = cv2.resize(image, (128, 128))

        # Normalization
        image = (image - np.mean(image)) / np.std(image)

        return image

    def run(self):
        """Run the process of reading dataset and save processed data."""
        # Read and process data.
        print("Beginning reading images...")
        X, y = self.read_dataset()
        X = np.array(X)
        y = np.array(y)
        print("Finished reading images.")

        # Save processed dataset into a file.
        print("Saving processed dataset...")
        pickle.dump((X, y), open("Preprocessing/processed_dataset.pkl", "wb"))
        print("Finished saving processed dataset.")

    def shuffle_data(self, X, y):
        """Randomly shuffle arrays of data.

        Args:
            X (numpy array): an array of processed images.
            y (numpy array): an array of corresponding labels. 

        Returns:
            (tuple): A tuple with first element is an array 
            of shuffled processed images, and second element 
            is an array of shuffled labels.
        """
        shuffle_index = np.arange(X.shape[0])
        np.random.shuffle(shuffle_index)
        X_shuffled = X[shuffle_index]
        y_shuffled = y[shuffle_index]
        return X_shuffled, y_shuffled

    def train_test_split(self, X, y):
        """Splitting the data into training set and test set.

        Args:
            X (numpy array): an array of processed images.
            y (numpy array): an array of corresponding labels. 

        Returns:
            (tuple): A tuple containing training set and test set.
        """
        split_index = (int) (X.shape[0] * self.split_prob)
        X_train, y_train = X[:split_index], y[:split_index]
        X_test, y_test = X[split_index:], y[split_index:]
        return X_train, y_train, X_test, y_test

    def reshape_data(self, X_train, y_train, X_test, y_test):
        """Reshape processed data to feed into the model.

        Args:
            X_train (numpy array): processed images for training.
            y_train (numpy array): corresponding labels for training.
            X_test (numpy array): processed images for testing.
            y_test (numpy array): corresponding labels for testing.

        Returns:
            (tuple): A tuple containing reshaped data 
            for training and testing.
        """
        X_train = X_train.reshape(X_train.shape[0], -1).T
        X_test = X_test.reshape(X_test.shape[0], -1).T
        y_train = y_train.reshape(-1, 1).T
        y_test = y_test.reshape(-1,1).T
        return X_train, y_train, X_test, y_test

    def load_data(self):
        """Load processed data to prepare for the learning process.

        Returns:
            (tuple): A tuple containing loaded data 
            for training and testing.
        """
        # Load processed dataset.
        X, y = pickle.load(open("Preprocessing/processed_dataset.pkl", "rb"))

        # Split dataset into training and test set.
        X_shuffled, y_shuffled = self.shuffle_data(X, y)
        X_train, y_train, X_test, y_test = \
            self.train_test_split(X_shuffled, y_shuffled)

        # Reshape data to feed into the model.
        X_train, y_train, X_test, y_test = \
            self.reshape_data(X_train, y_train, X_test, y_test)
        return X_train, y_train, X_test, y_test
    
    def read_new_image(self):
        """Read new images from the user to make a diagnosis.

        Returns:
            (numpy array): read images.
        """
        X = []
        for file in os.listdir(self.data_location):
            image = cv2.imread("{}/{}".format(self.data_location, file))
            X.append(self.process_image(image))
        X = np.array(X)
        return X.reshape(X.shape[0], -1).T
    
if __name__ == "__main__":
    data_preprocess = DataPreProcess()
    data_preprocess.run()