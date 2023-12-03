# Covid19-Detection
This is my first year project. The primary goal of this project is to develop a deep neural network model that can identify disease from chest X-ray pictures. The model will specifically learn all chest X-ray pictures gathered from COVID-19 patients as well as healthy people. Then, the model expects to receive chest X- ray images from the user and attempts to categorize whether the patient is positive or negative for COVID-19.

While using the program, you have the options to use a pre-trained model or customize your own model. For the customization, you can choose your preferred initialization method, activation function, optimization method, hidden layers, number of iterations and hyper-parameter for regularization technique. Then, you will input chest X-ray image and the program will run diagnosis to classify whether it is a COVID-19 positive or negative case. As a result, the program will print a summary of all diagnosed cases.

## Prerequisites

In order to use this program, several Python modules and packages are used as supplementary materials to build the whole program. All these modules and packages must be installed to run the program without errors. Here is a list of some modules and packages used:

| Modules & Packages | Version Requirement |
| -------------------| --------------------|
| numpy              | 1.26.2              |
| matplotlib         | 3.6.3               |
| opencv_python      | 4.7.0.72            |
| pandas             | 2.1.3               |
| joblib             | 1.2.0               |

These packages can be install by typing following code in terminal:

```
pip install requirements
```
## Program structure

Here is the structure of our program:

```
COVID19-Detection
├── dataset: containing raw dataset
│   ├── COVID
│   └── Normal
├── model: source code for building neural network
│   ├── activation_functions
│   │   ├── __init__.py
│   │   └── ActivationFunctions.py
│   ├── initializer
│   │   ├── __init__.py
│   │   └── initializer.py
│   ├── optimizer
│   │   ├── __init__.py
│   │   └── Optimization.py
│   ├── __init__.py
│   └── MultiLayerPerceptron.py
├── new_prediction: containing images for diagnosis
├── preprocessing: pre-process raw dataset
│   ├── __init__.py
│   ├── DataProcessing.py
│   └── processed_dataset.pkl
├── trained_model: save location for trained model
│   └── pre_trained.pkl
└── main.py: source code to run our program
```
In order to use our application, just executing the following code in terminal:

```
python3 main.py
```

When sucessfully starting the program, the option menu is display. It will ask users to choose between using a pre-trained model or customize a new model to begin the diagnosis. The output should look like this:
```
Starting the program. Please wait for a moment...
Welcome to COVID-19 Diagnosis program.            
Please choose the folowing options: 
   1. Using a pre-trained model
   2. Customize your model
-----------------------------------------------------
>> Enter your choice (1 or 2): 
```



