# Spins Projects

# TODO

1. Create a running version that:
    - Load data using the `load_data` function
    - prepare data for model and maybe load it into tf.datasets
    - create a simple cnn model for classification of 4 classes
    - train the model
    - evaluate the model
    - save the model
    - visualize the results
    - (Depending on training time try to train the model on CIP-Pool)

2. Make everything **modular**
    - Create a **DataLoader** class for loading and preparing the dataset for training based on different parameters
    - Create a **Model** class for creating the model based on different parameters and add method to save model
    - Create a **Evaluation** Class for visualizing the results and storing the results and any statistics in local files
        - Evaluate each model and store all parameters and model results in an Excel or sth
        - Visualize the results of the models
    - Create a **config** file for configuring hyperparameters and create methods in the classes that take these hyperparameters as parameters
        - Suggest to use **omegaconf** for this: https://omegaconf.readthedocs.io/en/2.3_branch/index.html
        - Look into Keras Tuner for hyperparameter tuning and GridSearchCV or RandomizedSearchCV
     
     
    
