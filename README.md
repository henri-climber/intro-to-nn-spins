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
    - Create a file/class for creating the model based on different parameters
    - Create a **Visualizer** Class for visualizing the results
   
    - Create a file for training the model based on a larger selection of hyperparameters
      - Look into Keras Tuner for hyperparameter tuning and GridSearchCV or RandomizedSearchCV
      - Evaluate each model and store all parameters and model results in an Excel or sth
      - Visualize the results of the models
