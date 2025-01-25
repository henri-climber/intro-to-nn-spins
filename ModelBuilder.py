import tensorflow as tf
from tensorflow.keras import layers, models


class ModelBuilder:
    """
    Class to build Convolutional Neural Network models for spin classification.
    Allows easy customization of model architecture.
    """

    def __init__(self, input_shape=(10, 10, 1), num_classes=2, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run=None):
        """
        Initialize ModelBuilder with input shape, number of classes, and training configurations.

        Args:
            input_shape (tuple): Shape of the input data (height, width, channels).
            num_classes (int): Number of output classes for classification.
            optimizer (str or tf.keras.optimizers.Optimizer): Optimizer for model compilation.
            loss (str or callable): Loss function for model compilation.
            metrics (list): List of metrics for model evaluation.
            run (neptune.Run): Neptune run object for logging (optional).
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.run = run  # Neptune Run object

    def build_simple_cnn(self):
        """
        Build a simple CNN model.

        Returns:
            tf.keras.Model: Compiled CNN model.
        """
        model = models.Sequential(name="SimpleCNN")

        # Use an Input layer explicitly
        model.add(layers.Input(shape=self.input_shape))

        # Add Convolutional layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2), padding='same'))

        # Flatten and add Dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))  # Dropout for regularization
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=self.metrics)

        # Log model architecture and hyperparameters to Neptune
        if self.run:
            self.run["model/summary"] = model.to_json()
            self.run["model/hyperparameters"] = {
                "input_shape": str(self.input_shape),  # Convert tuple to string
                "num_classes": self.num_classes,
                "optimizer": str(self.optimizer),  # Convert optimizer to string
                "loss": self.loss,
                "metrics": str(self.metrics)  # Convert list to string
            }

        return model

    def build_custom_cnn(self, conv_layers, dense_layers):
        """
        Build a customizable CNN model.

        Args:
            conv_layers (list of dict): List of convolutional layer configurations.
                Each dict should have keys: 'filters', 'kernel_size', 'activation', and 'pooling'.
                Example: [{'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu', 'pooling': (2, 2)}]
            dense_layers (list of tuple(int,float)): List of dense layer units (int) and dropout rates (float).
                Example: [128, 64]
            dropout_rate (float): Dropout rate for regularization.

        Returns:
            tf.keras.Model: Compiled CNN model.
        """
        model = models.Sequential(name="CustomCNN")
        # Use an Input layer explicitly
        model.add(layers.Input(shape=self.input_shape))

        for layer_config in conv_layers:
            model.add(layers.Conv2D(layer_config['filters'],
                                    layer_config['kernel_size'],
                                    activation=layer_config['activation']))
            if layer_config['pooling']:
                model.add(layers.MaxPooling2D(layer_config['pooling']))

        # Flatten and add Dense layers
        model.add(layers.Flatten())

        for units, dropout_rate in dense_layers:
            model.add(layers.Dense(units, activation='relu'))
            if type(dropout_rate) is float:
                model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=self.metrics)

        # Log model architecture and hyperparameters to Neptune
        if self.run:
            self.run["model/summary"] = model.to_json()
            self.run["model/hyperparameters"] = {
                "input_shape": self.input_shape,
                "num_classes": self.num_classes,
                "conv_layers": conv_layers,
                "dense_layers": dense_layers,
                "dropout_rate": dropout_rate,
                "optimizer": str(self.optimizer),
                "loss": self.loss,
                "metrics": self.metrics
            }

        return model
