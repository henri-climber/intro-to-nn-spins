import copy

import tensorflow as tf


class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, epochs=20, callbacks=None, metrics=[], run=None):
        """
        Initialize the Trainer with datasets and training configuration.

        Args:
            model (tf.keras.Model): The model to train.
            train_dataset (tf.data.Dataset): Training dataset.
            val_dataset (tf.data.Dataset): Validation dataset.
            test_dataset (tf.data.Dataset): Testing dataset.
            epochs (int): Number of training epochs.
            callbacks (list): List of Keras callbacks.
            run (neptune.Run): Neptune run object for logging (optional).
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.callbacks = callbacks if callbacks else []
        self.metrics = metrics
        self.run = run  # Neptune Run object

    def train(self):
        """
        Train the model using the provided datasets.
        Logs metrics to Neptune if the run object is provided.

        Returns:
            tf.keras.callbacks.History: Training history object.
        """
        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.epochs,
            callbacks=self.callbacks
        )

        # Log metrics to Neptune after training
        if self.run:
            completed_epochs = len(history.history['loss'])
            for epoch in range(completed_epochs):
                self.run["train/loss"].append(history.history['loss'][epoch])
                self.run["train/accuracy"].append(history.history['accuracy'][epoch])
                self.run["val/loss"].append(history.history['val_loss'][epoch])
                self.run["val/accuracy"].append(history.history['val_accuracy'][epoch])

        return history

    def evaluate(self):
        """
        Evaluate the model on the test dataset.
        Logs test metrics to Neptune if the run object is provided.

        Returns:
            dict: Evaluation metrics.
        """
        m = copy.deepcopy(self.metrics)
        m.insert(0, "loss")  # Add loss as the first metric
        results = self.model.evaluate(self.test_dataset)
        metrics = {metric: value for metric, value in zip(m, results)}

        # Log test metrics to Neptune
        if self.run:
            for metric, value in metrics.items():
                self.run[f"test/{metric}"] = value

        return metrics
