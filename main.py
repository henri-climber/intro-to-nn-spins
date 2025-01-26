import os
import tensorflow as tf
import neptune
from dotenv import load_dotenv
from itertools import product
from ModelBuilder import ModelBuilder
from ModelTrainer import ModelTrainer
from DataLoader import get_data_loaders

tf.compat.v1.enable_eager_execution()

# Load API key from .env file
load_dotenv()
NEPTUNE_API_KEY = os.getenv("NEPTUNE_API")

if not NEPTUNE_API_KEY:
    raise ValueError("Neptune API key not found. Please set NEPTUNE_API in the .env file.")

classes = ["exp", "AS"]

# Example datasets (Replace with actual tf.data.Dataset)
train_data, val_data, test_data, data_loader = get_data_loaders(
    cases=classes,
    doping=6.0,
    max_shots=1500,
    batch_size=200,
    train_split=0.8)

# Define lists for each parameter
input_shapes = [(10, 10, 1)]
num_classes_list = [len(classes)]
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.96
)
learning_rates = [
    0.001, 0.0001, 0.01
]
losses = ["categorical_crossentropy"]
metrics_list = [[tf.keras.metrics.CategoricalAccuracy(name="accuracy"), tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]]
callbacks_list = [[tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss to detect overfitting
    min_delta=0.001,  # Minimum improvement to reset patience
    patience=100,  # Number of epochs with no improvement before stopping
    verbose=1,  # Provide verbose output for better monitoring
    mode='min',  # Stop when the monitored value stops decreasing
    restore_best_weights=False
)], None]

# Generate all combinations of parameters
for input_shape, num_classes, learning_rate, loss, metrics, callbacks in product(
        input_shapes, num_classes_list, learning_rates, losses, metrics_list, callbacks_list):
    # TODO: try different model architectures
    conv_layer_options = [
        # Single-layer convolution for simplicity
        [
            {"filters": 20, "kernel_size": (3, 3), "activation": "relu", "pooling": (2, 2)}
        ], [
            {"filters": 20, "kernel_size": (3, 3), "activation": "relu", "pooling": None}
        ],
        [
            {"filters": 24, "kernel_size": (4, 4), "activation": "relu", "pooling": (2, 2)}
        ],
        # Two-layer convolution with small kernels
        [
            {"filters": 16, "kernel_size": (3, 3), "activation": "relu", "pooling": (2, 2)},
            {"filters": 4, "kernel_size": (2, 2), "activation": "relu", "pooling": None}
        ],
        # Slightly deeper network for more complex relationships
        [
            {"filters": 32, "kernel_size": (3, 3), "activation": "relu", "pooling": (2, 2)},
            {"filters": 64, "kernel_size": (2, 2), "activation": "relu", "pooling": (2, 2)}
        ]
    ]

    dense_layer_options = [
        # Simple fully connected layer with minimal regularization
        [(32, 0.2)],
        # Two dense layers with moderate dropout
        [(64, 0.3), (32, 0.1)],
        # Deeper network with stronger regularization
        [(128, 0.7), (64, 0.3), (32, 0.5)],
        # Very minimal dense layers for simplicity
        [(16, 0.1)]
    ]

    # Build model
    for conv_layers, dense_layers in product(conv_layer_options, dense_layer_options):
        # Initialize Neptune run
        run = neptune.init_run(
            project="research-nitrat/SpinsTUM",
            api_token=NEPTUNE_API_KEY,
            monitoring_namespace="monitoring",
            capture_hardware_metrics=True
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        run["sys/tags"].add([
            f"input_shape_{input_shape}",
            f"num_classes_{num_classes}",
            f"optimizer_{optimizer}",
            f"loss_{loss}",
            f"metrics_{metrics}"
        ])

        # Build model using ModelBuilder
        builder = ModelBuilder(
            input_shape=input_shape,
            num_classes=num_classes,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            run=run
        )

        model = builder.build_custom_cnn(conv_layers, dense_layers)

        model.summary()

        # Train and evaluate model using Trainer
        model_trainer = ModelTrainer(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            test_dataset=test_data,
            callbacks=callbacks,
            metrics=metrics,
            epochs=1000,
            run=run
        )

        history = model_trainer.train()
        results = model_trainer.evaluate()

        # wait for the run to be stopped
        run.wait()

        # Stop the Neptune run
        run.stop()
