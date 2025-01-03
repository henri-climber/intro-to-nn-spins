import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

from loadSnapshots import loading


class DataLoader:
    """Dataset class for loading and processing snapshot data using TensorFlow."""

    def __init__(self, cases, doping, max_shots, target_size=(10, 10), train=True, train_split=0.8):
        """
        Initialize the dataset.
        
        Args:
            cases (list): List of cases to load ('hf', 'pi', etc.)
            doping (float): Doping value
            max_shots (int): Maximum number of shots per case
            target_size (tuple): Target size for the images (height, width)
            train (bool): Whether to return training or test set
            train_split (float): Fraction of data to use for training
        """
        self.target_size = target_size

        # Load the raw data
        samples, labels, self.sms = loading(cases, doping, max_shots)

        # Convert to numpy arrays
        self.samples = np.array(samples, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.float32)

        # Get dimensions
        self.num_samples = len(self.samples)
        self.input_size = self.samples[0].shape[0]
        self.num_classes = self.labels[0].shape[0]

        # Create train/test split indices
        indices = list(range(self.num_samples))
        random.shuffle(indices)
        split = int(train_split * self.num_samples)

        # Select appropriate indices based on train/test
        if train:
            self.indices = indices[:split]
        else:
            self.indices = indices[split:]

        # First reshape to square
        orig_side_length = int(np.sqrt(self.input_size))
        self.samples = self.samples.reshape(-1, orig_side_length, orig_side_length, 1)

        # Resize to target size
        self.samples = self._resize_samples(self.samples)

    def _resize_samples(self, samples):
        """
        Resize samples to target size using TensorFlow's resize operation.
        
        Args:
            samples (np.ndarray): Array of samples to resize
            
        Returns:
            np.ndarray: Resized samples
        """
        # Convert to TensorFlow tensor
        samples_tf = tf.convert_to_tensor(samples)

        # Resize using bilinear interpolation
        resized = tf.image.resize(
            samples_tf,
            self.target_size,
            method=tf.image.ResizeMethod.BILINEAR
        )

        return resized.numpy()

    def get_tf_dataset(self, batch_size=32, shuffle=True):
        """
        Create a TensorFlow Dataset.
        
        Args:
            batch_size (int): Size of each batch
            shuffle (bool): Whether to shuffle the data
            
        Returns:
            tf.data.Dataset: TensorFlow dataset
        """
        # Select the appropriate data using indices
        x = tf.gather(self.samples, self.indices)
        y = tf.gather(self.labels, self.indices)

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.indices))

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def visualize_sample(self, index, show_grid=True):
        """
        Visualize a single sample.
        
        Args:
            index (int): Index of the sample to visualize
            show_grid (bool): Whether to show grid lines
        """
        if index >= len(self.indices):
            raise ValueError(f"Index {index} is out of bounds for dataset of size {len(self.indices)}")

        actual_idx = self.indices[index]
        sample = self.samples[actual_idx]
        label = self.labels[actual_idx]

        plt.figure(figsize=(8, 8))
        plt.imshow(sample[:, :, 0], cmap=plt.cm.coolwarm, interpolation='nearest')

        if show_grid:
            plt.grid(True, which='both', color='black', linewidth=0.5, alpha=0.3)

        plt.title(f"Sample {index}\nClass: {np.argmax(label)}\nSize: {sample.shape[:2]}")
        plt.colorbar()
        plt.show()

    def visualize_batch(self, num_samples=4, rows=2):
        """
        Visualize multiple samples in a grid.
        
        Args:
            num_samples (int): Number of samples to visualize
            rows (int): Number of rows in the visualization grid
        """
        cols = (num_samples + rows - 1) // rows

        plt.figure(figsize=(4 * cols, 4 * rows))

        for i in range(min(num_samples, len(self.indices))):
            actual_idx = self.indices[i]
            sample = self.samples[actual_idx]
            label = self.labels[actual_idx]

            plt.subplot(rows, cols, i + 1)
            plt.imshow(sample[:, :, 0], cmap=plt.cm.coolwarm, interpolation='nearest')
            plt.grid(True, which='both', color='black', linewidth=0.5, alpha=0.3)
            plt.title(f"Class: {np.argmax(label)}\nSize: {sample.shape[:2]}")
            plt.colorbar()

        plt.tight_layout()
        plt.show()

    def visualize_class_distribution(self):
        """
        Visualize the class distribution in the dataset.
        """
        counts = np.zeros(self.num_classes)

        for i in self.indices:
            label = self.labels[i]
            counts[np.argmax(label)] += 1

        plt.figure(figsize=(8, 8))
        plt.bar(range(self.num_classes), counts)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("Class Distribution")
        plt.show()

def get_data_loaders(cases, doping, max_shots, target_size=(10, 10), batch_size=32, train_split=0.8):
    """
    Create TensorFlow datasets for training and testing.
    
    Args:
        cases (list): List of cases to load
        doping (float): Doping value
        max_shots (int): Maximum number of shots per case
        target_size (tuple): Target size for the images (height, width)
        batch_size (int): Size of each batch
        train_split (float): Fraction of data to use for training
        
    Returns:
        tuple: (train_dataset, test_dataset, train_data_obj, test_data_obj)
    """
    # Create training dataset
    train_data = DataLoader(
        cases=cases,
        doping=doping,
        max_shots=max_shots,
        target_size=target_size,
        train=True,
        train_split=train_split
    )

    # Create test dataset
    test_data = DataLoader(
        cases=cases,
        doping=doping,
        max_shots=max_shots,
        target_size=target_size,
        train=False,
        train_split=train_split
    )

    # Create TensorFlow datasets
    train_dataset = train_data.get_tf_dataset(batch_size=batch_size)
    test_dataset = test_data.get_tf_dataset(batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_data, test_data


if __name__ == "__main__":
    train_dataset, test_dataset, train_data, test_data = get_data_loaders(
        cases=["AS", "exp", "pi"],
        doping=6.0,
        max_shots=1000,
        train_split=0.8)
    # print out some shapes and sizes with according titles
    print("samples shape: ", train_data.samples.shape)
    print("labels shape: ", train_data.labels.shape)
    print("num samples: ", train_data.num_samples)
    print("input size: ", train_data.input_size)
    print("num classes", train_data.num_classes)
    print("target size: ", train_data.target_size)

    # Visualize some samples
    train_data.visualize_sample(0)  # Show first sample
    train_data.visualize_batch(num_samples=8)  # Show grid of 8 samples

    # Visualize class distribution
    train_data.visualize_class_distribution()
    test_data.visualize_class_distribution()
