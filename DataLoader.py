import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from loadSnapshots import loading


class DataLoader:
    """Dataset class for loading and processing snapshot data using TensorFlow."""

    def __init__(self, cases, doping, max_shots, combine_exp_as=False, target_size=(10, 10), train_split=0.7,
                 val_split=0.1):
        """
        Initialize the dataset.
        
        Args:
            cases (list): List of cases to load ('hf', 'pi', etc.)
            doping (float): Doping value
            max_shots (int): Maximum number of shots per case
            target_size (tuple): Target size for the images (height, width)
            train_split (float): Fraction of data to use for training
        """
        self.target_size = target_size
        self.classes = cases
        # Load the raw data
        samples, labels, self.sms = loading(cases, doping, max_shots)

        # Convert to numpy arrays
        self.samples = np.array(samples, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.float32)

        if combine_exp_as:
            self.labels = self._combine_exp_as(self.labels)

        # Split into train+val and test
        train_val_samples, self.samples_test, train_val_labels, self.labels_test = train_test_split(
            self.samples, self.labels, test_size=1.0 - (train_split + val_split), random_state=42
        )

        # Split train+val into train and validation
        self.samples_train, self.samples_val, self.labels_train, self.labels_val = train_test_split(
            train_val_samples, train_val_labels, test_size=val_split / (train_split + val_split), random_state=42
        )

        # Convert to NumPy arrays
        self.samples_train = np.array(self.samples_train, dtype=np.float32)
        self.samples_test = np.array(self.samples_test, dtype=np.float32)
        self.samples_val = np.array(self.samples_val, dtype=np.float32)
        self.labels_train = np.array(self.labels_train, dtype=np.float32)
        self.labels_test = np.array(self.labels_test, dtype=np.float32)
        self.labels_val = np.array(self.labels_val, dtype=np.float32)

        # Get dimensions
        self.num_samples_train = len(self.samples_train)
        self.num_samples_test = len(self.samples_test)
        self.num_samples_val = len(self.samples_val)
        self.input_size = self.samples_train[0].shape[0]
        self.num_classes = self.labels_train[0].shape[0]

        # First reshape to square
        orig_side_length = int(np.sqrt(self.input_size))
        self.samples_train = self.samples_train.reshape(-1, orig_side_length, orig_side_length, 1)
        self.samples_test = self.samples_test.reshape(-1, orig_side_length, orig_side_length, 1)
        self.samples_val = self.samples_val.reshape(-1, orig_side_length, orig_side_length, 1)

        # Resize to target size
        self.samples_train = self._resize_samples(self.samples_train)
        self.samples_test = self._resize_samples(self.samples_test)
        self.samples_val = self._resize_samples(self.samples_val)

    def _combine_exp_as(self, labels):
        """
        Combine 'exp' and 'AS' labels into a single class.

        Args:
            labels (np.ndarray): Array of one-hot encoded labels

        Returns:
            np.ndarray: Combined labels
        """
        # Get indices of 'exp' and 'AS' classes
        exp_idx = self.classes.index("exp")
        as_idx = self.classes.index("AS")

        # Combine classes
        combined_labels = np.copy(labels)
        combined_labels[:, exp_idx] = np.logical_or(labels[:, exp_idx], labels[:, as_idx])
        combined_labels = np.delete(combined_labels, as_idx, axis=1)

        return combined_labels

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

    def get_tf_dataset(self, batch_size=32):
        """
        Create a TensorFlow Dataset.
        
        Args:
            batch_size (int): Size of each batch
            
        Returns:
            tf.data.Dataset: training dataset, validation dataset, test dataset
        """

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((self.samples_train, self.labels_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((self.samples_test, self.labels_test))
        val_dataset = tf.data.Dataset.from_tensor_slices((self.samples_val, self.labels_val))

        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset, test_dataset

    def visualize_sample(self, index, sample_type: str, show_grid=True):
        """
        Visualize a single sample.
        
        Args:
            index (int): Index of the sample to visualize
            show_grid (bool): Whether to show grid lines
            :param sample_type: Either train, test or val, Used to select appropriate samples and labels
        """

        # Get samples and labels
        if sample_type == "train":
            samples = self.samples_train
            labels = self.labels_train
        elif sample_type == "test":
            samples = self.samples_test
            labels = self.labels_test
        elif sample_type == "val":
            samples = self.samples_val
            labels = self.labels_val
        else:
            raise ValueError("Invalid sample type. Must be 'train', 'test', or 'val'.")

        sample = samples[index]
        label = labels[index]

        plt.imshow(sample[:, :, 0], cmap=plt.cm.coolwarm, interpolation='nearest')
        plt.grid(show_grid, which='both', color='black', linewidth=0.5, alpha=0.3)
        plt.title(f"Class: {np.argmax(label)} {self.classes[np.argmax(label)]}\nSize: {sample.shape[:2]}")
        plt.colorbar()
        plt.show()

    def visualize_batch(self, sample_type: str, num_samples=4, rows=2):
        """
        Visualize multiple samples in a grid.
        
        Args:
            num_samples (int): Number of samples to visualize
            rows (int): Number of rows in the visualization grid
            :param sample_type: Either train, test or val, Used to select appropriate samples and labels
        """
        # Get samples
        if sample_type == "train":
            samples = self.samples_train
            labels = self.labels_train
        elif sample_type == "test":
            samples = self.samples_test
            labels = self.labels_test
        elif sample_type == "val":
            samples = self.samples_val
            labels = self.labels_val
        else:
            raise ValueError("Invalid sample type. Must be 'train', 'test', or 'val'.")

        # Create grid
        cols = num_samples // rows
        plt.figure(figsize=(cols * 4, rows * 4))

        for i in range(num_samples):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(samples[i][:, :, 0], cmap=plt.cm.coolwarm, interpolation='nearest')
            plt.title(f"Class: {self.classes[np.argmax(labels[i])]}\nSize: {samples[i].shape[:2]}")
            plt.colorbar()
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_class_distribution(self):
        """
        Visualize the class distribution in the dataset for training and testing.

        """
        # Get class labels
        train_labels = np.argmax(self.labels_train, axis=1)
        test_labels = np.argmax(self.labels_test, axis=1)
        val_labels = np.argmax(self.labels_val, axis=1)

        # Plot histograms
        plt.figure(figsize=(20, 6))

        plt.subplot(1, 3, 1)
        plt.hist(train_labels, bins=np.arange(self.num_classes + 1) - 0.5, rwidth=0.5, alpha=0.75)
        plt.title("Training Set")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(range(self.num_classes))

        plt.subplot(1, 3, 2)
        plt.hist(test_labels, bins=np.arange(self.num_classes + 1) - 0.5, rwidth=0.5, alpha=0.75)
        plt.title("Testing Set")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(range(self.num_classes))

        plt.subplot(1, 3, 3)
        plt.hist(val_labels, bins=np.arange(self.num_classes + 1) - 0.5, rwidth=0.5, alpha=0.75)
        plt.title("Validation Set")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(range(self.num_classes))

        plt.tight_layout()
        plt.show()


def get_data_loaders(cases, doping, max_shots, combine_exp_as=False, target_size=(10, 10), batch_size=32,
                     train_split=0.8) -> tuple[
    tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, DataLoader]:
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
        tuple: (train_dataset, test_dataset, data_loader_obj)
    """
    data_loader_obj = DataLoader(cases, doping, max_shots, combine_exp_as=combine_exp_as, target_size=target_size,
                                 train_split=train_split)
    train_dataset, val_dataset, test_dataset = data_loader_obj.get_tf_dataset(batch_size=batch_size)

    return train_dataset, val_dataset, test_dataset, data_loader_obj


if __name__ == "__main__":
    train_data, val_data, test_data, data_loader = get_data_loaders(
        cases=["AS", "exp", "pi"],
        doping=6.0,
        combine_exp_as=True,
        max_shots=1000,
        train_split=0.8)

    # Visualize a single sample
    data_loader.visualize_sample(0, sample_type="train")

    # Visualize a batch of samples
    data_loader.visualize_batch(sample_type="train", num_samples=8, rows=2)

    # Visualize class distribution
    data_loader.visualize_class_distribution()
