import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd

def generate_dataset(num_samples):
    data, labels = [], []
    for _ in range(num_samples):
        # at least 3 points are needed to form a polygon. We set the upper bound to be 4 points.
        num_points = np.random.randint(3, 5)
        points = np.random.rand(num_points, 3) * 10  # random points in 3D space

        # a convex hull is the smallest convex sset to contain all the points and will have a non-zero volume.
        try:
            hull = ConvexHull(points)
            label = 1 if hull.volume > 0 else 0      # 1 - polygon is formed and 0 - no polygon is formed.
        except:
            label = 0     # set the label to zero if the hull fails to form.

        # convert the points set into a 1 dimensional input vector and standardise the length to 30.
        points = points.flatten()
        if len(points) < 30:  
            # pad the input vector with zeros until it has a length of 30.
            points = np.pad(points, (0, 30 - len(points)), 'constant')
        data.append(points)
        labels.append(label)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def process_dataset(dataset_location):
    point_dataset = pd.read_csv(dataset_location)

    # split the data in training and testing dataset
    train_size = int(0.8 * len(point_dataset))
    train_data = point_dataset[: train_size]
    test_data = point_dataset[train_size: ]

    # split dataset into feature vector (input) and target vector (output) - pytorch tensors
    X_train = torch.tensor(train_data[:, :-1].values, dtype=torch.float32)
    y_train = torch.tensor(train_data[:, -1].values, dtype=torch.float32)

    X_test = torch.tensor(test_data[:, :-1].values, dtype=torch.float32)
    y_test = torch.tensor(test_data[:, -1].values, dtype=torch.float32)

    # create  tensor datasets
    training_data = TensorDataset(X_train, y_train)
    testing_data = TensorDataset(X_test, y_test)

    # create a dataloaders for testing and training datasets
    batch_size = 16
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
