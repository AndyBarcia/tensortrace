      
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import numpy as np
from tensortrace import ModelTracer, stack_padded_tensors
from typing import List


class ComplexCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4x4

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool1(self.relu1(self.batch_norm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.batch_norm2(self.conv2(x))))
        x = self.pool3(self.relu3(self.batch_norm3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)  # Flatten
        x = self.fc1(x)
        x = self.relu4(self.dropout(x))
        x = self.fc2(x)
        return x


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'Batch {i + 1}, Loss: {running_loss / 100:.3f}')
            running_loss = 0.0


cifar10_labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def evaluate(model, test_loader, device):
    model.eval()
    class_correct = {i: 0 for i in range(len(cifar10_labels))}
    class_total = {i: 0 for i in range(len(cifar10_labels))}

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1

    per_class_accuracy = {cls: 100 * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
                          for cls in class_correct.keys()}
    global_accuracy = sum(class_correct.values()) / sum(class_total.values()) * 100

    return global_accuracy, per_class_accuracy


def load_cifar10(subset_size=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    if subset_size:
        train_dataset = Subset(train_dataset, range(subset_size))
        test_dataset = Subset(test_dataset, range(subset_size))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    return train_loader, test_loader


def exponential_smoothing(series, alpha=0.3):
    """
    Apply exponential smoothing to a time series.
    
    Parameters:
    - series: Input numpy array to be smoothed
    - alpha: Smoothing factor between 0 and 1 
             (lower values = more smoothing, higher values = less smoothing)
    
    Returns:
    - Smoothed series
    """
    smoothed = np.zeros_like(series)
    smoothed[0] = series[0]
    
    for t in range(1, len(series)):
        smoothed[t] = alpha * series[t] + (1 - alpha) * smoothed[t-1]
    
    return smoothed


def visualize_variance(vars):
    plt.figure(figsize=(10, 8))
    for i,var in enumerate(vars):
        plt.plot(var, label=f'BatchNorm{i} Variance')
    plt.title('Batch Normalization Variance')
    plt.xlabel('Iteration')
    plt.ylabel('Variance')
    plt.legend()
    plt.show()


def visualize_logits(logits: np.ndarray, accuracies: np.ndarray, per_class_accuracies: List[np.ndarray]):
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot the tensor data on the main axis
    for i in range(10):
        series = logits[:, i]
        smoothed_series = exponential_smoothing(series, 0.1)
        # TODO x-axis
        ax1.plot(accuracies[:,0], smoothed_series, label=cifar10_labels[i])

    # Create a second y-axis for the accuracy plot
    ax2 = ax1.twinx()
    ax2.plot(accuracies[:, 0], accuracies[:, 1], label="Accuracy", color="black", linestyle="--")

    ax1.set_title('Mean Across Batch Dimension for 10 Dimensions')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Logits across Batch Dimension')
    ax2.set_ylabel('Accuracy')

    # Add legends for both y-axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot per-class accuracies in the second subplot
    for i, class_accuracies in enumerate(per_class_accuracies):
        smoothed_series = exponential_smoothing(class_accuracies[:,1], 0.1)
        ax3.plot(class_accuracies[:,0], smoothed_series, label=f'{cifar10_labels[i]} Accuracy')

    ax3.set_title('Per-Class Accuracies')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Accuracy')
    ax3.legend(loc='upper left')
    ax3.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.show()


def visualize_tensor_tsne(tensor, accuracies, bn_var_1, bn_var_2):
    """
    Create a comprehensive visualization with 4 subplots:
    1. t-SNE colored by iteration
    2. t-SNE colored by accuracy
    3. t-SNE colored by penultimate layer BatchNorm variance
    4. t-SNE colored by last layer BatchNorm variance
    
    Parameters:
    tensor (torch.Tensor): Input tensor of shape (N, B, 512)
    accuracies (list or numpy.ndarray): List of accuracy values, one per epoch
    penultimate_bn_var (numpy.ndarray): Variance values for penultimate BatchNorm layer per iteration
    last_bn_var (numpy.ndarray): Variance values for last BatchNorm layer per iteration
    
    Returns:
    matplotlib.figure.Figure: Figure with four t-SNE visualization plots
    """
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(tensor)
    
    # Get original dimensions
    original_shape = tensor.shape
    n_iterations = original_shape[0]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # 1. First subplot: Color by iteration (as in original function)
    norm_iteration = mcolors.Normalize(vmin=0, vmax=n_iterations)
    cmap_iteration = plt.cm.rainbow
    point_indices = np.repeat(np.arange(n_iterations), 1)
    
    # Connect points to show trajectories
    for i in range(n_iterations-1):
        axes[0, 0].plot([tsne_results[i, 0], tsne_results[i+1, 0]], 
                        [tsne_results[i, 1], tsne_results[i+1, 1]],
                        color=cmap_iteration(norm_iteration(i)), 
                        alpha=0.7, linewidth=2)
    
    scatter1 = axes[0, 0].scatter(tsne_results[:, 0], tsne_results[:, 1],
                                 c=point_indices, cmap=cmap_iteration, norm=norm_iteration,
                                 alpha=0.7)
    
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('Iteration', rotation=270, labelpad=15)
    axes[0, 0].set_title(f't-SNE by Iteration\nTensor Shape: {original_shape}')
    axes[0, 0].set_xlabel('t-SNE Dimension 1')
    axes[0, 0].set_ylabel('t-SNE Dimension 2')
    
    # 2. Second subplot: Color by accuracy
    # Map accuracy values to iterations
    iterations_per_epoch = n_iterations // len(accuracies)
    
    # Create an array that maps each iteration to its corresponding accuracy
    accuracy_per_iteration = np.zeros(n_iterations)
    for i in range(len(accuracies)):
        start_idx = i * iterations_per_epoch
        end_idx = (i + 1) * iterations_per_epoch if i < len(accuracies) - 1 else n_iterations
        accuracy_per_iteration[start_idx:end_idx] = accuracies[i, 1]
    
    norm_accuracy = mcolors.Normalize(vmin=0.45, vmax=accuracies[:,1].max())
    cmap_accuracy = plt.cm.viridis
    
    # Connect points with lines colored by accuracy
    for i in range(n_iterations-1):
        axes[0, 1].plot([tsne_results[i, 0], tsne_results[i+1, 0]], 
                        [tsne_results[i, 1], tsne_results[i+1, 1]],
                        color=cmap_accuracy(norm_accuracy(accuracy_per_iteration[i])), 
                        alpha=0.7, linewidth=2)
    
    scatter2 = axes[0, 1].scatter(tsne_results[:, 0], tsne_results[:, 1],
                                 c=accuracy_per_iteration, cmap=cmap_accuracy, norm=norm_accuracy,
                                 alpha=0.7)
    
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
    cbar2.set_label('Accuracy', rotation=270, labelpad=15)
    axes[0, 1].set_title(f't-SNE by Accuracy\nTensor Shape: {original_shape}')
    axes[0, 1].set_xlabel('t-SNE Dimension 1')
    axes[0, 1].set_ylabel('t-SNE Dimension 2')
    
    # 3. Third subplot: Color by penultimate BatchNorm variance
    norm_penult_var = mcolors.Normalize(vmin=bn_var_1.min(), vmax=bn_var_1.max())
    cmap_penult_var = plt.cm.plasma
    
    # Connect points with lines colored by penultimate BN variance
    for i in range(n_iterations-1):
        axes[1, 0].plot([tsne_results[i, 0], tsne_results[i+1, 0]], 
                        [tsne_results[i, 1], tsne_results[i+1, 1]],
                        color=cmap_penult_var(norm_penult_var(bn_var_1[i])), 
                        alpha=0.7, linewidth=2)
    
    scatter3 = axes[1, 0].scatter(tsne_results[:, 0], tsne_results[:, 1],
                                 c=bn_var_1, cmap=cmap_penult_var, norm=norm_penult_var,
                                 alpha=0.7)
    
    cbar3 = plt.colorbar(scatter3, ax=axes[1, 0])
    cbar3.set_label('Penultimate BatchNorm Variance', rotation=270, labelpad=15)
    axes[1, 0].set_title(f't-SNE by Penultimate BatchNorm Variance\nTensor Shape: {original_shape}')
    axes[1, 0].set_xlabel('t-SNE Dimension 1')
    axes[1, 0].set_ylabel('t-SNE Dimension 2')
    
    # 4. Fourth subplot: Color by last BatchNorm variance
    norm_last_var = mcolors.Normalize(vmin=bn_var_2.min(), vmax=bn_var_2.max())
    cmap_last_var = plt.cm.inferno
    
    # Connect points with lines colored by last BN variance
    for i in range(n_iterations-1):
        axes[1, 1].plot([tsne_results[i, 0], tsne_results[i+1, 0]], 
                        [tsne_results[i, 1], tsne_results[i+1, 1]],
                        color=cmap_last_var(norm_last_var(bn_var_2[i])), 
                        alpha=0.7, linewidth=2)
    
    scatter4 = axes[1, 1].scatter(tsne_results[:, 0], tsne_results[:, 1],
                                 c=bn_var_2, cmap=cmap_last_var, norm=norm_last_var,
                                 alpha=0.7)
    
    cbar4 = plt.colorbar(scatter4, ax=axes[1, 1])
    cbar4.set_label('Last BatchNorm Variance', rotation=270, labelpad=15)
    axes[1, 1].set_title(f't-SNE by Last BatchNorm Variance\nTensor Shape: {original_shape}')
    axes[1, 1].set_xlabel('t-SNE Dimension 1')
    axes[1, 1].set_ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def main():
    # Create model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComplexCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load dataset
    train_loader, test_loader = load_cifar10(subset_size=1024)

    # Variables that we are interested in
    variables_to_trace = [
        "fc1.forward", # Save output of forward function
        "fc2.forward", # Save input of forward function
        "forward.self.*.running_var" # Variance of all normalization layers
    ]

    # Take the mean of all batches and all iterations collected since the last gathering.
    def mean_post_gather(name, values, iterations, ranks):
        values[0] = values[0].mean(dim=(0,1)) # (C)
        pass

    tracer = ModelTracer(
        model, 
        variables_to_trace,
        post_gather_callbacks=[stack_padded_tensors, mean_post_gather],
        post_trace_callbacks=[stack_padded_tensors],
        gathering_interval=8,
        train_only=True
    )
    
    # Train model with the tracer.
    epochs = 100
    accuracies = []
    per_class_accuracies = [[] for _ in cifar10_labels ]
    with tracer:
        for epoch in tqdm(range(epochs)):
            train(model, train_loader, criterion, optimizer, device)
            acc, per_class_acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1}/{epochs}, Accuracy: {acc:.2f}%")
            accuracies.append((tracer.current_iteration, acc))
            for i, acc in per_class_acc.items():
                per_class_accuracies[i].append((tracer.current_iteration, acc))
    
    fc1 = tracer.results["fc1.forward"].values[0] # (N,512)
    fc2 = tracer.results["fc2.forward"].values[0] # (N,10)
    vars = [
        tracer.results[f"forward.self.batch_norm{i}.running_var"].values[0] # (N,)
        for i in range(1,3+1)
    ]
    accuracies = np.array(accuracies)
    per_class_accuracies = [ np.array(v) for v in per_class_accuracies]

    visualize_logits(fc2, accuracies, per_class_accuracies)
    visualize_tensor_tsne(fc1, accuracies, vars[1], vars[2])
    visualize_variance(vars)


if __name__ == "__main__":
    main()

    