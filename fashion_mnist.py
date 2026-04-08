from d2l import torch as d2l
import torch
from torch.utils import data
import torchvision
from torchvision import transforms
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_input = 28 * 28
num_output = 10
W = torch.normal(0, 0.01, size=(num_input, num_output), requires_grad=True, device=device)
b = torch.zeros(num_output, requires_grad=True, device=device)

def get_fashion_mnist_labels(labels):
    text_labels = [
        "t-shirt", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle boot"
    ]
    return [text_labels[int(label)] for label in labels]

def show_images(imgs, num_rows, num_cols, titles=None):
    """Plot a grid of images."""
    figsize = (num_cols * 2, num_rows * 2)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image: squeeze channel dim (1,28,28) → (28,28) for imshow
            img = img.squeeze().numpy()
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition
    
def net(X):
    return softmax(torch.matmul(X.reshape(-1, num_input), W) + b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y)), y])

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) / y.numel()

def SGD(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True
)

print(len(mnist_train), len(mnist_test))

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 1, 28, 28), 2, 9, get_fashion_mnist_labels(y))

EPOCH = 20
batch_size = 2048
lr = 0.1

# Use PyTorch DataLoader for efficient batching
train_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader  = data.DataLoader(mnist_test,  batch_size=batch_size, shuffle=False)

for epoch in range(EPOCH):
    # --- Training ---
    train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        X = X.reshape(-1, num_input)   # flatten (batch, 1, 28, 28) → (batch, 784)
        y_hat = net(X)
        loss = cross_entropy(y_hat, y)
        loss.sum().backward()
        SGD([W, b], lr, batch_size)
        train_loss_sum += loss.sum().item()
        train_acc_sum  += accuracy(y_hat, y) * len(y)
        n += len(y)

    # --- Evaluation on test set ---
    test_acc_sum, m = 0.0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            X = X.reshape(-1, num_input)
            test_acc_sum += accuracy(net(X), y) * len(y)
            m += len(y)

    print(f'epoch {epoch + 1:2d}  '
          f'loss {train_loss_sum / n:.4f}  '
          f'train acc {train_acc_sum / n:.3f}  '
          f'test acc {test_acc_sum / m:.3f}')