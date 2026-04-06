import numpy as np
import random
import torch
from torch.utils import data
from d2l import torch as d2l

# set device as cuda if it is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# force to use cpu
# device = 'cpu'

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

batch_size = 10000
lr = 0.03

true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = d2l.synthetic_data(true_w, true_b, 100000)
features, labels = features.to(device), labels.to(device)

w = torch.tensor([0.0, 0.0], requires_grad=True, device=device)
b = torch.zeros(1, requires_grad=True, device=device)

EPOCH = 40

for epoch in range(EPOCH):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = linreg(X, w, b)
        loss = squared_loss(y_hat, y)
        loss.sum().backward()
        sgd([w, b], lr, batch_size)
    
    with torch.no_grad():
        train_l = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f"true_w: {true_w}, true_b: {true_b}")
print(f"w: {w.data}, b: {b.data}")
