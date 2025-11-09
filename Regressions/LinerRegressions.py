import numpy
import torch
import torch.optim as optim 
import matplotlib.pyplot as plt


torch.manual_seed(42)
x = torch.rand(100,1)
y = 1 + 2*x*torch.rand(100,1)

idx = (torch.rand(100)*100).long()

train_idx = idx[:80]
val_idx = idx[80:]
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

a = torch.rand(1,requires_grad = True, dtype = torch.float64)
b = torch.rand(1,requires_grad = True, dtype = torch.float64)
print("initial values: ", a, b)
lr = 1e-1
n_epochs = 1000

optimazire = optim.SGD([a,b], lr = lr)

for epoch in range(n_epochs):
    yhat = a + b*x_train
    error = y_train - yhat
    loss = (error**2).mean()

    loss.backward()

    optimazire.step()
    optimazire.zero_grad()

print("optimizer_vakues: ", a ,b)

with torch.no_grad():
    yhat = a + b*x_train
    yres = a + b*x_val

print("train erroe: ", torch.mean((y_train - yhat)**2))
print("validation error: ", torch.mean((y_val - yres)**2) )
plt.figure("Result")
plt.plot(x_val.numpy(), y_val.numpy(), "r", lw = 0, marker = "^")
plt.plot(x_val.numpy(), yres.numpy(), "g", lw = 0, marker = "+")
plt.show()