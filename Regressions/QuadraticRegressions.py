import numpy
import torch
import torch.optim as optim 
import matplotlib.pyplot as plt
import pandas as pd


torch.manual_seed(42)

def train(init_type, lr, noise_level):   
    x = torch.rand(100,1, dtype = torch.float64)
    y = 1 + 2*x + 0.5*x**2 + noise_level*torch.rand(100,1, dtype = torch.float64)

    idx = (torch.rand(100)*100).long()

    train_idx = idx[:80]
    val_idx = idx[80:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    if init_type == 'zeroes':
        a = torch.zeros(1,requires_grad = True, dtype = torch.float64)
        b = torch.zeros(1,requires_grad = True, dtype = torch.float64)
        c = torch.zeros(1, requires_grad = True, dtype = torch.float64)
    elif init_type == 'N(0,1)':
        a = torch.rand(1,requires_grad = True, dtype = torch.float64)
        b = torch.rand(1,requires_grad = True, dtype = torch.float64)
        c = torch.rand(1, requires_grad = True, dtype = torch.float64)
    elif init_type == 'U(1000,2000)':
        a = torch.tensor(torch.rand(1,requires_grad = True, dtype = torch.float64)*1000 + 1000, dtype = torch.float64, requires_grad= True)
        b = torch.tensor(torch.rand(1,requires_grad = True, dtype = torch.float64)*1000 + 1000, dtype = torch.float64, requires_grad= True)
        c = torch.tensor(torch.rand(1,requires_grad = True, dtype = torch.float64)*1000 + 1000, dtype = torch.float64, requires_grad= True)

    print("initial values: ", a, b , c )
    print('\n')
    n_epochs = 1000

    optimaizer = optim.SGD([a,b,c], lr = lr)

    for epoch in range(n_epochs):
        yhat = a + b*x_train + c*x_train**2
        error = y_train - yhat
        loss = (error**2).mean()

        loss.backward()

        optimaizer.step()
        optimaizer.zero_grad()

    print("optimizer_values: ", a ,b, c)

    with torch.no_grad():
        yhat = a + b*x_train + c*x_train**2
        yres = a + b*x_val + c*x_val**2

#Расскоментируйте, если хотите посмотреть графики функций!
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_val.numpy(), y_val.numpy(), "r", lw=0, marker="^", label='True values')
    # plt.plot(x_val.numpy(), yres.numpy(), "g", lw=0, marker="+", label='Predictions')
    # plt.title(f'Init: {init_type}, LR: {lr}, Noise: {noise_level}\nParams: a={a.item():.3f}, b={b.item():.3f}, c={c.item():.3f}')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()

    
    return {'a': a,
            'b': b,
            'c': c,
            'lr': lr,
            'init_type': init_type,
            'x_val': x_val,
            'y_val': y_val,
            'x_train': x_train,
            'y_train': y_train,
            'yhat': yhat,
            'yres': yres,
            'noise_level': noise_level
            } 



init_types = ['zeroes', 'N(0,1)', 'U(1000,2000)']
learning_rates = [0.001, 0.01, 0.1, 0.5, 0.7, 0.9]
noise_levels = [0.01, 0.1, 0.5, 1.0]

results = []
for init_type in init_types:
    for lr in learning_rates:
        for noise_level in noise_levels:
            print(f"Trainig with init_types = {init_type}, lr = {lr}, noise = {noise_level}")
            result = train(init_type, lr, noise_level)
            results.append(result)
            train_error = torch.mean((result['y_train'] - result['yhat'])**2)
            val_error = torch.mean((result['y_val'] - result['yres'])**2)
            
            print(f"Train error: {train_error.item():.6f}")
            print(f"Validation error: {val_error.item():.6f}")
            print("-" * 50)

data = []
for result in results:
    row = {
        'init_type': result['init_type'],
        'lr': result['lr'],
        'noise_level': result['noise_level'],
        'a': result['a'].item(),
        'b': result['b'].item(),
        'c': result['c'].item(),
        'train_error': torch.mean((result['y_train'] - result['yhat'])**2).item(),
        'val_error': torch.mean((result['y_val'] - result['yres'])**2).item()
    }
    data.append(row)

print(f"\nВсего экспериментов: {len(data)}")

#Раскоментируйте если хотите создать DataFrame!
# df = pd.DataFrame(data)
# df.to_csv('training_results.csv', index=False)
# print("\nDataFrame сохранен в 'training_results.csv'") 