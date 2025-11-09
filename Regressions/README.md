# QUADRATIC REGRESSION

## Used libraries
1)numpy
2)torch
3)matplotlib
4)pandas

## Fucntion
### Train(init_type, lr, noise_level)
Starting with creating our functon for which we will later introduce parameters.
```
    x = torch.rand(100,1, dtype = torch.float64)
    y = 1 + 2*x + 0.5*x**2 + noise_level*torch.rand(100,1, dtype = torch.float64)

    idx = (torch.rand(100)*100).long()

    train_idx = idx[:80]
    val_idx = idx[80:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
```
After that I initialize our parameters in different ways and enable gradient calculation for them.
```
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
```
Next, optimization begins, using the SGD optimizer, and the value without the gradient is also calculated in order to later see the loss function for the training sample and validation data.

```
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
```

Afterwards, if you want, you can remove the comments to see the graphs of our functions.And at the end of the function, all parameters are returned to us in the form of a dictionary.
```
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
```
## Main
The parameters are initialized and after that the function is run with all possible parameters.
```
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
```
And after that all the things are done to create a DataFrame.
```
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

 df = pd.DataFrame(data)
 df.to_csv('training_results.csv', index=False)
 print("\nDataFrame сохранен в 'training_results.csv'") 
```
