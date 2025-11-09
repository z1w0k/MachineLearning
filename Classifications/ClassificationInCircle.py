import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np

data_size = 5000
x = torch.rand(data_size)
y = torch.rand(data_size)
center = (0.5,0.5)
radius = 0.3
distance = torch.sqrt( (center[0]-x)**2+(center[1]-y)**2 )
pos_labels = distance <= radius
neg_labels = distance > radius
labels = torch.zeros(data_size).long()
labels[pos_labels] = 1

pos_data_x,pos_data_y = x[pos_labels],y[pos_labels]
neg_data_x,neg_data_y = x[neg_labels],y[neg_labels]
plt.figure("All data")
plt.plot(pos_data_x.numpy(),pos_data_y.numpy(),"r",marker="*",lw=0)
plt.plot(neg_data_x.numpy(),neg_data_y.numpy(),"b",marker="*",lw=0)

N_train = int(data_size *0.8)
train_data_x = x[:N_train]
train_data_y = y[:N_train]
train_labels = labels[:N_train]
test_data_x = x[N_train:]
test_data_y = y[N_train:]
test_labels = labels[N_train:]

plt.figure("Train and test data")
plt.plot(test_data_x.numpy(),test_data_y.numpy(),"g",marker="*",lw=0)
plt.plot(train_data_x.numpy(),train_data_y.numpy(),"m",marker="*",lw=0)
plt.show()

class MLP(nn.Module):
    def __init__(self,size,out_size):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(size, 15)
        self.mlp2 = nn.Linear(15,out_size)
        self.func = nn.ReLU()
    
    def forward(self, x):
        x = self.func(self.mlp1(x))
        return self.mlp2(x)
    
    def output_to_label(self, output, softmax):
        with torch.no_grad():
            output2 = softmax(output)
            predicted = torch.argmax(output2, 1)
            return predicted

train_data_all = torch.cat([train_data_x.unsqueeze(1),train_data_y.unsqueeze(1)],dim=1)
test_data_all = torch.cat([test_data_x.unsqueeze(1),test_data_y.unsqueeze(1)],dim=1)
model = MLP(2,2)
lr = 0.001

optimizer = optim.Adam(model.parameters(), lr=lr)
model_loss = nn.CrossEntropyLoss()

softmax = nn.Softmax(-1)
batch_size = 32
n_iterations = train_data_all.size(0) // batch_size
print("n iter:",n_iterations)
train_errors = []
test_errors = []

for epoch in range(50):
    print("### epoch:",epoch)
    epoch_loss = 0
    error = 0
    
    for iter in range(n_iterations):
        batch_idx = (torch.rand(batch_size)*train_data_all.size(0)).long()
        batch = Variable(train_data_all[batch_idx].contiguous(), requires_grad=True)
        ref = train_labels[batch_idx]

        out = model(batch)
        loss = model_loss(out, ref)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        lab = model.output_to_label(out, softmax)
        error += torch.sum(lab != ref)
        epoch_loss += loss.item()
                          
    train_error_rate = error.item()/(n_iterations*batch_size)
    train_errors.append([epoch, train_error_rate])
    print('train error:', error.item(), 'of', (n_iterations*batch_size))
    
    if (epoch % 1 == 0):
        test_error = 0
        with torch.no_grad():
            batch = test_data_all
            out = model(batch)
            lab = model.output_to_label(out, softmax)
            test_error = torch.sum(lab != test_labels)
            test_error_rate = test_error.item()/test_labels.size(0)
            test_errors.append([epoch, test_error_rate])
            print("test error:", test_error.item(), " of ", test_labels.size(0))
            print("epoch loss:", epoch_loss/n_iterations)

plt.figure("Errors")
train_errors = np.array(train_errors)
test_errors = np.array(test_errors)
plt.plot(train_errors[:,0], train_errors[:,1], "r-", label="train")
plt.plot(test_errors[:,0], test_errors[:,1], "g-", label="test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Error Rate")
plt.show()