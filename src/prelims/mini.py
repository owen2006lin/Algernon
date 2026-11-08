import os
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets
import torchvision.transforms as transforms
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# task = 1 for parity, -1 for fashion
def make_transform(task_flag):
    return transforms.Compose([
        transforms.ToTensor(),                    
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x : torch.cat([x.flatten(), torch.tensor([float(task_flag)])])),      
    ])

def load_datasets(seed=0):
    g = torch.Generator().manual_seed(seed)

    # Parity
    mnist_train = datasets.MNIST('./data', train=True,  download=True, transform=make_transform(1))
    mnist_test  = datasets.MNIST('./data', train=False, download=True, transform=make_transform(1))
    # Even -> Unif{0,2,4,6,8}, Odd -> Unif{1,3,5,7,9}
    mnist_train.targets = 2 * torch.randint(0, 5, mnist_train.targets.shape, generator=g) + mnist_train.targets % 2
    mnist_test.targets  = 2 * torch.randint(0, 5, mnist_test.targets.shape,  generator=g) + mnist_test.targets % 2

    # Fashion
    fashion_train = datasets.FashionMNIST('./data', train=True,  download=True, transform=make_transform(-1))
    fashion_test  = datasets.FashionMNIST('./data', train=False, download=True, transform=make_transform(-1))

    return mnist_train, mnist_test, fashion_train, fashion_test

# Split indices are generated once from the seed and saved, so every run sees the identical split
def make_splits(n_mnist, n_fashion, n_pretrain=500, n_val=5000, seed=0, path='./data/splits.pt'):
    config = {'n_mnist': n_mnist, 'n_fashion': n_fashion, 'n_pretrain': n_pretrain, 'n_val': n_val, 'seed': seed}
    if os.path.exists(path):
        splits = torch.load(path)
        assert splits['config'] == config, f'{path} was made with {splits["config"]}, delete it to regenerate'
        return splits

    g = torch.Generator().manual_seed(seed)
    mnist_idx   = torch.randperm(n_mnist,   generator=g)
    fashion_idx = torch.randperm(n_fashion, generator=g)
    splits = {
        'config':           config,
        'mnist_pretrain':   mnist_idx[:n_pretrain],
        'mnist_val':        mnist_idx[n_pretrain:n_pretrain + n_val],
        'mnist_finetune':   mnist_idx[n_pretrain + n_val:],          # everything left, ~54,500
        'fashion_pretrain': fashion_idx[:n_pretrain],
        'fashion_val':      fashion_idx[n_pretrain:n_pretrain + n_val],
        # rest of Fashion train is unused: Fashion never appears in fine-tuning
    }
    torch.save(splits, path)
    return splits

mnist_train, mnist_test, fashion_train, fashion_test = load_datasets()
splits = make_splits(len(mnist_train), len(fashion_train))

pretrain_data = ConcatDataset([Subset(mnist_train,   splits['mnist_pretrain']),
                               Subset(fashion_train, splits['fashion_pretrain'])])
# Same images for SFT, RL and oracle; only the source of the labels differs
finetune_data = Subset(mnist_train,   splits['mnist_finetune'])
parity_val    = Subset(mnist_train,   splits['mnist_val'])
fashion_val   = Subset(fashion_train, splits['fashion_val'])

pretrain_loader = DataLoader(pretrain_data, batch_size = 128, shuffle = True)
finetune_loader = DataLoader(finetune_data, batch_size = 128, shuffle = True)

parity_val_loader  = DataLoader(parity_val,  batch_size = 128, shuffle = False)
fashion_val_loader = DataLoader(fashion_val, batch_size = 128, shuffle = False)

parity_test_loader  = DataLoader(mnist_test,   batch_size = 128, shuffle = False)   # new-task accuracy and KL
fashion_test_loader = DataLoader(fashion_test, batch_size = 128, shuffle = False)   # forgetting


# Plotting
'''
import matplotlib.pyplot as plt
import numpy as np
def matplotlib_imshow(img, one_channel = False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(pretrain_loader)
images, labels = next(dataiter)

flags = images[:, -1]
pixels = images[:, :784].view(-1, 1, 28, 28)

img_grid = torchvision.utils.make_grid(pixels[:20])
matplotlib_imshow(img_grid, one_channel=True)

parity_classes = ['even', 'odd']     # indexed by label % 2
fashion_classes = datasets.FashionMNIST.classes

def label_name(flag, label):
    if flag == 1:
        return f'{parity_classes[label % 2]} ({label})'
    return fashion_classes[label]

print('  '.join(label_name(flags[j].item(), labels[j].item()) for j in range(20)))
plt.show()
'''

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28 + 1, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x,1)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def epoch_pass(model, optimizer, loss_fn, loader):
    running_loss = 0.0

    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, parity):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)
            if parity:
                correct += ((preds % 2) == (labels % 2)).sum().item()
            else:
                correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train(model, optimizer, loss_fn, loader, n_epochs, path):
    
    best_score = -9999999
    epoch_number = 0

    for epoch in range(n_epochs):
        print(f'Epoch {epoch_number + 1}' )
        model.train(True)
        avg_loss = epoch_pass(model, optimizer, loss_fn, loader)

        model.eval()

        parity_acc = evaluate(model, parity_val_loader, parity=True)
        fashion_acc = evaluate(model, fashion_val_loader, parity=False)
        print(f'LOSS train {avg_loss}  parity acc {parity_acc:.3f}  fashion acc {fashion_acc:.3f}')

        score = (parity_acc + fashion_acc) / 2
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), path)

        epoch_number += 1
#========================================Metrics==================================#
#accuracy = evaluate(pi_0, parity_val_loader, parity = True)

def forgetting(pi, pi_0, fashion_loader):
    old_acc = evaluate(pi_0, fashion_loader, parity = False)
    new_acc = evaluate(pi, fashion_loader, parity = False)

    return old_acc - new_acc

import math
# For testing, use only for testing
def forward_kl_single(pi_0, pi, x):
    device = next(pi_0.parameters()).device
    x = x.to(device)
    pi.eval()
    pi_0.eval()
    with torch.no_grad():
        p_logits = pi_0(x)
        q_logits = pi(x)
        p = torch.softmax(p_logits, dim = 1)
        q = torch.softmax(q_logits, dim = 1)

        sum = 0
        print(p)
        for i in range(10):
            sum += p[0][i] * math.log(p[0][i] / q[0][i])

    return sum

def forward_kl_batch(pi_0, pi, batch):
    device = next(pi_0.parameters()).device
    batch = batch.to(device)
    pi.eval()
    pi_0.eval()
    with torch.no_grad():
        p_logits = pi_0(batch)
        q_logits = pi(batch)
        p = torch.softmax(p_logits, dim = 1)
        q = torch.softmax(q_logits, dim = 1)
        
        vals = p * torch.log(p / q)
        kl = torch.sum(vals, dim = 1)
        return torch.sum(kl)

def forward_kl(pi_0, pi, parity_loader):
    kl = 0
    for _, (batch, _) in enumerate(parity_loader):
        batch_kl = forward_kl_batch(pi_0, pi, batch)
        (batch_kl)
        kl += batch_kl 
    return kl / len(parity_loader.dataset)
    
#kl = forward_kl(pi_0, pi_0, parity_test_loader)


#==========================================SFT Supervised Fine Tuning==================================#

def sft1_targets(y):
    return y % 2

def sft2_targets(y):
    c = torch.randint(0, 2, y.shape, device = y.device)
    return y % 2 + 4 * c 

def train_sft(model, scheduler, loader, optimizer, labeling, n_epochs, path):    
    best_acc = -9999999
    epoch_number = 0
    for epoch in range(n_epochs):
        epoch_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), labeling(y))

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        model.eval()

        acc = evaluate(model, parity_val_loader, parity = True)
        print(f'LOSS train {epoch_loss}  parity acc {acc:.3f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), path)
        epoch_number += 1

#print("SFT Labeling 1")
#train_sft(pi_1, 20, finetune_loader, sft1_targets, "models/sft_1")
#print("SFT Labeling 2")
#train_sft(pi_2, 20, finetune_loader, sft2_targets, "models/sft_2")

#===========================================Warmup=================================================#
from torch.optim.lr_scheduler import LambdaLR

def make_scheduler(optimizer, warmup_steps, total_steps, kind="constant"):
    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps          # linear ramp up to 1.0
        if kind == "constant":
            return 1.0
        # cosine: decay from 1.0 to 0.0 over the remaining steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_factor)

#==========================================GRPO Reinforcement Learning================================#

def train_grpo(model, n_epochs, loader, path, G = 10, lr = 1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    best_acc = -9999999
    for epoch in range(n_epochs):
        model.train()
        epoch_reward = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            dist = torch.distributions.Categorical(logits = model(x))
            actions = dist.sample((G,)).T
            rewards = (actions % 2 == (y % 2).unsqueeze(1)).float()

            advantage = (rewards - rewards.mean(1, keepdim = True)) / (rewards.std(1, keepdim = True) + 1e-8)
            logp = dist.log_prob(actions.T).T
            loss = -(advantage * logp).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_reward += rewards.mean().item()

        acc = evaluate(model, parity_val_loader, parity = True)
        print(f'reward {epoch_reward / len(loader):.3f} parity acc {acc:.3f}')

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), path)

#train_grpo(pi_rl, 20, finetune_loader, "models/rl_1")


#=========================================Full pipeline + Hyperparameter Sweeps======================================#
import numpy as np
import copy

learning_rates = np.geomspace(3e-6, 1e-3, 15)
epochs = [1,2]

# Pretraining
model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

#train(model, optimizer, loss_fn, pretrain_loader, 20, path = "models/mlp1")
# Initial distribution
pi_0 = MLP().to(device)
pi_0.load_state_dict(torch.load("models/mlp1", map_location = device, weights_only = True))



#===================================================Starter Sweep=====================================#
learning_rates = np.geomspace(1e-5, 1e-3, 6)

# SFT 1

for lr in learning_rates:
    model = copy.deepcopy(pi_0)
    loader = DataLoader(finetune_data, batch_size = 64, shuffle = True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    total_steps = len(finetune_loader)

    scheduler = make_scheduler(optimizer, warmup_steps=50,
                            total_steps=total_steps, kind="constant")

    train_sft(model, scheduler, loader, optimizer, sft1_targets, 1, f"models/starter_sweep/sft_1_{lr}")


print("Pre-trained model parity accuracy: ")
print(evaluate(pi_0, parity_test_loader, parity = True))

print("Pre-trained model fashion accuracy: ")
print(evaluate(pi_0, fashion_test_loader, parity = False))
for lr in learning_rates:
    model = MLP().to(device)
    model.load_state_dict(torch.load(f"models/starter_sweep/sft_1_{lr}", map_location = device, weights_only = True))
    print(f"Model lr = {lr} forgetting")
    print(forgetting(model, pi_0, fashion_test_loader))



