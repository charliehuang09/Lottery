from torch.utils.tensorboard import SummaryWriter
import torch
from model import Model
from torchsummary import summary
from torchvision import datasets, transforms
from torch.optim import Adam
import config
from tqdm import trange
from torch import nn
from torch.utils.data import DataLoader
from misc import Logger, accuracy
from lottery import Lottery
import numpy as np

np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.use_deterministic_algorithms(True)

model = Model()
summary(model, (1, 28, 28))
writer = SummaryWriter()
# writer = SummaryWriter('/Users/charlie/Documents/ML/Lottery/archive')

transform=transforms.Compose([transforms.ToTensor()])

train = datasets.MNIST('../data', train=True, download=True, transform=transform)
test = datasets.MNIST('../data', train=False, download=True, transform=transform)
train_loader = DataLoader(train, batch_size=2048)
test_loader = DataLoader(test, batch_size=1024)

device = torch.device('mps')
opt = Adam(model.parameters(), lr=config.warmup_lr)
loss_fn = nn.CrossEntropyLoss()

trainLossLogger = Logger(writer, "trainLossLogger")
trainAccuracyLogger = Logger(writer, "trainAccuracyLogger")

testLossLogger = Logger(writer, "testLossLogger")
testAccuracyLogger = Logger(writer, "testAccuracyLogger")

model = model.to(device)
model.train()

for epoch in trange(config.pretrain_epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
    
    if epoch == config.warmup_steps:
        opt = Adam(model.parameters(), lr=config.lr)

lottery = Lottery(model, config.prune_percent, config.iterations)

for iteration in range(0, config.iterations):
    print(f"Iteration {iteration + 1} ----------------------------------------------------")
    opt = Adam(model.parameters(), lr=config.lr)
    for epoch in range(config.iteration_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            model = lottery.clampWeights(model)
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            opt.step()
            trainLossLogger.add(loss.item(), len(output))
            trainAccuracyLogger.add(accuracy(output, target), 1)
        for batch_idx, (data, target) in enumerate(test_loader):
            model = lottery.clampWeights(model)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            testLossLogger.add(loss.item(), len(output))
            testAccuracyLogger.add(accuracy(output, target), 1)

        print(f"Train Loss: {trainLossLogger.get()} Train Accuracy: {trainAccuracyLogger.get()*100}% Test Loss: {testLossLogger.get()} Test Accuracy: {testAccuracyLogger.get()*100} Epoch: {epoch + 1}")
    lottery.updateMask(model)
    model = lottery.applyMask(model)

trainLossLogger.clear()
testLossLogger.clear()
trainAccuracyLogger.clear()
testAccuracyLogger.clear()

trainLossLogger.setWrite(True)
testLossLogger.setWrite(True)
trainAccuracyLogger.setWrite(True)
testAccuracyLogger.setWrite(True)
opt = Adam(model.parameters(), lr=config.lr)
print("FINAL-Training----------------------------------------------------------------")
for epoch in range(config.epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        model = lottery.clampWeights(model)
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        trainLossLogger.add(loss.item(), len(output))
        trainAccuracyLogger.add(accuracy(output, target), 1)
    for batch_idx, (data, target) in enumerate(test_loader):
        model = lottery.clampWeights(model)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        testLossLogger.add(loss.item(), len(output))
        testAccuracyLogger.add(accuracy(output, target), 1)
    print(f"Train Loss: {trainLossLogger.get()} Train Accuracy: {trainAccuracyLogger.get()*100}% Test Loss: {testLossLogger.get()} Test Accuracy: {testAccuracyLogger.get()*100} Epoch: {epoch + 1}")


print("FINAL")
print(f"Train Loss: {trainLossLogger.getMin()} Train Accuracy: {trainAccuracyLogger.getMax()*100}% Test Loss: {testLossLogger.getMin()} Test Accuracy: {testAccuracyLogger.getMax()*100} Epoch: {epoch + 1}")

writer.add_scalar("Final trainLossLogger", trainLossLogger.getMin(), 0) 
writer.add_scalar("Final trainAccuracyLogger", trainAccuracyLogger.getMax(), 0)
writer.add_scalar("Final testLossLogger", testLossLogger.getMin(), 0)
writer.add_scalar("Final testAccuracyLogger", testAccuracyLogger.getMax(), 0)

print(lottery.getMask())