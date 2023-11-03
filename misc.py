import torch

class Logger:
  def __init__(self, writter, name):
    self.value = 0
    self.length = 0
    self.writter = writter
    self.name = name
    self.idx = 0
    self.max = None
    self.min = None
    self.write = False
    return
    
  def add(self, input, length):
    input /= length
    self.value += input
    self.length += 1
    return
  
  def get(self):
    output = self.value / self.length
    self.value = 0
    self.length = 0
    if self.write == True:
      self.writter.add_scalar(self.name, output, self.idx)
    self.idx += 1
    if self.max == None:
      self.max = output
    if self.min == None:
      self.min = output
    self.max = max(self.max, output)
    self.min = min(self.min, output)
    return output
  
  def getMax(self):
    return self.max
  
  def getMin(self):
    return self.min

  def clear(self):
    self.value = 0
    self.length = 0
    self.idx = 0
    self.max = None
    self.min = None
    return

  def setWrite(self, input):
    self.write = input
  
def accuracy(output, target):
  output = output.argmax(1)
  train_acc = torch.sum(output == target)
  output =  train_acc / len(output)
  return output
  

  
