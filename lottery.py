import torch
import numpy as np
import copy

class Lottery:
  def __init__(self, model, prune_percent, iterations, device = torch.device('mps')):
    self.prune_percent = prune_percent 
    self.device = device
    self.mask = None
    self.original_model = self.getWeights(model)
    self.iterations = iterations
    self.idx = 1

  def getWeights(self, model):
    with torch.no_grad():
      weights = []
      for name, param in model.named_parameters():
        if name.endswith('.weight'):
          weights.append(param.data.cpu().numpy())
      return weights
    
  def large_final(self, init_weight, current_weight):
    return abs(current_weight)

  def makeMask(self, model):
    mask = []
    original_model = self.original_model
    for i in range(len(model)):
      mask.append(self.large_final(original_model[i], model[i]))

    flat_mask = []
    for i in range(len(mask)):
      kernal = mask[i].flatten()
      for i in range(len(kernal)):
        flat_mask.append(kernal[i])
    flat_mask = sorted(flat_mask)
    prune_threshold = flat_mask[round(len(flat_mask) * min(1.0, 1.0 * self.idx / self.iterations) * self.prune_percent)]

    for i in range(len(mask)):
      mask[i][np.abs(mask[i]) < prune_threshold] = 0
      mask[i][mask[i] != 0] = 1

    self.mask = mask
  
  def applyMask(self, model):
    if self.mask == None:
      return model
    mask = self.mask
    model = model.cpu()
    i = 0
    for name, param in model.named_parameters():
      if name.endswith('.weight'):
        param.data = torch.from_numpy(self.original_model[i] * mask[i])
        i += 1
    
    model = model.to(self.device)
    return model

  def displayWeights(self, model):
    model = self.getWeights(model)
    print(model[0][:10])

  def updateMask(self, model):
    self.makeMask(self.getWeights(model))
    self.idx += 1

  def clampWeights(self, model):
    if self.mask == None:
      return model
    mask = self.mask
    model = model.cpu()
    i = 0
    for name, param in model.named_parameters():
      if name.endswith('.weight'):
        param.data *= mask[i]
        i += 1

    model = model.to(self.device)
    return model

  def getIdx(self):
    return self.idx
  
  def getMask(self):
    return self.mask