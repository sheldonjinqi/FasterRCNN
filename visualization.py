import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

epoch_num = 49
# path = os.path.join('./1113_1/box_epoch_' + str(epoch_num))
path = os.path.join('./box_epoch_' + str(epoch_num))
checkpoint = torch.load(path)
print(checkpoint.keys())
# epoch_loaded = checkpoint['epoch']

train_loss = checkpoint['train_loss']
train_total_loss = train_loss[0]
train_classfier_loss = train_loss[1]
train_regr_loss = train_loss[2]

test_loss = checkpoint['test_loss']
test_total_loss = test_loss[0]
test_classfier_loss = test_loss[1]
test_regr_loss = test_loss[2]

print(train_total_loss[-1])
print(test_total_loss[-1])

plt.figure()
plt.plot(train_total_loss,label='Training')
plt.plot(test_total_loss,label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('total_loss.png')
plt.figure()
plt.plot(train_classfier_loss, label = 'Training')
plt.plot(test_classfier_loss,label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('clas_loss.png')
plt.figure()
plt.plot(train_regr_loss, label='Training')
plt.plot(test_regr_loss,label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('regr_loss.png')

plt.show()