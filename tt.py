import torch
import torch.nn as nn
from torch.autograd import Variable, grad

SIZE=[1, 1, 171*21, 171*21]
input = Variable(torch.cuda.FloatTensor(*SIZE).uniform_(), requires_grad=True)
conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, dilation=1, padding=1,bias=False).cuda()
output = conv1(input)
loss = output.sum()
loss.backward()