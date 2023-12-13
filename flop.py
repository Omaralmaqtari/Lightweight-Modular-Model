import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import re

#Model thats already available
net = models.densenet161()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=True)
# Extract the numerical value
flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
# Extract the unit
flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

print('Computational complexity: {:<8}'.format(macs))
print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
print('Number of parameters: {:<8}'.format(params))