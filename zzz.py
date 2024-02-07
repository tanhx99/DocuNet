import torch

n = 2
a = torch.zeros((5,5))
a[1,3] = 1
a[0,2] = 1
a[2,1] = 1
a[(n+1):] = a[n]
a[2,2] = 1
print(a)

