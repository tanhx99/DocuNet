import json
import torch
import torch.nn as nn

num_labels = 2
reduced_dim = 256

ent_rep_spec = torch.randn((65, num_labels, reduced_dim))

classifier = nn.Parameter(torch.randn(num_labels, reduced_dim, reduced_dim))
classifier_bais = nn.Parameter(torch.randn(num_labels))
bilinear = nn.Bilinear(reduced_dim, reduced_dim, 2)

# logits = torch.matmul(ent_rep_spec.unsqueeze(3), classifier.expand(ent_rep_spec.shape[0], ent_rep_spec.shape[1], num_labels, reduced_dim, reduced_dim))
# print(logits.size())
# logits = torch.mul(logits.squeeze(3).unsqueeze(2).repeat(1, 1, 35, 1, 1), ent_rep_spec.unsqueeze(1).repeat(1, 35, 1, 1, 1)).sum(dim=-1)
# print(logits.size())

# logits = torch.matmul(ent_rep_spec.unsqueeze(3), classifier.expand(ent_rep_spec.shape[0], num_labels, reduced_dim, reduced_dim))
# print(logits.size())
# logits = torch.mul(logits.squeeze(2).unsqueeze(1).repeat(1, 65, 1, 1), ent_rep_spec.unsqueeze(0).repeat(65, 1, 1, 1)).sum(dim=-1)
# print(logits.size())

logits = torch.einsum("nkd,kdp,nkp->nk", ent_rep_spec, classifier, ent_rep_spec) + classifier_bais
# logits = bilinear(ent_rep_spec, ent_rep_spec)
print(logits.size())

a = torch.randn([4,8])
b = torch.randn([4,8])
print(torch.stack([a]).mean(0).size())

