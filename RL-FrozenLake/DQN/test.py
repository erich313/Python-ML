import torch

q = torch.load('frozen_lake_dql.pt')

num = 8

print(q)

# for i in range(num):
#     for j in range(num):
#         print(f'state{i*num+j}: {q[i*num+j]}')
