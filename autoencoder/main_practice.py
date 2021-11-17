
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_mnist
from trainer import Trainer

from argparse import Namespace
from model import CBOW
from model import Autoencoder_CBOW
from copy import deepcopy

config = {
    'train_ratio': .8,
    'batch_size': 256,
    'n_epochs': 5,
    'verbose': 1,
    'btl_size': 2
}

config = Namespace(**config)

print(config)

checkpoint = torch.load('./model')
EMBEDDING_DIM = checkpoint['EMBEDDING_DIM']
CONTEXT_SIZE = checkpoint['CONTEXT_SIZE']
BATCH_SIZE = checkpoint['BATCH_SIZE']
vocab = checkpoint['vocab']
model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, BATCH_SIZE)
model.load_state_dict(checkpoint['model_state_dict'])
print(model.eval())
print(model.embeddings.weight)
print(model.embeddings.weight[0])
print(len(model.embeddings.weight[0]))

train_x = model.embeddings.weight
print('train_x[0]', train_x[0])

train_cnt = int(train_x.size(0) * config.train_ratio)
valid_cnt = train_x.size(0) - train_cnt
print(f'train_cnt {train_cnt}, valid_cnt {valid_cnt}')

indices = torch.randperm(train_x.size(0))
train_x, valid_x = torch.index_select(
    train_x,
    dim=0,
    index=indices
).split([train_cnt, valid_cnt], dim=0)





print("Train:", train_x.shape)
print("Valid:", valid_x.shape)

from model import Autoencoder_CBOW
from trainer import Trainer

print(model.parameters())

model = Autoencoder_CBOW(btl_size=config.btl_size)
optimizer = optim.Adam(model.parameters())
crit = nn.MSELoss()

trainer = Trainer(model, optimizer, crit)

trainer.train((train_x, train_x), (valid_x, valid_x), config)

def show_image(x):
    print(x.size())
    if x.dim() == 1:
        x = x.view(int(x.size(0) ** .5), -1)

    plt.imshow(x, cmap='gray')
    plt.show()


if config.btl_size == 2:
    color_map = [
        'brown', 'red', 'orange', 'yellow', 'green',
        'blue', 'navy', 'purple', 'gray', 'black',
    ]

    plt.figure(figsize=(20, 10))
    with torch.no_grad():
        latents = model.encoder(train_x[:1000])
        print('latents', latents)
        plt.scatter(latents[:, 0],
                    latents[:, 1],
                    marker='o')
        plt.show()

# if config.btl_size == 2:
#     min_range, max_range = -2., 2.
#     n = 20
#     step = (max_range - min_range) / float(n)
#
#     with torch.no_grad():
#         lines = []
#
#         for v1 in np.arange(min_range, max_range, step):
#             z = torch.stack([
#                 torch.FloatTensor([v1] * n),
#                 torch.FloatTensor([v2 for v2 in np.arange(min_range,
#                                                           max_range, step)]),
#             ], dim=-1)
#
#             line = torch.clamp(model.decoder(z).view(n, 10, 5), 0, 1)
#             print(line)
#             line = torch.cat([line[i] for i in range(n - 1, 0, -1)], dim=0)
#             print(line)
#             lines += [line]
#
#         lines = torch.cat(lines, dim=-1)
#         plt.figure(figsize=(20, 20))
#         show_image(lines)









