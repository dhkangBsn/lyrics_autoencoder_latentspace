
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
    'train_ratio': 1,
    'batch_size': 256,
    'n_epochs': 5,
    'verbose': 1,
    'btl_size': 1
}

config = Namespace(**config)

print(config)

checkpoint = torch.load('./model_temp')
EMBEDDING_DIM = checkpoint['EMBEDDING_DIM']
CONTEXT_SIZE = checkpoint['CONTEXT_SIZE']
BATCH_SIZE = checkpoint['BATCH_SIZE']
vocab = checkpoint['vocab']
embedding = torch.tensor(checkpoint['embedding_document'])

model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, BATCH_SIZE)
model.load_state_dict(checkpoint['model_state_dict'])
print(model.eval())

train_x = embedding
print('train_x[0]', train_x[0])
print('train_x length', len(train_x))

train_cnt = int(train_x.size(0) * config.train_ratio)
valid_cnt = train_x.size(0) - train_cnt
print(f'train_cnt {train_cnt}, valid_cnt {valid_cnt}')

indices = torch.randperm(train_x.size(0))
# train_x, valid_x = torch.index_select(
#     train_x,
#     dim=0,
#     index=indices
# ).split([train_cnt, valid_cnt], dim=0)


print("Train:", train_x.shape)
# print("Valid:", valid_x.shape)

from model import Autoencoder_CBOW
from trainer import Trainer

print(model.parameters())

model = Autoencoder_CBOW(btl_size=config.btl_size)
optimizer = optim.Adam(model.parameters())
crit = nn.MSELoss()


# get the layers as a list
model_children = list(model.children())
print('model_children', model_children)

trainer = Trainer(model, optimizer, crit)

trainer.train((train_x, train_x), (train_x, train_x), config)

def show_image(x):
    print(x.size())
    if x.dim() == 1:
        x = x.view(int(x.size(0) ** .5), -1)

    plt.imshow(x, cmap='gray')
    plt.show()


if config.btl_size == 2:

    plt.figure(figsize=(20, 10))
    with torch.no_grad():
        latents = model.encoder(train_x)
        print('train length', len(train_x))
        print('latents', latents)
        print('latents length', len(latents))
        plt.scatter(latents[:, 0],
                    latents[:, 1],
                    marker='o')
        plt.show()
elif config.btl_size == 1:
    plt.figure(figsize=(20, 10))
    with torch.no_grad():
        latents = model.encoder(train_x)
        print('train length', len(train_x))
        print('latents', latents)
        print('latents length', len(latents))
        plt.plot(latents, 'o')
        plt.show()

# sklearn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
# 시각화
import seaborn as sns
import matplotlib.pyplot as plt

# 최적의 K 찾기 : 군집 갯수 k 찾기
from sklearn import metrics
from scipy.spatial.distance import cdist




# K=50 개의 클러스터에 대해서 시각화
distortions = []
K = range(2, 100)
# tqdm.pandas()

for k in K:
    k_means = KMeans(n_clusters=k, random_state=42).fit(latents)
    k_means.fit(latents)
    distortions.append(
        sum(np.min(cdist(latents, k_means.cluster_centers_, 'euclidean'), axis=1)) / latents.shape[0])

    print('Found distortion for {} clusters'.format(k))

# Visualization
X_line = [K[0], K[-1]]
Y_line = [distortions[0], distortions[-1]]

# Plot the elbow
plt.plot(K, distortions, 'b-')
plt.plot(X_line, Y_line, 'r')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Optimal K')
plt.show()

# clustering
k = 13
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(latents)

# tsne
if config.btl_size == 2:
    tsne = TSNE(verbose=1, perplexity=100, random_state=42)     # perplexity : 유사정도
    X_embedded = tsne.fit_transform(latents)
    print('Embedding shape 확인', X_embedded.shape)

    # 시각화
    sns.set(rc={'figure.figsize':(10,10)})
    # colors
    # palette = sns.hls_palette(10, l=.4, s=.9)
    # plot
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred,
                    legend='full')     # kmeans로 예측

    plt.title('t-SNE with KMeans Labels and word Embedding using cbow')
    # plt.savefig(BASE_DIR + "/t-sne_question_glove_embedding.png")
    plt.show()

if config.btl_size == 1:
    tsne = TSNE(verbose=1, perplexity=100, random_state=42, n_components=1)  # perplexity : 유사정도
    X_embedded = tsne.fit_transform(latents)
    print('Embedding shape 확인', X_embedded.shape)

    # 시각화
    sns.set(rc={'figure.figsize': (10, 10)})
    # colors
    # palette = sns.hls_palette(10, l=.4, s=.9)
    # plot
    import itertools
    X_embedded = list(itertools.chain(*X_embedded))
    sns.scatterplot(X_embedded, y_pred)

    plt.title('t-SNE with KMeans Labels and Glove Embedding')
    # plt.savefig(BASE_DIR + "/t-sne_question_glove_embedding.png")
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









