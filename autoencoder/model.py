import torch.nn as nn
from torch import functional as F

class Autoencoder(nn.Module):

    def __init__(self, btl_size=2):
        self.btl_size = btl_size

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, btl_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(btl_size, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 28 * 28),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, batch_size):
        super(CBOW, self).__init__()
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim * 2, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        # print('inputs', inputs)
        # print('embed row', self.embeddings(inputs))
        embeds = self.embeddings(inputs).view((self.batch_size, -1))
        # print('embeds', embeds)
        out = F.relu(self.linear1(embeds))
        # print('linear1', out)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        # print('log_probs', log_probs)
        return log_probs



class Autoencoder_CBOW(nn.Module):

    def __init__(self, btl_size=2, vocab_size=50):
        self.btl_size = btl_size

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, btl_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(btl_size, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, vocab_size),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y