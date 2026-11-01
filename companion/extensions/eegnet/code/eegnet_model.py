"""Compact EEGNet-like participant-bag encoder. Original implementation, MIT."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 64), bias=False)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-3, momentum=0.01)
        self.spatial = nn.Conv2d(8, 16, (14, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-3, momentum=0.01)
        self.depth = nn.Conv2d(16, 16, (1, 16), groups=16, bias=False)
        self.point = nn.Conv2d(16, 16, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16, eps=1e-3, momentum=0.01)
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(128, 1)

    def embedding(self, x):
        x = self.bn1(self.temporal(F.pad(x, (31, 32, 0, 0))))
        x = self.drop(F.avg_pool2d(F.elu(self.bn2(self.spatial(x))), (1, 4)))
        x = self.point(self.depth(F.pad(x, (7, 8, 0, 0))))
        return self.drop(F.avg_pool2d(F.elu(self.bn3(x)), (1, 8))).flatten(1)

    def forward(self, bags):
        n, w, c, t = bags.shape
        return self.head(self.embedding(bags.reshape(n*w, 1, c, t)).reshape(n, w, 128).mean(1)).flatten()

    def constrain(self):
        with torch.no_grad():
            for weight, limit in ((self.spatial.weight, 1.0), (self.head.weight, 0.25)):
                dims = tuple(range(1, weight.ndim))
                norm = weight.square().sum(dims, keepdim=True).sqrt().clamp_min(1e-12)
                weight.mul_((limit/norm).clamp_max(1.0))


def training_moments(windows, train_ids):
    means = [windows[i].astype(np.float64).mean(axis=(0, 2)) for i in train_ids]
    seconds = [np.square(windows[i].astype(np.float64)).mean(axis=(0, 2)) for i in train_ids]
    mean = np.mean(means, axis=0)
    scale = np.sqrt(np.maximum(np.mean(seconds, axis=0)-mean**2, 1e-12))
    return mean.astype(np.float32)[None, :, None], scale.astype(np.float32)[None, :, None]


def train_encoder(windows, train_ids, training_labels, seed, device='cpu', epochs=50):
    # Only training labels are accepted: held-out labels cannot enter this function.
    if len(train_ids) != len(training_labels) or len(set(train_ids)) != len(train_ids):
        raise ValueError('Invalid participant roster')
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    mean, scale = training_moments(windows, train_ids)
    normalized = {i: (windows[i]-mean)/scale for i in train_ids}
    model = Encoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=1e-4)
    curves = []
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(train_ids))
        total = 0.0
        seen = 0
        for start in range(0, len(order), 8):
            batch = order[start:start+8]
            bags = []
            for pos in batch:
                a = normalized[train_ids[pos]]
                bags.append(a[rng.choice(len(a), 8, replace=len(a)<8)])
            x = torch.from_numpy(np.stack(bags)).to(device)
            y = torch.as_tensor(np.asarray(training_labels)[batch], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            (loss * (len(batch)/8)).backward()
            optimizer.step()
            model.constrain()
            total += float(loss.detach().cpu())*len(batch)
            seen += len(batch)
        assert seen == len(train_ids)
        curves.append(total/seen)
    model.eval()
    return model, mean, scale, curves


def embeddings(model, windows, ids, mean, scale, device='cpu'):
    model.eval()
    out = []
    with torch.no_grad():
        for i in ids:
            a = (windows[i]-mean)/scale
            parts = []
            for start in range(0, len(a), 128):
                x = torch.from_numpy(a[start:start+128, None]).to(device)
                parts.append(model.embedding(x).cpu().numpy())
            out.append(np.concatenate(parts).astype(np.float64).mean(axis=0))
    return np.asarray(out)
