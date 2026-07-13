"""BC training: CNN over padded lossless encodings, cross-entropy on teacher actions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training.common import IN_CHANNELS, N_ACTIONS, N_CHANNELS, PAD_H, PAD_W, pad_obs


def build_model():
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(IN_CHANNELS, 96, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(96, 96, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(96, 64, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * PAD_H * PAD_W, 256),
        nn.ReLU(),
        nn.Linear(256, N_ACTIONS),
    )


class GroupedDataset:
    """Dataset over shape-grouped uint8 samples; pads per item."""

    def __init__(self, groups):
        self.groups = groups
        self.index = []
        for g, (obs, labels) in enumerate(groups):
            self.index.extend((g, i) for i in range(len(labels)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        g, j = self.index[i]
        obs, labels = self.groups[g]
        x = pad_obs(obs[j].astype(np.float32))
        return x, labels[j]


def train_model(groups, epochs=10, batch_size=1024, lr=1e-3, device=None, init_model=None, log=print):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model or build_model()
    model = model.to(device)

    ds = GroupedDataset(groups)
    n_val = max(1000, int(0.03 * len(ds)))
    gen = torch.Generator().manual_seed(0)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=gen)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    labels_all = np.concatenate([g[1] for g in groups])
    hist = np.bincount(labels_all, minlength=N_ACTIONS).astype(np.float64)
    log(f"  dataset: {len(ds)} samples, action histogram {hist.astype(int).tolist()}")
    # Mild inverse-frequency weighting: rare-but-critical actions (interact at
    # the right tile) matter far more for returns than common movement steps.
    weights = (hist.sum() / np.maximum(hist, 1.0)) ** 0.3
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=0.05,
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
    )

    for epoch in range(epochs):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            tot += len(y)
        sched.step()

        model.eval()
        v_tot, v_correct = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                v_correct += int((model(x).argmax(1) == y).sum())
                v_tot += len(y)
        log(
            f"  epoch {epoch + 1}/{epochs} loss={loss_sum / tot:.4f} "
            f"train_acc={correct / tot:.3f} val_acc={v_correct / v_tot:.3f}"
        )
    return model


def export_npz(model, out_path: Path):
    """Save weights for dependency-free numpy inference."""
    import torch

    layers = [m for m in model if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    arrays = {}
    for i, layer in enumerate(layers):
        arrays[f"w{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    return arrays


def numpy_forward(arrays, x):
    """Numpy replica of the model; x: (26, PAD_H, PAD_W) float32 -> logits (6,)."""

    def conv2d(x, w, b):
        c_out, c_in, kh, kw = w.shape
        _, h, wd = x.shape
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        cols = np.empty((c_in * kh * kw, h * wd), dtype=np.float32)
        idx = 0
        for ci in range(c_in):
            for i in range(kh):
                for j in range(kw):
                    cols[idx] = xp[ci, i : i + h, j : j + wd].ravel()
                    idx += 1
        out = w.reshape(c_out, -1) @ cols + b[:, None]
        return out.reshape(c_out, h, wd)

    h = conv2d(x, arrays["w0"], arrays["b0"])
    np.maximum(h, 0, out=h)
    h = conv2d(h, arrays["w1"], arrays["b1"])
    np.maximum(h, 0, out=h)
    h = conv2d(h, arrays["w2"], arrays["b2"])
    np.maximum(h, 0, out=h)
    v = h.reshape(-1)
    v = arrays["w3"] @ v + arrays["b3"]
    np.maximum(v, 0, out=v)
    return arrays["w4"] @ v + arrays["b4"]
