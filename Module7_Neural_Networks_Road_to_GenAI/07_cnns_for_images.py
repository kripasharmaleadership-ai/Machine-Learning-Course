"""
LAB 07 - Convolutional Neural Networks for Images
=================================================

THE PROBLEM WITH A FEED-FORWARD NETWORK ON PIXELS
    Every pixel gets its own weight. A loop at the top-left and the same loop at the bottom-right activate
    completely different weights, so the network must learn "loop" separately for every position.

THE CNN'S ONE NEW IDEA: CONVOLUTION
    Slide a small filter (e.g. 3x3, 9 weights) across the image; at each position, multiply and sum.
    The SAME 9 weights are reused everywhere (weight sharing), so a filter that detects "vertical edge"
    detects it anywhere. Stacking layers builds edges -> strokes -> loops -> digits (or eyes -> faces).
        conv  -> ReLU -> pool (keep the max of each 2x2 block: shrink, tolerate small shifts)  -> repeat
        -> global pooling ("is the feature anywhere?") -> a small classifier
    The same principle (process locally, reuse the weights) works for audio and for the image encoders of
    multimodal GenAI models.

LEVELS
    EASY   : A convolution by hand: edge filters on a digit; move the digit and the feature map moves with it.
    MEDIUM : Digits placed anywhere on a 16x16 canvas: a feed-forward network vs a CNN, with parameter counts.
    HARD   : Train on centred digits only, test on shifted ones. Who generalises? Then: augmentation, and
             what the filters learned.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_lab_utils import HAS_MPL, banner, make_shifted_digits, note, plt, save_fig, show, step
from nn_toolkit import HAS_TORCH, conv2d, max_pool

if not HAS_TORCH:
    raise SystemExit("This lab needs PyTorch:  pip install torch")
import torch
import torch.nn as nn

from nn_toolkit import n_trainable, torch_accuracy, train_classifier

torch.set_num_threads(4)
imgs, labels, offs = make_shifted_digits(shift="random")
i_tr, i_te = train_test_split(np.arange(len(imgs)), test_size=0.3, random_state=0, stratify=labels)
X = torch.tensor(imgs[:, None], dtype=torch.float32)                 # (n, 1 channel, 16, 16)
y = torch.tensor(labels)

# %% =========================================================================
# LEVEL 1 (EASY): Convolution by hand
# ============================================================================
banner("A 3x3 filter slid across an image", "EASY")
img = imgs[0]
vertical = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], float)     # responds to left-dark / right-bright edges
horizontal = vertical.T
fv, fh = conv2d(img, vertical), conv2d(img, horizontal)
show("Image / filter / feature map shapes", f"{img.shape} * {vertical.shape} -> {fv.shape}")
show("Weights in the filter (reused at every position)", vertical.size)
show("Feature map after 2x2 max pooling", max_pool(np.abs(fv)).shape)
note("""A 3x3 filter visits all 14 x 14 positions of the 16 x 16 image, computing a
weighted sum of the 9 pixels under it each time. The result, a FEATURE MAP, is
bright wherever the image contains the pattern the filter encodes (here
vertical or horizontal strokes). Pooling keeps the strongest response in each
2x2 block: a smaller map that cares less about the exact position.""")

step("1.1  Move the digit: the feature map moves with it")
shifted = np.roll(img, (3, -2), axis=(0, 1))
f1, f2 = conv2d(img, vertical), conv2d(shifted, vertical)
show("Feature map of shifted image == shifted feature map", bool(np.allclose(np.roll(f1, (3, -2), axis=(0, 1))[3:, :-2], f2[3:, :-2])))
show("Strongest response, original / shifted", f"{np.abs(f1).max():.3f} / {np.abs(f2).max():.3f}")
note("""Convolution is EQUIVARIANT to shifts: move the input and the output moves the
same way, with the same values. Take the maximum over the whole map (global
pooling) and the answer ('is there a vertical stroke?') doesn't depend on
position at all. A feed-forward network has no such property: it must learn
it from data, position by position.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Feed-forward vs CNN on digits that move
# ============================================================================
banner("Digits anywhere on the canvas: MLP vs CNN", "MEDIUM")


def make_mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(),
                         nn.Linear(128, 10))


def make_cnn(global_pool=True):
    head = [nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(64, 10)] if global_pool else [nn.Flatten(), nn.Linear(64 * 4 * 4, 10)]
    return nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 16x16 -> 8x8
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # 8x8 -> 4x4
                         *head)


results, histories = {}, {}
for name, build in {"feed-forward (MLP)": make_mlp, "CNN": make_cnn}.items():
    torch.manual_seed(0)
    model = build()
    histories[name] = train_classifier(model, X[i_tr], y[i_tr], X[i_te], y[i_te], epochs=30, lr=2e-3)
    results[name] = {"trainable parameters": n_trainable(model), "test accuracy": histories[name][-1]["test_acc"],
                     "best test accuracy": max(h["test_acc"] for h in histories[name])}
res = pd.DataFrame(results).T
print(res.to_string(formatters={"trainable parameters": "{:,.0f}".format, "test accuracy": "{:.3f}".format,
                                "best test accuracy": "{:.3f}".format}))
note(f"""Lab 01 showed logistic regression at 18% on these images. A feed-forward
network does better ({res.iloc[0, 1]:.0%}) but it has to learn every digit at every position
from about 1,250 examples. The CNN reaches {res.iloc[1, 1]:.0%} with {res.iloc[1, 0] / res.iloc[0, 0]:.0%} of the parameters:
its filters are shared across positions, so every training image teaches
every position at once. The right assumption is worth more than more weights.""")

# %% =========================================================================
# LEVEL 3 (HARD): Train centred, test shifted
# ============================================================================
banner("Plot twist: train on centred digits, test on shifted ones", "HARD")
cen, _, _ = make_shifted_digits(shift="center")
Xc = torch.tensor(cen[:, None], dtype=torch.float32)
rows = {}
models = {}
for name, build in {"feed-forward (MLP)": make_mlp, "CNN, flatten head": lambda: make_cnn(global_pool=False),
                    "CNN, global max pooling": make_cnn}.items():
    torch.manual_seed(0)
    m = build()
    train_classifier(m, Xc[i_tr], y[i_tr], epochs=30, lr=2e-3)
    models[name] = m
    rows[name] = {"test: centred": torch_accuracy(m, Xc[i_te], y[i_te]), "test: shifted anywhere": torch_accuracy(m, X[i_te], y[i_te])}
shift_tab = pd.DataFrame(rows).T
print(shift_tab.round(3).to_string())
note(f"""Everyone is near-perfect on centred test digits. Shift them and the MLP
collapses to {shift_tab.iloc[0, 1]:.0%}: it never saw a digit off-centre. The CNN with a FLATTEN head
falls too ({shift_tab.iloc[1, 1]:.0%}): its convolutions find the strokes wherever they are, but its
final layer still has one weight per position. With GLOBAL max pooling the
question becomes 'is this feature anywhere?', and accuracy holds much better
({shift_tab.iloc[2, 1]:.0%}). Architecture choices encode assumptions; get them right and the model
generalises to situations it never saw.""")

step("3.1  The data fix: augmentation (train on randomly shifted copies)")
aug = []
rng = np.random.default_rng(0)
for k in range(4):
    a, _, _ = make_shifted_digits(shift="random", seed=100 + k)
    aug.append(a[i_tr])
Xaug = torch.tensor(np.concatenate(aug)[:, None], dtype=torch.float32)
yaug = y[i_tr].repeat(4)
torch.manual_seed(0)
mlp_aug = make_mlp()
train_classifier(mlp_aug, Xaug, yaug, epochs=30, lr=2e-3)
show("MLP trained on 4 shifted copies: test on shifted", torch_accuracy(mlp_aug, X[i_te], y[i_te]))
note("""Augmentation teaches the MLP the invariance through data: show it every digit at
many positions. It helps a lot, at 4x the training cost, and it's how image
models are still trained (crops, flips, colour changes). Architecture and
data are two ways to give a model the same knowledge.""")

step("3.2  What did the first layer learn?")
filters = models["CNN, global max pooling"][0].weight.detach().numpy()[:, 0]
show("First-layer filters (32 of them, 3x3 each)", filters.shape)
diag = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], float)
hand = {"vertical edge": vertical, "horizontal edge": horizontal, "diagonal edge": diag, "other diagonal": np.fliplr(diag)}
flat = (filters - filters.mean(axis=(1, 2), keepdims=True)).reshape(32, -1)
for nm, kern in hand.items():
    kv = (kern - kern.mean()).ravel()
    corr = np.abs(flat @ kv) / (np.linalg.norm(flat, axis=1) * np.linalg.norm(kv))
    show(f"Best match to our hand-made {nm} filter", f"filter {int(corr.argmax()):>2}, |correlation| {corr.max():.2f}")
note("""Nobody designed these filters, yet for every edge detector we wrote by hand in
Level 1 the network grew a close twin (|correlation| 0.8-0.9; see the chart).
The second layer combines such edges into loops and corners. Big image networks
learn the same progression on millions of photos.""")

if HAS_MPL:
    fig = plt.figure(figsize=(16, 4.6))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(np.hstack([img, np.pad(np.abs(fv), 1), np.pad(np.abs(fh), 1)]), cmap="magma")
    ax.set(title="Digit | vertical edges | horizontal edges", xticks=[], yticks=[])
    ax = fig.add_subplot(1, 4, 2)
    for name, h in histories.items():
        ax.plot([r["epoch"] for r in h], [r["test_acc"] for r in h], label=name)
    ax.set(xlabel="epoch", ylabel="test accuracy", title="Digits anywhere: MLP vs CNN")
    ax.legend()
    ax = fig.add_subplot(1, 4, 3)
    shift_tab.plot.bar(ax=ax, rot=15)
    ax.set(title="Trained on centred digits only", ylim=(0, 1.05))
    ax = fig.add_subplot(1, 4, 4)
    grid = np.ones((4 * 4, 8 * 4)) * filters.min()
    for k in range(32):
        r, c = divmod(k, 8)
        grid[r * 4:r * 4 + 3, c * 4:c * 4 + 3] = filters[k]
    ax.imshow(grid, cmap="RdBu")
    ax.set(title="32 learned 3x3 filters", xticks=[], yticks=[])
    save_fig(fig, "07_cnns_for_images")

banner("KEY TAKEAWAYS")
print("""  1. A convolution slides one small filter over the image: the same weights at every position.
  2. Feature maps move when the input moves (equivariance); global pooling makes the answer position-free.
  3. CNNs beat feed-forward networks on images with far fewer parameters: the right assumption is data-efficient.
  4. Train centred / test shifted: the MLP collapses, the global-pooling CNN generalises.
  5. Augmentation teaches the same invariance through data, at a training cost.""")
