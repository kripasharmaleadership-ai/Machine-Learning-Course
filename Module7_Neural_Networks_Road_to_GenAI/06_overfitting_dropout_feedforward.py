"""
LAB 06 - Overfitting, Dropout, and the Feed-Forward Network
===========================================================

OVERFITTING (Module 4, now with millions of knobs)
    A network with enough parameters can memorise its training set, noise included. Training loss keeps
    falling; validation loss falls, bottoms out, then RISES. That turning point is where memorising begins.

THE REMEDIES
    early stopping     keep the weights from the epoch with the best validation loss
    dropout            during training, switch off each hidden unit with probability p (e.g. 0.2-0.5), a new
                       random set every batch: no unit can rely on any other, so the network learns redundant,
                       robust features. At prediction time dropout is OFF.
    weight decay (L2)  pull weights towards zero (Module 3's ridge penalty)
    more data          the best regulariser of all

DEEP LEARNING IN ONE TABLE
    architecture           built-in assumption                         typical input           lab
    feed-forward (MLP)     none: every input connects to every unit    a fixed-size vector     02-06
    CNN                    nearby pixels matter; a pattern can move     images, audio           07
    RNN / LSTM             read in order, carry a memory forward        sequences               08
    transformer            any position may look at any other           text, everything now    09-13

LEVELS
    EASY   : Watch a 17,000-parameter network memorise 200 noisy points; the validation curve turns.
    MEDIUM : Early stopping, dropout, weight decay and more data, compared.
    HARD   : A deep feed-forward classifier for handwritten digits; dropout must be OFF at prediction time;
             and dropout is no magic: on these datasets it fixes confidence more than accuracy.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from ml_lab_utils import HAS_MPL, banner, make_spirals, note, plt, save_fig, show, step
from nn_toolkit import MLP, Adam

Xa, ya = make_spirals(n=6000, noise=1.0, seed=7)                    # noisy spirals: many points sit in the "wrong" arm
X_test, y_test = Xa[4000:], ya[4000:]
X_val, y_val = Xa[3000:4000], ya[3000:4000]
X_small, y_small = Xa[:200], ya[:200]
BIG = [2, 128, 128, 1]
EPOCHS = 600

# %% =========================================================================
# LEVEL 1 (EASY): Watching a network memorise
# ============================================================================
banner("200 noisy points, 17,000 parameters: memorisation", "EASY")
big = MLP(BIG, activation="relu", loss="bce", seed=0).fit(X_small, y_small, epochs=EPOCHS, batch_size=32, optimizer=Adam(0.003),
                                                         X_val=X_val, y_val=y_val)
h = pd.DataFrame(big.history_).set_index("epoch")
best_ep = int(h.val_loss.idxmin())
print(h.loc[sorted({1, 25, best_ep, 400, EPOCHS}), ["train_loss", "val_loss", "train_acc", "val_acc"]].round(3).to_string())
show("Parameters / training points", f"{big.n_params:,} / {len(X_small)}")
show("Epoch with the lowest validation loss", best_ep)
show(f"Test accuracy after {EPOCHS} epochs", big.score(X_test, y_test))
note(f"""Training accuracy reaches {h.train_acc.iloc[-1]:.0%}, but these spirals are so noisy that no
honest model gets near that on new data: the network has memorised the noisy
points sitting in the 'wrong' arm. Validation loss bottoms out at epoch {best_ep}
({h.val_loss.min():.2f}) and then climbs to {h.val_loss.iloc[-1]:.2f}: from there on, more training makes the
model WORSE on new data. With {big.n_params // len(X_small)} parameters per training point, memorising is
easier than generalising.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Remedies compared
# ============================================================================
banner("Early stopping, dropout, weight decay, more data", "MEDIUM")
rows = {}


def fit_eval(name, X, y, epochs=EPOCHS, **kw):
    m = MLP(BIG, activation="relu", loss="bce", seed=0, **kw).fit(X, y, epochs=epochs, batch_size=32, optimizer=Adam(0.003))
    rows[name] = {"train accuracy": m.score(X, y), "test accuracy": m.score(X_test, y_test),
                  "test loss": float(m.loss_and_grad(m.forward(X_test), y_test)[0])}
    return m


fit_eval(f"no regularisation ({EPOCHS} epochs)", X_small, y_small)
fit_eval(f"early stopping (stop at epoch {best_ep})", X_small, y_small, epochs=best_ep)
fit_eval("dropout 0.3", X_small, y_small, dropout=0.3)
fit_eval("dropout 0.5", X_small, y_small, dropout=0.5)
fit_eval("weight decay (L2 = 0.001)", X_small, y_small, l2=1e-3)
fit_eval("no regularisation, 3,000 points (15x data)", Xa[:3000], ya[:3000], epochs=60)
reg = pd.DataFrame(rows).T
print(reg.round(3).to_string())
note(f"""Read the test LOSS first: {reg.iloc[0, 2]:.2f} without regularisation, about {reg.iloc[1:5, 2].mean():.2f} with any remedy.
The memorising network is not just wrong, it is CONFIDENTLY wrong, and every
remedy fixes that. Test ACCURACY does not improve (best {reg.iloc[1:5, 1].max():.0%} vs {reg.iloc[0, 1]:.0%}), and
dropout 0.5 drops it to {reg.iloc[3, 1]:.0%}: too much regularisation underfits. Only more data moves both ({reg.iloc[-1, 1]:.0%}, loss
{reg.iloc[-1, 2]:.2f}): with 15x the points there's far less room to memorise. In practice
you combine them, tune their strength on a validation set, and get as much
data as you can.""")

# %% =========================================================================
# LEVEL 3 (HARD): A deep feed-forward network for digits
# ============================================================================
banner("A feed-forward digit classifier, and two dropout surprises", "HARD")
d = load_digits()
Xd = d.data / 16.0
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, d.target, test_size=0.3, random_state=0, stratify=d.target)
step("3.1  64 pixels -> 128 -> 64 -> 10 classes, softmax cross-entropy")
ffn = MLP([64, 128, 64, 10], activation="relu", loss="ce", dropout=0.2, seed=0).fit(
    Xd_tr, yd_tr, epochs=60, batch_size=32, optimizer=Adam(0.003), X_val=Xd_te, y_val=yd_te)
show("Trainable parameters", f"{ffn.n_params:,}")
show("Test accuracy (10 classes)", ffn.score(Xd_te, yd_te))
probs = ffn.predict_proba(Xd_te[:3])
for i in range(3):
    top = np.argsort(probs[i])[::-1][:3]
    show(f"Image {i} (true {yd_te[i]}): top 3 classes", ", ".join(f"{c}: {probs[i, c]:.2f}" for c in top))
note("""The same MLP class, now with 10 outputs and a softmax: each image gets a
probability for every digit. Every pixel connects to every hidden unit; the
network has no idea which pixels are neighbours. That works for centred
8x8 digits. Lab 07 shows where it breaks and how the CNN's assumption fixes it.""")

step("3.2  Surprise 1: forget to switch dropout off and predictions become random")
amb = int(np.argmin(ffn.predict_proba(Xd_te).max(axis=1)))          # the test image the model is least sure about
x0 = Xd_te[amb:amb + 1]
show(f"Least certain test image (true digit {yd_te[amb]}): top 3", ", ".join(
    f"{c}: {ffn.predict_proba(x0)[0, c]:.2f}" for c in np.argsort(ffn.predict_proba(x0)[0])[::-1][:3]))
on = [int(np.argmax(ffn.forward(x0, train=True))) for _ in range(20)]
off = [int(np.argmax(ffn.forward(x0, train=False))) for _ in range(20)]
show("20 predictions, dropout ON (training mode)", on)
show("20 predictions, dropout OFF (evaluation mode)", off)
acc_on = np.mean([np.mean(np.argmax(ffn.forward(Xd_te, train=True), 1) == yd_te) for _ in range(10)])
show("Test accuracy in training mode / evaluation mode", f"{acc_on:.3f} / {ffn.score(Xd_te, yd_te):.3f}")
note("""In training mode a random 20% of units vanish on every call, so the same image
gets different answers from one call to the next. (Clear-cut images survive
this; borderline ones like this don't, and accuracy drops.) This is why PyTorch has model.train() and
model.eval(), and forgetting model.eval() is a classic production bug.
'Inverted dropout' scales the surviving units up by 1/(1-p) during training, so
that at evaluation time nothing needs rescaling.""")

step("3.3  Surprise 2: dropout is no magic (noisy spirals, dropout 0.3 vs none)")
rows = {}
for n in [200, 1000, 3000]:
    for p in [0.0, 0.3]:
        m = MLP(BIG, activation="relu", loss="bce", dropout=p, seed=0).fit(Xa[:n], ya[:n], epochs=max(60, EPOCHS * 200 // n),
                                                                        batch_size=32, optimizer=Adam(0.003))
        rows.setdefault(n, {}).update({f"accuracy, dropout {p}": m.score(X_test, y_test),
                                       f"loss, dropout {p}": float(m.loss_and_grad(m.forward(X_test), y_test)[0])})
dd = pd.DataFrame(rows).T
dd.index.name = "training points"
print(dd.round(3).to_string())
note(f"""With 200 points, dropout's real contribution is the loss ({dd.iloc[0, 1]:.2f} -> {dd.iloc[0, 3]:.2f}):
it stops the network from being certain about points it memorised; accuracy
even drops a little. With 1,000 and 3,000 points it makes almost no
difference either way: there's little memorising left to prevent. Dropout
was a breakthrough for the wide, data-starved networks of 2012; many large
language models, trained on vast data, use little or none in pretraining.
Regularise in proportion to how much the model can memorise, and measure.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    axes[0].plot(h.index, h.train_loss, label="training loss")
    axes[0].plot(h.index, h.val_loss, label="validation loss")
    axes[0].axvline(best_ep, color="grey", ls=":", label=f"best epoch ({best_ep})")
    axes[0].set(xscale="log", xlabel="epoch", ylabel="loss", title="Memorisation: validation loss turns", ylim=(0, 3))
    axes[0].legend()
    gx, gy = np.meshgrid(np.linspace(-3.5, 3.5, 250), np.linspace(-3.5, 3.5, 250))
    axes[1].contourf(gx, gy, big.predict_proba(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape),
                     levels=20, cmap="RdBu", alpha=0.6)
    axes[1].scatter(*X_small.T, c=y_small, cmap="RdBu", s=14, edgecolor="k", linewidth=0.3)
    axes[1].set(title="The memorising network's boundary (200 points)", xticks=[], yticks=[])
    reg["test accuracy"].plot.barh(ax=axes[2], color="C2")
    axes[2].set(xlim=(0.5, 1), title="Test accuracy by remedy")
    save_fig(fig, "06_overfitting_dropout_feedforward")

banner("KEY TAKEAWAYS")
print("""  1. Big networks memorise: training loss -> 0 while validation loss turns and rises.
  2. Early stopping, dropout and weight decay all narrow the gap; more data narrows it most.
  3. Dropout: randomly silence hidden units during training only. model.eval() at prediction time!
  4. Feed-forward networks connect everything to everything; CNNs, RNNs and transformers add assumptions.
  5. Regularisers mostly fix overconfidence; too much underfits; with lots of data they matter little.""")
