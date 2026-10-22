import os, csv, json, math, argparse
import numpy as np
import matplotlib.pyplot as plt
from engine.data import load_mnist, BatchIterator
from engine import (
    Tensor,
    Linear,
    Sequential,
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ELU,
    Dropout,
    Adam,
    binary_cross_entropy_with_logits,
    softmax_cross_entropy,
    sigmoid,
)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def scalar(x):
    return float(np.asarray(x).item())


def check_finite(name, x):
    arr = np.asarray(x)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"nonfinite: {name}")


def onehot(y, k):
    a = np.zeros((len(y), k), dtype=np.float64)
    a[np.arange(len(y)), y.astype(np.int64)] = 1.0
    return a


def accuracy_binary_from_logits(logits, y):
    p = sigmoid(logits).data.reshape(-1)
    pred = (p >= 0.5).astype(np.uint8)
    return float((pred == y.reshape(-1).astype(np.uint8)).mean())


def accuracy_multiclass(logits, y):
    pred = np.argmax(logits.data, axis=1)
    return float((pred == y.astype(np.int64)).mean())


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def activation_by_name(name):
    n = name.lower()
    if n == "tanh":
        return Tanh()
    if n == "leakyrelu":
        return LeakyReLU()
    if n == "elu":
        return ELU()
    raise ValueError(name)


def build_mlp(sizes, act_cls, init, seed, dropout_p=0.0):
    layers = []
    li = 0
    for i in range(len(sizes) - 2):
        layers.append(Linear(sizes[i], sizes[i + 1], init=init, seed=seed + li))
        li += 1
        layers.append(act_cls())
        if dropout_p > 0:
            layers.append(Dropout(dropout_p, seed=seed + 100 + li))
            li += 1
    layers.append(Linear(sizes[-2], sizes[-1], init=init, seed=seed + li))
    return Sequential(*layers)


def evaluate_binary(model, x, y, batch=512):
    model.eval()
    losses, correct, total = [], 0, 0
    yt = y.reshape(-1, 1).astype(np.float64)
    for s in range(0, len(x), batch):
        xb = Tensor(x[s : s + batch].astype(np.float64))
        tb = Tensor(yt[s : s + batch])
        logits = model(xb)
        loss = binary_cross_entropy_with_logits(logits, tb)
        losses.append(scalar(loss.data) * len(xb.data))
        pred = (sigmoid(logits).data.reshape(-1) >= 0.5).astype(np.uint8)
        yy = y[s : s + batch].astype(np.uint8)
        correct += int((pred == yy).sum())
        total += len(yy)
    return float(np.sum(losses) / total), correct / total


def evaluate_multiclass(model, x, y, batch=512):
    model.eval()
    losses, correct, total = [], 0, 0
    for s in range(0, len(x), batch):
        xb = Tensor(x[s : s + batch].astype(np.float64))
        yb = y[s : s + batch].astype(np.int64)
        logits = model(xb)
        loss = softmax_cross_entropy(logits, Tensor(onehot(yb, 10)))
        losses.append(scalar(loss.data) * len(xb.data))
        pred = np.argmax(logits.data, axis=1)
        correct += int((pred == yb).sum())
        total += len(yb)
    return float(np.sum(losses) / total), correct / total


def cmd_xor(args):
    ensure_dir(args.output)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([0, 1, 1, 0], dtype=np.uint8)
    configs = [("zero", "Zero"), ("random", "Random"), ("he", "He")]
    if args.include_xavier:
        configs.append(("xavier", "Xavier"))
    all_rows, summary = [], {}
    plt.figure()
    for init, label in configs:
        model = Sequential(
            Linear(2, 16, init=init, seed=args.seed),
            ReLU(),
            Linear(16, 1, init=init, seed=args.seed + 1),
        )
        opt = Adam(model.parameters(), lr=0.05, weight_decay=0.0)
        losses = []
        snaps = {}
        for epoch in range(args.epochs + 1):
            xb = Tensor(x)
            yb = Tensor(y.reshape(-1, 1).astype(np.float64))
            logits = model(xb)
            loss = binary_cross_entropy_with_logits(logits, yb)
            lv = scalar(loss.data)
            check_finite("xor_loss", lv)
            acc = accuracy_binary_from_logits(logits, y)
            losses.append(lv)
            all_rows.append(
                {"config": label, "epoch": epoch, "loss": lv, "accuracy": acc}
            )
            if epoch in (0, 50, 100):
                snaps[str(epoch)] = {"loss": lv, "accuracy": acc}
            if epoch < args.epochs:
                model.train()
                loss.backward()
                opt.step()
        plt.plot(range(len(losses)), losses, label=label)
        fail_zero = bool(
            label == "Zero"
            and len(losses) > 50
            and abs(losses[50] - losses[0]) <= 1e-12
        )
        success_he = bool(
            label == "He" and (losses[-1] < 0.1 or all_rows[-1]["accuracy"] >= 0.95)
        )
        summary[label] = {
            "epoch0": snaps.get("0"),
            "epoch50": snaps.get("50"),
            "epoch100": snaps.get("100"),
            "zero_failure": fail_zero,
            "he_success_target": success_he,
        }
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "xor_loss.png"))
    plt.close()
    save_csv(
        os.path.join(args.output, "xor_history.csv"),
        all_rows,
        ["config", "epoch", "loss", "accuracy"],
    )
    with open(
        os.path.join(args.output, "xor_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


def stratified_split(y, val_per_class=500, seed=42):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in range(10):
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        va.append(idx[:val_per_class])
        tr.append(idx[val_per_class:])
    train_indices, val_indices = np.concatenate(tr), np.concatenate(va)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def cmd_mnist(args):
    ensure_dir(args.output)
    xtr, ytr, xte, yte = load_mnist(args.data)
    tri, vai = stratified_split(ytr, 500, seed=args.seed)
    xva, yva = xtr[vai], ytr[vai]
    xtr, ytr = xtr[tri], ytr[tri]
    if args.train_limit and args.train_limit > 0:
        xtr, ytr = xtr[: args.train_limit], ytr[: args.train_limit]
    model = Sequential(
        Linear(784, 128, init="he", seed=args.seed),
        ReLU(),
        Linear(128, 10, init="he", seed=args.seed + 1),
    )
    opt = Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        it = BatchIterator(len(xtr), args.batch, shuffle=True, seed=args.seed + epoch)
        batch_losses = []
        for idx in it:
            xb = Tensor(xtr[idx].astype(np.float64))
            yb = ytr[idx].astype(np.int64)
            logits = model(xb)
            loss = softmax_cross_entropy(logits, Tensor(onehot(yb, 10)))
            lv = scalar(loss.data)
            check_finite("mnist_loss", lv)
            batch_losses.append(lv)
            loss.backward()
            opt.step()
        tr_loss, tr_acc = evaluate_multiclass(model, xtr, ytr, args.batch)
        va_loss, va_acc = evaluate_multiclass(model, xva, yva, args.batch)
        te_loss, te_acc = evaluate_multiclass(model, xte, yte, args.batch)
        row = {
            "epoch": epoch,
            "batch_loss_mean": float(np.mean(batch_losses)),
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "test_loss": te_loss,
            "test_acc": te_acc,
        }
        hist.append(row)
        print(json.dumps(row, ensure_ascii=False))
    save_csv(os.path.join(args.output, "mnist_history.csv"), hist, list(hist[0].keys()))
    with open(
        os.path.join(args.output, "mnist_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {"config": vars(args), "history": hist}, f, ensure_ascii=False, indent=2
        )
    plot_mnist(hist, args.output)


def plot_mnist(hist, output):
    """한 번의 측정도 확인할 수 있도록 막대그래프로 표시한다."""
    plt.figure(figsize=(6, 4))
    if len(hist) == 1:
        values = [hist[0][key] for key in ("train_acc", "val_acc", "test_acc")]
        bars = plt.bar(["Train (55,000)", "Validation (5,000)", "Test (10,000)"], values)
        for bar, value in zip(bars, values):
            plt.text(bar.get_x()+bar.get_width()/2, value+.01, f"{value:.2%}", ha="center")
        plt.ylim(0, 1.04)
        plt.title("MNIST after one epoch")
    else:
        for key in ("train_acc", "val_acc", "test_acc"):
            plt.plot([r["epoch"] for r in hist], [r[key] for r in hist], marker="o", label=key)
        plt.xlabel("epoch")
        plt.legend()
    plt.ylabel("accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output, "mnist_accuracy.png"))
    plt.close()


def make_moons(n, noise, seed):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    t1 = rng.uniform(0, math.pi, n1)
    t2 = rng.uniform(0, math.pi, n2)
    x1 = np.c_[np.cos(t1), np.sin(t1)]
    x2 = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    x = np.vstack([x1, x2]) + rng.normal(0.0, noise, size=(n, 2))
    y = np.array([0] * n1 + [1] * n2, dtype=np.uint8)
    p = rng.permutation(n)
    return x[p].astype(np.float64), y[p]


def l2_norm(params):
    s = 0.0
    for p in params:
        s += float((p.data**2).sum())
    return math.sqrt(s)


def train_reg_variant(name, model, opt, data, epochs, batch, seed):
    xtr, ytr, xva, yva, xte, yte = data
    hist = []
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in BatchIterator(len(xtr), batch, shuffle=True, seed=seed + epoch):
            xb = Tensor(xtr[idx])
            yb = Tensor(ytr[idx].reshape(-1, 1).astype(np.float64))
            logits = model(xb)
            loss = binary_cross_entropy_with_logits(logits, yb)
            lv = scalar(loss.data)
            check_finite(name + "_loss", lv)
            loss.backward()
            opt.step()
        tr_loss, tr_acc = evaluate_binary(model, xtr, ytr, batch)
        va_loss, va_acc = evaluate_binary(model, xva, yva, batch)
        hist.append(
            {
                "variant": name,
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "weight_norm": l2_norm(model.parameters()),
            }
        )
    te_loss, te_acc = evaluate_binary(model, xte, yte, batch)
    hist[-1].update(test_loss=te_loss, test_acc=te_acc)
    return hist


def cmd_regularization(args):
    ensure_dir(args.output)
    xtr, ytrue = make_moons(120, args.noise, args.seed)
    xva, yva = make_moons(500, args.noise, args.seed + 1)
    xte, yte = make_moons(500, args.noise, args.seed + 2)
    ytr = ytrue.copy()
    rng = np.random.default_rng(args.seed + 3)
    flip_n = int(round(0.2 * len(ytr)))
    flip_idx = rng.choice(len(ytr), size=flip_n, replace=False)
    ytr[flip_idx] = 1 - ytr[flip_idx]
    act = lambda: activation_by_name(args.activation)

    def mk(drop):
        return Sequential(
            Linear(2, 64, init="he", seed=args.seed),
            act(),
            Dropout(0.2, seed=args.seed + 10) if drop else Sequential(),
            Linear(64, 64, init="he", seed=args.seed + 1),
            act(),
            Dropout(0.2, seed=args.seed + 11) if drop else Sequential(),
            Linear(64, 1, init="he", seed=args.seed + 2),
        )

    base = mk(False)
    reg = mk(True)
    reg_params = reg.parameters()
    assert all(
        np.array_equal(a.data, b.data) for a, b in zip(base.parameters(), reg_params)
    )
    base_hist = train_reg_variant(
        "baseline",
        base,
        Adam(base.parameters(), lr=args.lr, weight_decay=0.0),
        (xtr, ytr, xva, yva, xte, yte),
        args.epochs,
        args.batch,
        args.seed,
    )
    reg_hist = train_reg_variant(
        "dropout_l2",
        reg,
        Adam(reg_params, lr=args.lr, weight_decay=0.001),
        (xtr, ytr, xva, yva, xte, yte),
        args.epochs,
        args.batch,
        args.seed,
    )
    hist = base_hist + reg_hist
    save_csv(
        os.path.join(args.output, "regularization_history.csv"),
        hist,
        list(hist[-1].keys()),
    )
    summary = {
        "generator": {
            "train": 120,
            "validation": 500,
            "test": 500,
            "noise": args.noise,
            "seed": args.seed,
        },
        "flipcount": flip_n,
        "final": {
            "baseline": base_hist[-1],
            "dropout_l2": reg_hist[-1],
            "baseline_gap": base_hist[-1]["val_loss"] - base_hist[-1]["train_loss"],
            "dropout_l2_gap": reg_hist[-1]["val_loss"] - reg_hist[-1]["train_loss"],
        },
    }
    with open(
        os.path.join(args.output, "regularization_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    plt.figure()
    for name, h in [("baseline", base_hist), ("dropout_l2", reg_hist)]:
        e = [r["epoch"] for r in h]
        plt.plot(e, [r["train_loss"] for r in h], label=name + "_train")
        plt.plot(e, [r["val_loss"] for r in h], label=name + "_val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "regularization_loss.png"))
    plt.close()
    print(json.dumps(summary, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="figures/")
    p.add_argument("--seed", type=int, default=42)
    sp = p.add_subparsers(dest="cmd", required=True)

    px = sp.add_parser("xor")
    px.add_argument("--epochs", type=int, default=100)
    px.add_argument("--include-xavier", action="store_true")

    pm = sp.add_parser("mnist")
    pm.add_argument("--data", default="data/mnist")
    pm.add_argument("--epochs", type=int, default=1)
    pm.add_argument("--train-limit", type=int, default=0)
    pm.add_argument("--batch", type=int, default=128)
    pm.add_argument("--lr", type=float, default=0.001)

    pr = sp.add_parser("regularization")
    pr.add_argument("--epochs", type=int, default=800)
    pr.add_argument("--batch", type=int, default=64)
    pr.add_argument("--lr", type=float, default=0.01)
    pr.add_argument("--noise", type=float, default=0.12)
    pr.add_argument(
        "--activation", default="Tanh", choices=["Tanh", "LeakyReLU", "ELU"]
    )

    args = p.parse_args()
    if args.cmd == "xor":
        cmd_xor(args)
    elif args.cmd == "mnist":
        cmd_mnist(args)
    else:
        cmd_regularization(args)


if __name__ == "__main__":
    main()
