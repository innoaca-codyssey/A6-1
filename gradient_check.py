"""NumPy 엔진의 실제 연산과 검증."""

import sys
import numpy as np

from engine import (
    Tensor,
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
    LeakyReLU,
    ELU,
    Dropout,
    binary_cross_entropy_with_logits,
    softmax_cross_entropy,
)

EPS = 1e-5
TOL = 1e-7


def _rerr(a, n):
    """두 기울기의 노름 상대 오차를 계산한다."""
    na = np.linalg.norm(a)
    nn = np.linalg.norm(n)
    d = na + nn
    if d == 0.0:
        return 0.0
    return np.linalg.norm(a - n) / d


def _numgrad(loss_fn, arrays):
    """중심차분을 계산하고 입력 값을 복원한다."""
    grads = []
    for arr in arrays:
        g = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            old = arr[idx]
            arr[idx] = old + EPS
            lp = float(loss_fn())
            arr[idx] = old - EPS
            lm = float(loss_fn())
            arr[idx] = old
            g[idx] = (lp - lm) / (2.0 * EPS)
            it.iternext()
        grads.append(g)
    return grads


def _check(name, build, arrays):
    """해석 기울기와 독립 수치 미분을 비교한다."""
    ts = [Tensor(a.copy()) for a in arrays]
    loss = build(ts)
    loss.backward()
    analytic = [t.grad.copy() for t in ts]

    def wrapped():
        """입력을 바꾼 순전파 손실을 계산한다."""
        ts2 = [Tensor(a.copy()) for a in arrays]
        return build(ts2).data.item()

    numeric = _numgrad(wrapped, arrays)
    ok = True
    for i, (a, n) in enumerate(zip(analytic, numeric)):
        ma = float(np.max(np.abs(a - n))) if a.size else 0.0
        re = float(_rerr(a, n))
        finite = (
            np.isfinite(a).all()
            and np.isfinite(n).all()
            and np.isfinite(ma)
            and np.isfinite(re)
        )
        print(f"{name}[{i}] max_abs={ma:.3e} rel={re:.3e}")
        if (not finite) or re >= TOL:
            ok = False
    return ok


def run_all():
    """지원 연산과 레이어의 수치 검사를 실행한다."""
    rng = np.random.default_rng(123)
    checks = []

    a = rng.normal(size=(2, 3)) * 0.3
    b = rng.normal(size=(3,)) * 0.2
    checks.append(("add_broadcast", lambda ts: ((ts[0] + ts[1]).pow(2)).mean(), [a, b]))

    a = rng.normal(size=(2, 3)) * 0.4 + 0.2
    b = rng.normal(size=(2, 3)) * 0.3 + 0.5
    checks.append(
        (
            "mul_div_pow",
            lambda ts: (((ts[0] * ts[1]) / (ts[1] + 1.7)).pow(3)).mean(),
            [a, b],
        )
    )

    a = rng.normal(size=(2, 4)) * 0.3
    b = rng.normal(size=(4, 3)) * 0.2
    checks.append(("matmul_both", lambda ts: ((ts[0] @ ts[1]).pow(2)).mean(), [a, b]))

    a = rng.normal(size=(2, 3)) * 0.2 + 1.2
    checks.append(
        (
            "reductions_reshape",
            lambda ts: ts[0].reshape((3, 2)).sum(axis=1, keepdims=True).mean(),
            [a],
        )
    )

    a = rng.normal(size=(2, 3)) * 0.1 + 0.3
    checks.append(("exp_log", lambda ts: (ts[0].exp().log()).mean(), [a]))

    x = rng.normal(size=(3, 4)) * 0.2 + 0.1
    w = rng.normal(size=(4, 2)) * 0.2
    b = rng.normal(size=(1, 2)) * 0.1

    def linear_build(ts):
        """실제 Linear 레이어의 입력/가중치/편향을 검증한다."""
        layer = Linear(4, 2)
        layer.weight, layer.bias = ts[1], ts[2]
        return layer(ts[0]).pow(2).mean()

    checks.append(("linear_layer", linear_build, [x, w, b]))

    x = rng.normal(size=(2, 3)) * 0.4 + 0.3
    checks.append(("relu", lambda ts: ReLU()(ts[0]).pow(2).mean(), [x]))
    checks.append(("sigmoid", lambda ts: Sigmoid()(ts[0]).pow(2).mean(), [x]))
    checks.append(("tanh", lambda ts: Tanh()(ts[0]).pow(2).mean(), [x]))
    checks.append(("leaky_relu", lambda ts: LeakyReLU(0.1)(ts[0]).pow(2).mean(), [x]))
    checks.append(("elu", lambda ts: ELU(1.3)(ts[0]).pow(2).mean(), [x]))

    x = rng.normal(size=(2, 4)) * 0.3
    weights = rng.normal(size=(2, 4))
    checks.append(
        ("softmax_weighted", lambda ts: (Softmax()(ts[0]) * Tensor(weights)).sum(), [x])
    )

    x = rng.normal(size=(2, 3)) * 0.2 + 0.4
    checks.append(("shared_graph", lambda ts: ((ts[0] * ts[0]) + ts[0]).mean(), [x]))

    logits = rng.normal(size=(4, 1)) * 0.3
    target = (rng.random((4, 1)) > 0.5).astype(np.float64)
    checks.append(
        (
            "bce_logits",
            lambda ts, target=target: binary_cross_entropy_with_logits(
                ts[0], Tensor(target)
            ),
            [logits],
        )
    )

    logits = rng.normal(size=(3, 4)) * 0.3
    target = np.zeros((3, 4), dtype=np.float64)
    target[np.arange(3), np.array([1, 3, 0])] = 1.0
    checks.append(
        (
            "categorical_ce",
            lambda ts, target=target: softmax_cross_entropy(ts[0], Tensor(target)),
            [logits],
        )
    )

    x = rng.normal(size=(3, 4)) * 0.2

    def drop_build(ts):
        """고정 마스크 Dropout의 손실을 계산한다."""
        d = Dropout(0.25, seed=7)
        return d(ts[0]).pow(2).mean()

    checks.append(("dropout_fixed_seed", drop_build, [x]))

    ok = True
    for name, build, arrs in checks:
        arrs = [a.copy() for a in arrs]
        if not _check(name, build, arrs):
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run_all())
