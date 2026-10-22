"""신경망 레이어와 손실 함수."""

import numpy as np
from .tensor import Tensor


class Module:
    """모든 레이어의 기본 클래스."""

    def __init__(self):
        """초기 상태와 재현 가능한 파라미터를 설정한다."""
        self.training = True

    def parameters(self):
        """학습 가능한 텐서 목록을 반환한다."""
        return []

    def train(self):
        """학습 모드로 전환하고 하위 모듈에도 전파한다."""
        self.training = True
        for v in self.__dict__.values():
            if isinstance(v, Module):
                v.train()
            elif isinstance(v, (list, tuple)):
                for m in v:
                    if isinstance(m, Module):
                        m.train()
        return self

    def eval(self):
        """평가 모드로 전환하고 하위 모듈에도 전파한다."""
        self.training = False
        for v in self.__dict__.values():
            if isinstance(v, Module):
                v.eval()
            elif isinstance(v, (list, tuple)):
                for m in v:
                    if isinstance(m, Module):
                        m.eval()
        return self

    def __call__(self, *args, **kwargs):
        """forward를 호출한다."""
        return self.forward(*args, **kwargs)


class Linear(Module):
    """완전연결 레이어."""

    def __init__(self, in_features, out_features, bias=True, init="xavier", seed=0):
        """초기 상태와 재현 가능한 파라미터를 설정한다."""
        super().__init__()
        rng = np.random.default_rng(seed)
        if init == "zero":
            w = np.zeros((in_features, out_features), dtype=np.float64)
        elif init == "random":
            w = rng.standard_normal((in_features, out_features)).astype(np.float64)
        elif init == "he":
            w = rng.standard_normal((in_features, out_features)) * np.sqrt(
                2.0 / in_features
            )
        elif init == "xavier":
            w = rng.standard_normal((in_features, out_features)) * np.sqrt(
                2.0 / (in_features + out_features)
            )
        else:
            raise ValueError("init must be one of: zero, random, he, xavier")
        self.weight = Tensor(w)
        self.bias = (
            Tensor(np.zeros((1, out_features), dtype=np.float64)) if bias else None
        )

    def forward(self, x):
        """선형 변환을 계산한다."""
        x = x if isinstance(x, Tensor) else Tensor(x)
        if x.data.ndim != 2:
            raise ValueError("Linear expects 2D input of shape (batch, in_features)")
        y = x @ self.weight
        return y + self.bias if self.bias is not None else y

    def parameters(self):
        """가중치와 편향을 반환한다."""
        return [self.weight] + ([] if self.bias is None else [self.bias])


class ReLU(Module):
    """ReLU 활성화 레이어."""

    def forward(self, x):
        """ReLU를 계산한다."""
        return relu(x)


class Sigmoid(Module):
    """시그모이드 활성화 레이어."""

    def forward(self, x):
        """시그모이드를 계산한다."""
        return sigmoid(x)


class Tanh(Module):
    """tanh 활성화 레이어."""

    def forward(self, x):
        """tanh를 계산한다."""
        return tanh(x)


class Softmax(Module):
    """소프트맥스 활성화 레이어."""

    def __init__(self, axis=1):
        """초기 상태와 재현 가능한 파라미터를 설정한다."""
        super().__init__()
        self.axis = axis

    def forward(self, x):
        """소프트맥스를 계산한다."""
        return softmax(x, axis=self.axis)


class LeakyReLU(Module):
    """LeakyReLU 활성화 레이어."""

    def __init__(self, negative_slope=0.01):
        """초기 상태와 재현 가능한 파라미터를 설정한다."""
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        """LeakyReLU를 계산한다."""
        return leaky_relu(x, self.negative_slope)


class ELU(Module):
    """ELU 활성화 레이어."""

    def __init__(self, alpha=1.0):
        """초기 상태와 재현 가능한 파라미터를 설정한다."""
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        """ELU를 계산한다."""
        return elu(x, self.alpha)


class Dropout(Module):
    """역드롭아웃 레이어."""

    def __init__(self, p=0.5, seed=0):
        """초기 상태와 재현 가능한 파라미터를 설정한다."""
        super().__init__()
        if not (0 <= p < 1):
            raise ValueError("dropout p must satisfy 0 <= p < 1")
        self.p = p
        self.rng = np.random.default_rng(seed)

    def forward(self, x):
        """학습 시 역드롭아웃을 적용한다."""
        x = x if isinstance(x, Tensor) else Tensor(x)
        if not self.training or self.p == 0:
            return x
        mask = (self.rng.random(x.shape) >= self.p).astype(np.float64) / (1.0 - self.p)
        return x * Tensor(mask)


class Sequential(Module):
    """레이어를 순차 연결하는 컨테이너."""

    def __init__(self, *layers):
        """초기 상태와 재현 가능한 파라미터를 설정한다."""
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        """입력을 순서대로 통과시킨다."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """모든 하위 레이어의 파라미터를 반환한다."""
        ps = []
        for layer in self.layers:
            ps.extend(layer.parameters())
        return ps

    def train(self):
        """학습 모드 전환을 전파한다."""
        self.training = True
        for l in self.layers:
            l.train()
        return self

    def eval(self):
        """평가 모드 전환을 전파한다."""
        self.training = False
        for l in self.layers:
            l.eval()
        return self


def relu(x):
    """ReLU 활성화를 계산한다."""
    x = x if isinstance(x, Tensor) else Tensor(x)
    mask = (x.data > 0).astype(np.float64)
    return x * Tensor(mask)


def sigmoid(x):
    """수치적으로 안정적인 시그모이드를 계산한다."""
    x = x if isinstance(x, Tensor) else Tensor(x)
    pos = x.data >= 0
    out = np.empty_like(x.data)
    out[pos] = 1.0 / (1.0 + np.exp(-x.data[pos]))
    ex = np.exp(x.data[~pos])
    out[~pos] = ex / (1.0 + ex)
    y = Tensor(out, (x,), "sigmoid")

    def _backward():
        """부모 기울기를 연쇄 법칙으로 누적한다."""
        x.grad += y.grad * y.data * (1.0 - y.data)

    y._backward = _backward
    return y


def tanh(x):
    """tanh 활성화를 계산한다."""
    x = x if isinstance(x, Tensor) else Tensor(x)
    t = np.tanh(x.data)
    y = Tensor(t, (x,), "tanh")

    def _backward():
        """부모 기울기를 연쇄 법칙으로 누적한다."""
        x.grad += y.grad * (1.0 - y.data**2)

    y._backward = _backward
    return y


def leaky_relu(x, negative_slope=0.01):
    """LeakyReLU 활성화를 계산한다."""
    x = x if isinstance(x, Tensor) else Tensor(x)
    mask = np.where(x.data > 0, 1.0, negative_slope)
    return x * Tensor(mask)


def elu(x, alpha=1.0):
    """양수 분기에서 불필요한 exp를 계산하지 않는 ELU."""
    x = x if isinstance(x, Tensor) else Tensor(x)
    if alpha <= 0:
        raise ValueError("alpha는 양수여야 합니다")
    negative = x.data <= 0
    values = x.data.copy()
    values[negative] = alpha * np.expm1(x.data[negative])
    y = Tensor(values, (x,), "elu")

    def _backward():
        """부모 기울기를 연쇄 법칙으로 누적한다."""
        derivative = np.ones_like(x.data)
        derivative[negative] = alpha * np.exp(x.data[negative])
        x.grad += y.grad * derivative

    y._backward = _backward
    return y


def softmax(x, axis=1):
    """안정적인 소프트맥스를 계산한다."""
    x = x if isinstance(x, Tensor) else Tensor(x)
    if axis >= x.data.ndim or axis < -x.data.ndim:
        raise ValueError("softmax axis out of range")
    m = Tensor(np.max(x.data, axis=axis, keepdims=True))
    z = x - m
    ex = z.exp()
    return ex / ex.sum(axis=axis, keepdims=True)


def binary_cross_entropy(pred, target):
    """확률 입력에 대한 이진 크로스엔트로피를 계산한다."""
    pred = pred if isinstance(pred, Tensor) else Tensor(pred)
    target = target if isinstance(target, Tensor) else Tensor(target)
    if pred.shape != target.shape:
        raise ValueError("binary_cross_entropy expects same shapes")
    eps = 1e-12
    p = pred * (1 - 2 * eps) + eps
    return -((target * p.log()) + ((1.0 - target) * (1.0 - p).log())).mean()


def softmax_cross_entropy(logits, target):
    """로짓 입력에 대한 다중분류 크로스엔트로피를 계산한다."""
    logits = logits if isinstance(logits, Tensor) else Tensor(logits)
    target = target if isinstance(target, Tensor) else Tensor(target)
    if logits.data.ndim != 2:
        raise ValueError(
            "softmax_cross_entropy expects logits with shape (batch, classes)"
        )
    if target.shape != logits.shape:
        raise ValueError(
            "target must be one-hot or probability matrix with same shape as logits"
        )
    m = Tensor(np.max(logits.data, axis=1, keepdims=True))
    z = logits - m
    log_probs = z - z.exp().sum(axis=1, keepdims=True).log()
    return -(target * log_probs).sum(axis=1).mean()


def binary_cross_entropy_with_logits(logits, target):
    """극단 로짓에서도 안정적인 이진 교차 엔트로피."""
    z = logits if isinstance(logits, Tensor) else Tensor(logits)
    y = target if isinstance(target, Tensor) else Tensor(target)
    if z.shape != y.shape or np.any((y.data < 0) | (y.data > 1)):
        raise ValueError("로짓/타겟 shape과 타겟 범위를 확인하세요")
    values = np.maximum(z.data, 0) - z.data * y.data + np.log1p(np.exp(-np.abs(z.data)))
    loss = Tensor(values, (z, y), "bce_logits")

    def _backward():
        """부모 기울기를 연쇄 법칙으로 누적한다."""
        probabilities = sigmoid(z).data
        z.grad += loss.grad * (probabilities - y.data)
        y.grad -= loss.grad * z.data

    loss._backward = _backward
    return loss.mean()
