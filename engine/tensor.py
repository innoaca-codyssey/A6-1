"""NumPy 기반 자동미분 텐서."""

import numpy as np


def _arr(x):
    """Tensor에서 NumPy 데이터를 얻는다."""
    return x.data if isinstance(x, Tensor) else np.array(x, dtype=np.float64)


def _sum_to_shape(g, shape):
    """브로드캐스팅 역전파를 위해 기울기를 목표 모양으로 축소한다."""
    g = np.array(g, dtype=np.float64)
    if shape == ():
        return np.array(g.sum(), dtype=np.float64)
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


class Tensor:
    """데이터, 기울기, 그래프 정보를 담는 텐서.

    backward는 현재 출력에서 도달 가능한 그래프의 모든 grad를 먼저 0으로 재설정한 뒤,
    위상 정렬 역순으로 기울기를 전파한다. 같은 노드가 여러 경로에서 공유되어도 누적이 올바르다.
    """

    __array_priority__ = 1000

    def __init__(self, data, parents=(), op=""):
        """초기 상태와 재현 가능한 파라미터를 설정한다."""
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.parents = tuple(parents)
        self._backward = lambda: None
        self.op = op

    @property
    def shape(self):
        """데이터 모양을 반환한다."""
        return self.data.shape

    def zero_grad(self):
        """현재 텐서의 기울기를 0으로 만든다."""
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def backward(self, grad=None):
        """기울기를 초기화한 뒤 AutoGrad 엔진으로 전파한다.

        seed 미지정 시 출력의 합에 대한 미분을 계산한다.
        """
        from .autograd import backward

        backward(self, grad)

    def __add__(self, other):
        """덧셈을 수행한다."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            """부모 기울기를 연쇄 법칙으로 누적한다."""
            self.grad += _sum_to_shape(out.grad, self.shape)
            other.grad += _sum_to_shape(out.grad, other.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __sub__(self, other):
        """뺄셈을 수행한다."""
        return self + (-(other if isinstance(other, Tensor) else Tensor(other)))

    def __rsub__(self, other):
        """오른쪽 피연산자 기준 뺄셈을 수행한다."""
        return Tensor(other) + (-self)

    def __neg__(self):
        """부호 반전을 수행한다."""
        out = Tensor(-self.data, (self,), "neg")

        def _backward():
            """부모 기울기를 연쇄 법칙으로 누적한다."""
            self.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """원소별 곱셈을 수행한다."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            """부모 기울기를 연쇄 법칙으로 누적한다."""
            self.grad += _sum_to_shape(out.grad * other.data, self.shape)
            other.grad += _sum_to_shape(out.grad * self.data, other.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        """원소별 나눗셈을 수행한다."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        if np.any(other.data == 0):
            raise ZeroDivisionError("division by zero in Tensor")
        out = Tensor(self.data / other.data, (self, other), "/")

        def _backward():
            """부모 기울기를 연쇄 법칙으로 누적한다."""
            self.grad += _sum_to_shape(out.grad / other.data, self.shape)
            other.grad += _sum_to_shape(
                -out.grad * self.data / (other.data**2), other.shape
            )

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        """오른쪽 피연산자 기준 나눗셈을 수행한다."""
        return Tensor(other) / self

    def __pow__(self, power):
        """상수 지수 거듭제곱을 수행한다."""
        if isinstance(power, Tensor):
            raise TypeError("only constant power is supported")
        power = float(power)
        if not np.isfinite(power):
            raise ValueError("지수는 유한한 상수여야 합니다")
        if power < 0 and np.any(self.data == 0):
            raise ZeroDivisionError("음수 지수의 밑은 0일 수 없습니다")
        if np.any((self.data <= 0) & (power != int(power))):
            raise ValueError("fractional power requires positive base")
        out = Tensor(self.data**power, (self,), f"pow{power}")

        def _backward():
            """부모 기울기를 연쇄 법칙으로 누적한다."""
            if power != 0:
                self.grad += out.grad * (power * (self.data ** (power - 1)))

        out._backward = _backward
        return out

    def pow(self, power):
        """상수 지수 연산의 메서드 형식이다."""
        return self**power

    def __matmul__(self, other):
        """2차원 행렬곱을 수행한다."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("matmul supports only 2D tensors")
        if self.data.shape[1] != other.data.shape[0]:
            raise ValueError(
                f"matmul shape mismatch: {self.data.shape} @ {other.data.shape}"
            )
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            """부모 기울기를 연쇄 법칙으로 누적한다."""
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        """합계를 계산한다."""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            """부모 기울기를 연쇄 법칙으로 누적한다."""
            g = out.grad
            axes = axis if isinstance(axis, tuple) or axis is None else (axis,)
            if not keepdims and axis is not None:
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                for a in sorted(axes):
                    g = np.expand_dims(g, a)
            self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """평균을 계산한다."""
        if axis is None:
            n = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
            n = int(np.prod([self.data.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / n

    def exp(self):
        """지수 함수를 계산한다."""
        with np.errstate(over="raise", invalid="raise"):
            out_data = np.exp(self.data)
        out = Tensor(out_data, (self,), "exp")

        def _backward():
            """부모 기울기를 연쇄 법칙으로 누적한다."""
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def log(self):
        """자연로그를 계산한다."""
        if np.any(self.data <= 0):
            raise ValueError("log 입력은 양수여야 합니다")
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            """부모 기울기를 연쇄 법칙으로 누적한다."""
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def reshape(self, *shape):
        """모양을 변경한다."""
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            """부모 기울기를 연쇄 법칙으로 누적한다."""
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def detach(self):
        """그래프와 분리된 텐서를 반환한다."""
        return Tensor(self.data.copy())

    def __repr__(self):
        """디버깅용 표현을 반환한다."""
        return f"Tensor(data={self.data}, grad={self.grad})"
