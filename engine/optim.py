"""최적화 알고리즘."""

import numpy as np


class Optimizer:
    """최적화기 기본 클래스."""

    def __init__(self, params):
        """초기 상태와 재현 가능한 파라미터를 설정한다."""
        self.params = list(params)

    def zero_grad(self):
        """모든 파라미터의 기울기를 0으로 만든다."""
        for p in self.params:
            p.zero_grad()


class SGD(Optimizer):
    """확률적 경사하강법."""

    def __init__(self, params, lr=1e-2, weight_decay=0.0):
        """초기 상태와 재현 가능한 파라미터를 설정한다."""
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        """한 번의 SGD 갱신을 수행한다."""
        for p in self.params:
            g = p.grad + self.weight_decay * p.data
            p.data = p.data - self.lr * g


class Adam(Optimizer):
    """바이어스 보정이 포함된 Adam."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        """초기 상태와 재현 가능한 파라미터를 설정한다."""
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data, dtype=np.float64) for p in self.params]
        self.v = [np.zeros_like(p.data, dtype=np.float64) for p in self.params]

    def step(self):
        """한 번의 Adam 갱신을 수행한다."""
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1**self.t)
            vh = self.v[i] / (1 - self.b2**self.t)
            p.data = p.data - self.lr * mh / (np.sqrt(vh) + self.eps)
