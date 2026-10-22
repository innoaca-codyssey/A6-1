"""계산 그래프의 위상 정렬과 역전파 실행."""

import numpy as np


def topological_order(output):
    """깊은 그래프에서도 재귀 없이 부모 우선 순서를 계산한다."""
    states = {}
    order = []
    stack = [(output, False)]
    while stack:
        node, exiting = stack.pop()
        identity = id(node)
        if exiting:
            states[identity] = 2
            order.append(node)
            continue
        state = states.get(identity, 0)
        if state == 2:
            continue
        if state == 1:
            raise ValueError("계산 그래프에 순환이 있습니다")
        states[identity] = 1
        stack.append((node, True))
        for parent in reversed(node.parents):
            stack.append((parent, False))
    return order


def backward(output, seed=None):
    """도달 가능한 기울기를 초기화하고 역방향 연쇄 법칙을 실행한다."""
    seed = (
        np.ones_like(output.data)
        if seed is None
        else np.asarray(seed, dtype=np.float64)
    )
    if seed.shape != output.data.shape or not np.isfinite(seed).all():
        raise ValueError("역전파 seed의 shape과 유한값을 확인하세요")
    order = topological_order(output)
    for node in order:
        node.grad = np.zeros_like(node.data)
    output.grad = seed.copy()
    for node in reversed(order):
        node._backward()
