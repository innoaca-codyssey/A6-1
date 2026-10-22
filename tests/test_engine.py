"""NumPy 엔진의 실제 연산과 검증."""

import math
import unittest
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
    Sequential,
    Dropout,
    binary_cross_entropy_with_logits,
    softmax_cross_entropy,
    SGD,
    Adam,
)
import gradient_check


class TestEngine(unittest.TestCase):
    def test_deep_graph_without_recursion(self):
        """긴 그래프도 재귀 제한에 걸리지 않고 미분한다."""
        x = Tensor(2.0)
        result = x
        for _ in range(1500):
            result = result + 1
        result.backward()
        self.assertEqual(x.grad.item(), 1.0)

    def test_nonfinite_seed_rejected(self):
        """비유한 seed를 계산에 전파하지 않는다."""
        with self.assertRaises(ValueError):
            Tensor(2.0).backward(np.nan)

    def test_elu_extreme_positive_is_not_exponentiated(self):
        """ELU 양수 분기에서 지수 overflow가 발생하지 않는다."""
        x = Tensor([[1000.0, -1000.0]])
        result = ELU(1.5)(x)
        result.sum().backward()
        np.testing.assert_array_equal(result.data, [[1000.0, -1.5]])
        np.testing.assert_array_equal(x.grad, [[1.0, 0.0]])

    """정의한 수학적 값과 경계 조건을 확인한다."""

    def test_gradient_checker_runs(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        rc = gradient_check.run_all()
        self.assertEqual(rc, 0)

    def test_add_broadcast_grad(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64))
        b = Tensor(np.array([10.0, 20.0, 30.0], dtype=np.float64))
        y = (x + b).sum()
        y.backward()
        np.testing.assert_allclose(x.grad, np.ones((2, 3)))
        np.testing.assert_allclose(b.grad, np.array([2.0, 2.0, 2.0]))

    def test_matmul_explicit_grad(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        xdat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        wdat = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        x = Tensor(xdat.copy())
        w = Tensor(wdat.copy())
        y = (x @ w).sum()
        y.backward()
        exp_x = np.ones((2, 2)) @ wdat.T
        exp_w = xdat.T @ np.ones((2, 2))
        np.testing.assert_allclose(x.grad, exp_x)
        np.testing.assert_allclose(w.grad, exp_w)

    def test_shared_graph_grad(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        xdat = np.array([[2.0, -3.0]], dtype=np.float64)
        x = Tensor(xdat.copy())
        y = (x * x + x).sum()
        y.backward()
        np.testing.assert_allclose(x.grad, 2 * xdat + 1)

    def test_backward_reset_semantics(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        x = Tensor(np.array([[3.0, 4.0]], dtype=np.float64))
        y = (x * x).sum()
        y.backward()
        np.testing.assert_allclose(x.grad, np.array([[6.0, 8.0]]))
        y.backward()
        np.testing.assert_allclose(x.grad, np.array([[6.0, 8.0]]))

    def test_power_zero_gradient(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        x = Tensor(np.array([[0.0, 2.0, -3.0]], dtype=np.float64))
        y = x.pow(0).sum()
        y.backward()
        np.testing.assert_allclose(y.data, np.array(3.0))
        np.testing.assert_allclose(x.grad, np.zeros_like(x.data))

    def test_log_tiny_positive_exact(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        v = 1e-13
        x = Tensor(np.array([[v]], dtype=np.float64))
        y = x.log()
        y.backward()
        self.assertAlmostEqual(y.data.item(), math.log(v), places=15)
        self.assertAlmostEqual(x.grad.item(), 1.0 / v, places=1)

    def test_log_nonpositive_raises(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        with self.assertRaises(ValueError):
            Tensor(np.array([0.0], dtype=np.float64)).log()
        with self.assertRaises(ValueError):
            Tensor(np.array([-1.0], dtype=np.float64)).log()

    def test_exp_overflow_raises(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        with self.assertRaises(FloatingPointError):
            Tensor(np.array([[1000.0]], dtype=np.float64)).exp()

    def test_bce_logits_extreme(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        logits = Tensor(np.array([[-1000.0], [1000.0]], dtype=np.float64))
        target = Tensor(np.array([[1.0], [0.0]], dtype=np.float64))
        loss = binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        self.assertAlmostEqual(float(loss.data), 1000.0, places=12)
        np.testing.assert_allclose(
            logits.grad.reshape(-1), np.array([-0.5, 0.5]), atol=1e-12
        )

    def test_softmax_ce_extreme(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        logits = Tensor(np.array([[1000.0, -1000.0]], dtype=np.float64))
        target = Tensor(np.array([[0.0, 1.0]], dtype=np.float64))
        loss = softmax_cross_entropy(logits, target)
        loss.backward()
        self.assertAlmostEqual(float(loss.data), 2000.0, places=12)
        np.testing.assert_allclose(
            logits.grad.reshape(-1), np.array([1.0, -1.0]), atol=1e-12
        )

    def test_relu_sigmoid_tanh_values(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        xdat = np.array([[-1.0, 0.5]], dtype=np.float64)
        x = Tensor(xdat.copy())
        y = ReLU()(x).sum()
        y.backward()
        np.testing.assert_allclose(x.grad, np.array([[0.0, 1.0]]))
        x = Tensor(xdat.copy())
        y = Sigmoid()(x).sum()
        y.backward()
        s = 1.0 / (1.0 + np.exp(-xdat))
        np.testing.assert_allclose(x.grad, s * (1 - s))
        x = Tensor(xdat.copy())
        y = Tanh()(x).sum()
        y.backward()
        np.testing.assert_allclose(x.grad, 1 - np.tanh(xdat) ** 2)

    def test_leakyrelu_elu_grads(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        xdat = np.array([[-1.0, 2.0]], dtype=np.float64)
        x = Tensor(xdat.copy())
        y = LeakyReLU(0.1)(x).sum()
        y.backward()
        np.testing.assert_allclose(x.grad, np.array([[0.1, 1.0]]))
        x = Tensor(xdat.copy())
        a = 1.5
        y = ELU(a)(x).sum()
        y.backward()
        np.testing.assert_allclose(x.grad, np.array([[a * np.exp(-1.0), 1.0]]))

    def test_softmax_weighted_grad_formula(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        xdat = np.array([[0.2, -0.1, 0.4]], dtype=np.float64)
        w = np.array([[1.0, -2.0, 0.5]], dtype=np.float64)
        x = Tensor(xdat.copy())
        y = (Softmax()(x) * Tensor(w)).sum()
        y.backward()
        s = np.exp(xdat - np.max(xdat, axis=1, keepdims=True))
        s = s / s.sum(axis=1, keepdims=True)
        expected = s * (w - (s * w).sum(axis=1, keepdims=True))
        np.testing.assert_allclose(x.grad, expected, atol=1e-12)

    def test_linear_input_weight_bias_grads(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        layer = Linear(2, 3, bias=True, init="zero", seed=0)
        layer.weight.data[:] = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
        )
        layer.bias.data[:] = np.array([[0.5, -0.5, 1.0]], dtype=np.float64)
        xdat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        x = Tensor(xdat.copy())
        out = layer(x)
        loss = out.sum()
        loss.backward()
        np.testing.assert_allclose(x.grad, np.ones((2, 3)) @ layer.weight.data.T)
        np.testing.assert_allclose(layer.weight.grad, xdat.T @ np.ones((2, 3)))
        np.testing.assert_allclose(layer.bias.grad, np.array([[2.0, 2.0, 2.0]]))

    def test_bad_backward_seed_shape(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        x = Tensor(np.array([[1.0, 2.0]], dtype=np.float64))
        y = x * 2
        with self.assertRaises(ValueError):
            y.backward(np.ones((3,), dtype=np.float64))

    def test_dropout_mean_scaling_and_eval(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        xdat = np.ones((1000, 8), dtype=np.float64)
        d = Dropout(0.25, seed=123)
        d.train()
        out = d(Tensor(xdat)).data
        self.assertAlmostEqual(float(out.mean()), 1.0, places=1)
        d.eval()
        out2 = d(Tensor(xdat)).data
        np.testing.assert_allclose(out2, xdat)

    def test_sequential_train_eval(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        model = Sequential(Linear(3, 2), Dropout(0.5, seed=1), ReLU())
        model.eval()
        self.assertFalse(model.training)
        self.assertTrue(all(not layer.training for layer in model.layers))
        np.testing.assert_array_equal(
            model(Tensor(np.ones((20, 3)))).data, model(Tensor(np.ones((20, 3)))).data
        )
        model.train()
        self.assertTrue(model.training)

    def test_initializers_variance(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        fan_in, fan_out = 64, 32
        he = Linear(fan_in, fan_out, init="he", seed=0)
        xv = Linear(fan_in, fan_out, init="xavier", seed=0)
        rnd = Linear(fan_in, fan_out, init="random", seed=0)
        z = Linear(fan_in, fan_out, init="zero", seed=0)
        self.assertAlmostEqual(float(z.weight.data.var()), 0.0, places=15)
        self.assertTrue(np.isfinite(he.weight.data.var()))
        self.assertTrue(np.isfinite(xv.weight.data.var()))
        self.assertAlmostEqual(float(rnd.weight.data.var()), 1.0, delta=0.12)
        self.assertAlmostEqual(
            float(he.weight.data.var()), 2.0 / fan_in, delta=(2.0 / fan_in) * 0.12
        )
        self.assertAlmostEqual(
            float(xv.weight.data.var()),
            2.0 / (fan_in + fan_out),
            delta=(2.0 / (fan_in + fan_out)) * 0.12,
        )

    def test_sgd_coupled_l2(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        p = Tensor(np.array([[1.0, -2.0]], dtype=np.float64))
        p.grad = np.array([[0.3, -0.4]], dtype=np.float64)
        opt = SGD([p], lr=0.1, weight_decay=0.5)
        opt.step()
        expected = np.array([[1.0, -2.0]]) - 0.1 * (
            np.array([[0.3, -0.4]]) + 0.5 * np.array([[1.0, -2.0]])
        )
        np.testing.assert_allclose(p.data, expected)

    def test_adam_first_two_steps(self):
        """정의한 수학적 값과 경계 조건을 확인한다."""
        p = Tensor(np.array([1.0], dtype=np.float64))
        opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
        beta1, beta2 = 0.9, 0.999
        grad1 = 0.2 + 0.1 * 1.0
        m1 = (1 - beta1) * grad1
        v1 = (1 - beta2) * (grad1**2)
        m1h = m1 / (1 - beta1)
        v1h = v1 / (1 - beta2)
        p.grad = np.array([0.2], dtype=np.float64)
        opt.step()
        expected1 = 1.0 - 0.1 * m1h / (np.sqrt(v1h) + 1e-8)
        np.testing.assert_allclose(p.data, np.array([expected1]), atol=1e-12)
        grad2 = -0.1 + 0.1 * expected1
        m2 = beta1 * m1 + (1 - beta1) * grad2
        v2 = beta2 * v1 + (1 - beta2) * (grad2**2)
        m2h = m2 / (1 - beta1**2)
        v2h = v2 / (1 - beta2**2)
        p.grad = np.array([-0.1], dtype=np.float64)
        opt.step()
        expected2 = expected1 - 0.1 * m2h / (np.sqrt(v2h) + 1e-8)
        np.testing.assert_allclose(p.data, np.array([expected2]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
