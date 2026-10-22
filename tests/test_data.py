"""MNIST 바이너리 경계와 직접 미니배치의 검증."""

import gzip
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from engine.data import _read_images, _read_labels, BatchIterator


class DataTests(unittest.TestCase):
    def fixture(self, payload):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "sample.gz"
        with gzip.open(path, "wb") as f:
            f.write(payload)
        return path

    def test_idx_pixels_and_labels(self):
        pixels = np.arange(784, dtype=np.uint8)
        x = _read_images(
            self.fixture(struct.pack(">IIII", 2051, 1, 28, 28) + pixels.tobytes())
        )
        np.testing.assert_allclose(x[0], pixels / 255, atol=1e-7)
        y = _read_labels(self.fixture(struct.pack(">II", 2049, 2) + bytes([0, 9])))
        np.testing.assert_array_equal(y, [0, 9])

    def test_idx_bad_headers_and_lengths(self):
        for payload in (
            b"",
            struct.pack(">IIII", 2050, 1, 28, 28) + bytes(784),
            struct.pack(">IIII", 2051, 1, 28, 28) + bytes(783),
        ):
            with self.assertRaises(ValueError):
                _read_images(self.fixture(payload))
        with self.assertRaises(ValueError):
            _read_labels(self.fixture(struct.pack(">II", 2049, 1) + bytes([10])))

    def test_batches_reproducible_complete_and_partial(self):
        a = BatchIterator(11, 4, seed=42)
        b = BatchIterator(11, 4, seed=42)
        first = list(a)
        np.testing.assert_array_equal(np.concatenate(first), np.concatenate(list(b)))
        np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(11))
        self.assertEqual([len(v) for v in first], [4, 4, 3])
        self.assertFalse(np.array_equal(np.concatenate(first), np.concatenate(list(a))))
        with self.assertRaises(ValueError):
            BatchIterator(11, 0)
