import os, gzip, struct, hashlib, urllib.request
import numpy as np

BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_file(root, filename, expected_md5):
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, filename)
    if os.path.exists(path):
        got = _md5(path)
        if got != expected_md5:
            raise ValueError(
                f"기존 캐시 MD5 불일치: {path} expected={expected_md5} got={got}"
            )
        return path
    part = path + ".part"
    with (
        urllib.request.urlopen(BASE_URL + filename, timeout=90) as r,
        open(part, "wb") as f,
    ):
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    got = _md5(part)
    if got != expected_md5:
        raise ValueError(
            f"다운로드 MD5 불일치: {filename} expected={expected_md5} got={got}"
        )
    os.replace(part, path)
    return path


def download_mnist_raw(root="data/mnist"):
    """원본 MNIST gzip 다운로드 및 검증."""
    out = {}
    for k, (name, md5) in FILES.items():
        out[k] = _ensure_file(root, name, md5)
    return out


def _read_images(path):
    with gzip.open(path, "rb") as f:
        buf = f.read()
    if len(buf) < 16:
        raise ValueError("이미지 파일 헤더 부족")
    magic, count, rows, cols = struct.unpack(">IIII", buf[:16])
    if magic != 2051:
        raise ValueError(f"이미지 magic 오류: {magic}")
    if rows != 28 or cols != 28:
        raise ValueError(f"이미지 크기 오류: {rows}x{cols}")
    expected = 16 + count * rows * cols
    if len(buf) != expected:
        raise ValueError(f"이미지 길이 오류: expected={expected} got={len(buf)}")
    arr = np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(count, rows * cols)
    return arr.astype(np.float32) / 255.0


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        buf = f.read()
    if len(buf) < 8:
        raise ValueError("라벨 파일 헤더 부족")
    magic, count = struct.unpack(">II", buf[:8])
    if magic != 2049:
        raise ValueError(f"라벨 magic 오류: {magic}")
    expected = 8 + count
    if len(buf) != expected:
        raise ValueError(f"라벨 길이 오류: expected={expected} got={len(buf)}")
    arr = np.frombuffer(buf, dtype=np.uint8, offset=8)
    if np.any(arr > 9):
        raise ValueError("라벨 범위 오류")
    return arr


def load_mnist(root="data/mnist"):
    """MNIST 로드. float32 이미지와 uint8 라벨 반환."""
    paths = download_mnist_raw(root)
    xtr = _read_images(paths["train_images"])
    ytr = _read_labels(paths["train_labels"])
    xte = _read_images(paths["test_images"])
    yte = _read_labels(paths["test_labels"])
    if xtr.shape[0] != ytr.shape[0] or xte.shape[0] != yte.shape[0]:
        raise ValueError("개수 불일치")
    return xtr, ytr, xte, yte


class BatchIterator:
    """시드 고정 미니배치 반복기."""

    def __init__(self, n, batch_size, shuffle=True, seed=42):
        if int(n) <= 0 or int(batch_size) <= 0:
            raise ValueError("n과 batch_size는 양수여야 합니다")
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        idx = np.arange(self.n)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(idx)
        self.epoch += 1
        for s in range(0, self.n, self.batch_size):
            yield idx[s : s + self.batch_size]
