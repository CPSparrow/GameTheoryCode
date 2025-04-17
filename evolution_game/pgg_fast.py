import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from matplotlib.colors import ListedColormap
from time import time
from typing import Union

time_total, time_x, time_u, time_f = 0, 0, 0, 0
time_phi, time_s, time_op = 0, 0, 0


class PggModel:
    def __init__(self, rho: float, l: float, r: float, size: int):
        self.rho = rho
        self.l = l
        self.r = r
        self.size = size
        self.PolicyTable = np.full((size, size), 0.5)

        n_total = size * size
        n_ones = int(n_total * (1 - rho))
        converse = np.zeros(n_total, dtype=int)
        converse[:n_ones] = 1
        np.random.shuffle(converse)
        converse.resize(size, size)

        self.IsConverse = converse

    @staticmethod
    def get_n(X: np.ndarray) -> np.ndarray:
        # 计算上下左右邻居的存在性掩码
        exist_up = np.zeros_like(X, dtype=int)
        exist_up[1:, :] = 1
        exist_down = np.zeros_like(X, dtype=int)
        exist_down[:-1, :] = 1
        exist_left = np.zeros_like(X, dtype=int)
        exist_left[:, 1:] = 1
        exist_right = np.zeros_like(X, dtype=int)
        exist_right[:, :-1] = 1

        # 计算邻居的X值之和
        up = np.roll(X, 1, axis=0)
        up[0, :] = 0
        down = np.roll(X, -1, axis=0)
        down[-1, :] = 0
        left = np.roll(X, 1, axis=1)
        left[:, 0] = 0
        right = np.roll(X, -1, axis=1)
        right[:, -1] = 0
        sum_neighbors = up + down + left + right

        n_cooperators = X + sum_neighbors
        n_individuals = 1 + exist_up + exist_down + exist_left + exist_right
        return n_cooperators / n_individuals

    @staticmethod
    def get_phi(X: np.ndarray, r: float) -> np.ndarray:
        n = PggModel.get_n(X)

        # 计算邻居的n值之和
        up_n = np.roll(n, 1, axis=0)
        up_n[0, :] = 0
        down_n = np.roll(n, -1, axis=0)
        down_n[-1, :] = 0
        left_n = np.roll(n, 1, axis=1)
        left_n[:, 0] = 0
        right_n = np.roll(n, -1, axis=1)
        right_n[:, -1] = 0

        sum_neighbors_n = n + up_n + down_n + left_n + right_n
        return sum_neighbors_n * (r - 1 * X)

    @staticmethod
    def get_s(phi: np.ndarray) -> np.ndarray:
        if True:
            return np.tanh(2 * (phi - 2))
        else:
            phi -= phi.mean()
            a, b = phi.min(), phi.max()
            return np.tanh(2 * phi / max(abs(b), abs(a)))

    def get_x(self) -> np.ndarray:
        return (np.random.rand(*self.PolicyTable.shape) < self.PolicyTable).astype(int)

    def get_u(self, X: np.ndarray) -> np.ndarray:
        global time_phi, time_s, time_op
        a = time()
        phi = self.get_phi(X, self.r)
        b = time()
        s_tabel = self.get_s(phi)
        c = time()

        p = self.PolicyTable
        x_mask = X.astype(bool)
        s_ge_0 = s_tabel >= 0

        # 向量化计算u
        u_1 = np.where(s_ge_0, p + (1 - p) * s_tabel, p + p * s_tabel)
        u_0 = np.where(s_ge_0, p - p * s_tabel, p - (1 - p) * s_tabel)
        u = np.where(x_mask, u_1, u_0)
        # u = np.clip(u, 0, 1)

        d = time()
        time_phi += b - a
        time_s += c - b
        time_op += d - c
        return u

    def get_f(self, X: np.ndarray) -> np.ndarray:
        q = 0.01
        converse_flag = np.sign(X.mean()-0.5).astype(int)
        converse_flag *= -1

        # 计算每个位置的邻居多数意见
        up = np.roll(X, 1, axis=0)
        up[0, :] = 0
        down = np.roll(X, -1, axis=0)
        down[-1, :] = 0
        left = np.roll(X, 1, axis=1)
        left[:, 0] = 0
        right = np.roll(X, -1, axis=1)
        right[:, -1] = 0
        sum_neighbors = up + down + left + right

        exist_up = np.zeros_like(X, dtype=int)
        exist_up[1:, :] = 1
        exist_down = np.zeros_like(X, dtype=int)
        exist_down[:-1, :] = 1
        exist_left = np.zeros_like(X, dtype=int)
        exist_left[:, 1:] = 1
        exist_right = np.zeros_like(X, dtype=int)
        exist_right[:, :-1] = 1
        n_neighbors = exist_up + exist_down + exist_left + exist_right

        candidate_flag = np.sign(sum_neighbors/n_neighbors-0.5)

        flag = np.where(self.IsConverse, converse_flag, candidate_flag)
        omega = 0.5 * (1 + (1 - 2 * q) * X * flag)
        f = np.where(X, 1 - omega, omega)
        return f

    def forward(self):
        a = time()
        X = self.get_x()
        b = time()
        u = self.get_u(X)
        c = time()
        f = self.get_f(X)
        d = time()

        global time_x, time_u, time_f
        time_x += b - a
        time_u += c - b
        time_f += d - c

        self.PolicyTable = (1 - self.l) * u + self.l * f
        return X.mean()

    def train(self, n_steps: Union[int, float], n_logs: int) -> np.ndarray:
        if not isinstance(n_steps, int):
            n_steps = int(n_steps)
        log_steps = max(n_steps // n_logs, 1)
        fc_list = []
        for step in tqdm(range(n_steps), ncols=80):
            fc = self.forward()
            if step % log_steps == 0:
                fc_list.append(fc)
        return np.array(fc_list)


def main():
    n_steps, n_logs = 1e4, 250
    model = PggModel(rho=0.99, l=0.5, r=4.2, size=100)
    fc_logs = model.train(n_steps, n_logs)

    plt.figure(dpi=200)
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
    x = np.arange(len(fc_logs)) * (n_steps // n_logs)
    plt.scatter(x, fc_logs, s=7)
    # plt.plot(x, fc_logs)
    plt.show()
    print(fc_logs)


if __name__ == '__main__':
    t1 = time()
    main()
    total = time() - t1
    print("优化后的时间分析:")
    print(f"总时间: {total:.2f}s")
    print(f"get_x时间: {time_x:.2f}s ({time_x / total * 100:.1f}%)")
    print(f"get_u时间: {time_u:.2f}s ({time_u / total * 100:.1f}%)")
    print(f"get_f时间: {time_f:.2f}s ({time_f / total * 100:.1f}%)")
    print("get_u内部时间分析:")
    print(f"phi计算: {time_phi:.2f}s ({time_phi / time_u * 100:.1f}%)")
    print(f"s计算: {time_s:.2f}s ({time_s / time_u * 100:.1f}%)")
    print(f"其他操作: {time_op:.2f}s ({time_op / time_u * 100:.1f}%)")
