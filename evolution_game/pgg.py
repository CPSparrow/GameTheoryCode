import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from matplotlib.colors import ListedColormap
from time import time

time_total, time_x, time_u, time_f = 0, 0, 0, 0
time_phi, time_s, time_op = 0, 0, 0


class PggModel:
    def __init__(self, rho: float, l: float, r: float, size: int):
        """
        for 'IsConverse' , value 1 denotes conserve choice and 0 denotes normal action
        :param rho: noise traders(反选者) 的比例
        :param l: 历史回报函数和从众函数的权重。l越大越倾向于从众
        :param r: 投资的回报率
        :param size: 场地大小
        """
        self.rho = rho
        self.l = l
        self.r = r
        self.size = size
        self.PolicyTable, self.IsConverse = PggModel.DataInit(size, rho)
    
    @staticmethod
    def DataInit(size: int, rho: float) -> tuple[np.ndarray, np.ndarray]:
        """
        :param size:场地大小
        :param rho: noise traders(反选者) 的比例
        :return: policy,converse.分别代表起始策略和反选者名单
        """
        policy = np.full((size, size), 0.5)
        n_ones = int(size * size * (1 - rho))
        n_zeros = size * size - n_ones
        converse = [1] * n_ones + [0] * n_zeros
        random.shuffle(converse)
        converse = np.array(converse, dtype=int)
        converse.resize(size, size)
        
        return policy, converse
    
    @staticmethod
    def get_n(X) -> np.ndarray:
        """
        return cooperation rate based on choice tabel X
        :param X: choice tabel X
        :return: cooperation rate,shape:(size,size)
        """
        n = np.zeros_like(X, dtype=np.float64)
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                n_cooperators, n_individuals = X[row][col], 1
                if row > 0:
                    n_cooperators += X[row - 1][col]
                    n_individuals += 1
                if col > 0:
                    n_cooperators += X[row][col - 1]
                    n_individuals += 1
                if row < X.shape[0] - 1:
                    n_cooperators += X[row + 1][col]
                    n_individuals += 1
                if col < X.shape[1] - 1:
                    n_cooperators += X[row][col + 1]
                    n_individuals += 1
                n[row][col] = n_cooperators / n_individuals
        return n
    
    @staticmethod
    def get_phi(X, r) -> np.ndarray:
        """
        get Phi based on choices table X
        :param X: choices table of candidates
        :param r: reward
        :return: value of choices named phi,shape:(size,size)
        """
        phi = np.zeros_like(X, dtype=np.float64)
        n = PggModel.get_n(X)
        
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                phi[row][col] = n[row][col] * (r - 1 * X[row][col])
                if row > 0:
                    phi[row][col] += n[row - 1][col] * r
                if col > 0:
                    phi[row][col] += n[row][col - 1] * r
                if row < X.shape[0] - 1:
                    phi[row][col] += n[row + 1][col] * r
                if col < X.shape[1] - 1:
                    phi[row][col] += n[row][col + 1] * r
        
        return phi
    
    @staticmethod
    def get_s(phi) -> np.ndarray:
        """
        get S based on choices table phi
        :param phi: value of choices table of candidates
        :return: activate value of shape:(size,size)
        """
        if True:
            s = np.tanh(2 * (phi - 2))
        else:
            phi = phi.astype(np.float64)
            phi -= phi.mean()
            a, b = phi.min(), phi.max()
            s = np.tanh(2 * phi / max(abs(b), abs(a)))
        return s
    
    def get_x(self) -> np.ndarray:
        """
        return the choice tabel X
        :return: choice tabel X,shape:(size,size)
        """
        return (
                np.random.rand(*self.PolicyTable.shape) < self.PolicyTable
        ).astype(int)
    
    def get_u(self, X) -> np.ndarray:
        """
        get policy update table U based on reward s
        :param X:
        :return: policy update table of U,shape:(size,size)
        """
        a = time()
        phi = PggModel.get_phi(X, self.r)
        b = time()
        s_tabel = PggModel.get_s(phi)
        c = time()
        
        u = np.zeros_like(X, dtype=np.float64)
        
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                x, s, p = X[row][col], s_tabel[row][col], self.PolicyTable[row][col]
                
                if x == 1:
                    if s >= 0:
                        u[row][col] = p + (1 - p) * s
                    else:
                        u[row][col] = p + p * s
                else:
                    if s >= 0:
                        u[row][col] = p - p * s
                    else:
                        u[row][col] = p - (1 - p) * s
                if u[row][col] > 1:
                    u[row][col] = 1
                if u[row][col] < 0:
                    u[row][col] = 0
        d = time()
        
        global time_phi, time_s, time_op
        time_phi += b - a
        time_s += c - b
        time_op += d - c
        return u
    
    def get_f(self, X) -> np.ndarray:
        """
        get policy update table F based on surrounding choices table X
        :param X: choices table of candidates
        :return: policy update table of F,shape: (size,size)
        """
        f = np.zeros_like(X, dtype=np.float64)
        q = 0.01
        
        n_candidates = X.shape[0] * X.shape[1]
        converse_flag = X.sum() * 2 - n_candidates
        converse_flag = 0 if converse_flag == 0 else 1 if converse_flag > 0 else -1
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                x = X[row][col] if X[row][col] == 1 else -1
                if self.IsConverse[row][col] == 1:
                    f[row][col] = 0.5 * \
                                  (1 + (1 - 2 * q) * x * converse_flag)
                else:
                    candidate_flag, n_neighbors = 0, 0
                    if row > 0:
                        candidate_flag += X[row - 1][col]
                        n_neighbors += 1
                    if col > 0:
                        candidate_flag += X[row][col - 1]
                        n_neighbors += 1
                    if row < X.shape[0] - 1:
                        candidate_flag += X[row + 1][col]
                        n_neighbors += 1
                    if col < X.shape[1] - 1:
                        candidate_flag += X[row][col + 1]
                        n_neighbors += 1
                    candidate_flag = candidate_flag * 2 - n_neighbors
                    candidate_flag = 0 if candidate_flag == 0 else 1 if candidate_flag > 0 else -1
                    f[row][col] = 0.5 * \
                                  (1 + (1 - 2 * q) * x * candidate_flag)
        
        for row in range(f.shape[0]):
            for col in range(f.shape[1]):
                f[row][col] = 1 - f[row][col] if X[row][col] == 1 else f[row][col]
        return f
    
    def forward(self):
        """
        X denotes choice made based on p
        note that X in this implement is 0 or 1, not 1 or -1 in paper
        """
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
        
        fc = X.sum() / (X.shape[0] * X.shape[1])
        return fc
    
    def train(self, n_steps, n_logs) -> np.ndarray:
        """
        train the gpp model for n_steps steps,will return the fc in a np array
        :param n_steps: number of steps to train
        :param n_logs: number of logs to save
        :return: return fc list in a np array
        """
        log_steps = n_steps // n_logs
        if not isinstance(n_steps, int):
            n_steps = int(n_steps)
        if not isinstance(log_steps, int):
            log_steps = int(log_steps)
        
        fc_list = list()
        for step in tqdm(range(n_steps), ncols=80):
            fc = self.forward()
            if step % log_steps == 0:
                fc_list.append(fc)
        
        return np.array(fc_list, dtype=np.float64)


def main():
    n_steps, n_logs = 1e4, 200
    
    model = PggModel(rho=0.93, l=0.5, r=4.5, size=100)
    fc_logs = model.train(n_steps=n_steps, n_logs=n_logs)
    
    plt.figure(dpi=200)
    x = np.array(list(range(len(fc_logs)))) * (n_steps // n_logs)
    plt.scatter(x, fc_logs, s=7)
    plt.show()


def time_analyse():
    t1 = time()
    n_steps, n_logs = 250, 50
    
    model = PggModel(rho=0.93, l=0.5, r=4.5, size=100)
    fc_logs = model.train(n_steps=n_steps, n_logs=n_logs)
    
    plt.figure(dpi=200)
    x = np.array(list(range(len(fc_logs)))) * (n_steps // n_logs)
    plt.scatter(x, fc_logs, s=7)
    plt.show()
    total = time() - t1
    
    print("time analyse:")
    print(f"total time: {total:.2f}s")
    print(f"time x:{time_x:5.2f},rate:{time_x / total * 100:6.2f}%")
    print(f"time u:{time_u:5.2f},rate:{time_u / total * 100:6.2f}%")
    print(f"time f:{time_f:5.2f},rate:{time_f / total * 100:6.2f}%")
    
    print("u analyse:")
    print(f"u total: {time_u:.2f}s")
    print(f"time phi:{time_phi:5.2f},rate:{time_phi / time_u * 100:6.2f}%,in total:{time_phi / total * 100:6.2f}%")
    print(f"time  s :{time_s:5.2f},rate:{time_s / time_u * 100:6.2f}%,in total:{time_s / total * 100:6.2f}%")
    print(f"time op :{time_op:5.2f},rate:{time_op / time_u * 100:6.2f}%,in total:{time_op / total * 100:6.2f}%")
    # model = PggModel(rho=0.9, l=0.5, r=4.2, size=100)
    # print(model.train(n_steps=10, n_logs=10))


if __name__ == '__main__':
    tag = 2
    if tag == 1:
        main()
    elif tag == 2:
        time_analyse()
