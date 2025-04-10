import numpy as np
import random


class PggModel:
    def __init__(self, rho: float, l: float, r: float, size: int):
        """
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
    def get_phi(X, r) -> np.ndarray:
        """
        get Phi based on choices table X
        :param X: choices table of candidates
        :param r: reward
        :return: value of choices named phi,shape:(size,size)
        """
        phi = np.zeros_like(X)
        for row_index, row in enumerate(X):
            for col_index, item in enumerate(row):
                phi[row_index][col_index] += item * (r - 1)
                if row_index > 0:
                    phi[row_index - 1][col_index] += item * r
                if col_index > 0:
                    phi[row_index][col_index - 1] += item * r
                if row_index < X.shape[0] - 1:
                    phi[row_index + 1][col_index] += item * r
                if col_index < X.shape[1] - 1:
                    phi[row_index][col_index + 1] += item * r
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
            phi.astype(np.float64)
            phi -= phi.mean()
            a, b = phi.min(), phi.max()
            s = np.tanh(2 * phi / max(abs(b), abs(a)))
        return s
    
    def get_u(self, X) -> np.ndarray:
        """
        get policy update table U based on choices table X
        :param X:
        :return: policy update table of U,shape:(size,size)
        """
        phi = PggModel.get_phi(X, self.r)
        s_tabel = PggModel.get_s(phi)
        u = np.zeros_like(X).astype(np.float64)
        
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                flag = X[row][col] if X[row][col] == 1 else -1
                p = self.PolicyTable[row][col]
                s = s_tabel[row][col]
                u[row][col] = p + flag * (1 - p) * s if s >= 0 else p * (1 - flag * s)
        return u
    
    @staticmethod
    def get_w(X) -> float:
        pass
    
    def get_f(self, X) -> np.ndarray:
        """
        get policy update table F based on surrounding choices table X
        :param X: choices table of candidates
        :return: policy update table of F,shape:(size,size)
        """
        f = np.zeros_like(X).astype(np.float64)
        converse_weight = PggModel.get_w(X)
        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                f[row][col] = X[row][col]
        return f
    
    def forward(self):
        """
        X denotes choice made based on p
        note that X in this implement is 0 or 1, not 1 or -1 in paper
        """
        X = (np.random.rand(*self.PolicyTable.shape) < self.PolicyTable).astype(int)
        u = self.get_u(X)
        f = self.get_f(X)
        
        print(X, f, sep='\n')
        return X


if __name__ == '__main__':
    model = PggModel(rho=0.9, l=0.5, r=3.5, size=4)
    model.forward()
