import numpy as np
import random


class PggModel:
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
    
    def forward(self):
        pass


model = PggModel(rho=0.9, l=0.5, r=4.5, size=5)
print(model.IsConverse)
