# model(PGG) design:
每一个 PggModel 代表一个模拟，可选的参数是 noise traders 的比例 $\rho$ ,从众策略的比例 $\lambda$ 和投资的回报率 $𝑟$ 。
Attributes:
- X:每一个参与者的策略(1代表合作，-1代表背叛)

## 运行流程
每一次
