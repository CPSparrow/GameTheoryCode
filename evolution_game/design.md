# model(PGG) design:
每一个 PggModel 代表一个模拟，可选的参数是 noise traders 的比例 $\rho$ ,从众策略的比例 $\lambda$ 和投资的回报率 $𝑟$ 。博弈环境为 $100*100$ 的平面。

Attributes:
- `rho,l (lambda),r`: 模型的独立参数，对应上述参数
- `PolicyTable`: 每一个参与者的策略，代表设置决定X的概率(1代表合作，-1代表背叛)
- 根据噪声交易者比例 $\rho$ ,生成记录表`IsNoiseTrader`

## 运行流程
运行中，$t$代表步数
0. 根据初始策略 $P(0)=0.5$ ,生成 $X$。

1. 对于每一个 $X$ ,计算收益 $\Phi$ ($N_k$表示相邻组中合作者个数)
$$
\begin{equation}
\Phi_i=\left\{
    \begin{aligned}
    &r*N_k-c && \quad X=1\\
    &r*N_k && \quad X=-1\\
    \end{aligned}
    \right
    .
\end{equation}
$$

2. 根据 $\Phi$ 计算激活指数$s_i(t)$，其中$A=2$
$$
\begin{equation}
s_i(t)=tanh[2*(\Phi_i-A)]
\end{equation}
$$

3. 根据 $s(i)$ 进行基于收益的更新。和论文不同，这里应该记为 $U(t+1)$ ,使得最后更新的是 $P(t+1)$
![公式](/evolution_game/asset/equation_1.png)

4. 给出从众策略的公式，其中$q=2$
$$
\begin{equation}
F_i=\left\{
    \begin{aligned}
    &1-\omega_i && \quad X_i=1\\
    &\omega_i && \quad X_i=-1\\
    \end{aligned}
    \right
    .
\end{equation}
$$
其中，对于从众参与者: 
$$
\begin{equation}
\omega_i=\frac{1}{2}[1-(1-2q)X_i*\text{sgn}(\sum_{k\in{\bar{\Omega}}}X_k)]
\end{equation}
$$
> 也就是基于临近参与者更新策略。

而对于反选参与者：
$$
\begin{equation}
\omega_i=\frac{1}{2}[1+(1-2q)X_i*\text{sgn}(\sum_{i=1}^{N}X_i)]
\end{equation}
$$

5. 最后更新每一个参与者的策略
$$
P_i(t+1)=(1-\lambda)U_i+\lambda F_i
$$