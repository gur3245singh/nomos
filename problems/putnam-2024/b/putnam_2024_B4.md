Let $n$ be a positive integer. Set $a_{n,0} = 1$. For $k \ge 0$, choose an integer $m_{n,k}$ uniformly at random from the set $\{1,\ldots,n\}$, and let
$$a_{n,k+1} = \begin{cases}
a_{n,k} + 1, & \text{if } m_{n,k} > a_{n,k}; \\
a_{n,k}, & \text{if } m_{n,k} = a_{n,k}; \\
a_{n,k} - 1, & \text{if } m_{n,k} < a_{n,k}.
\end{cases}$$
Let $E(n)$ be the expected value of $a_{n,n}$. Determine $\lim_{n \to \infty} E(n)/n$.
