# DeepBSDE Thesis
Using deep learning to solve the semilinear parabolic partial differential equation (PDE)

$$
\begin{equation}
\begin{aligned}
    \frac{\partial u}{\partial t}(t,x) &+ \frac{1}{2} \text{Tr} \left(\sigma \sigma^T(t,x) (\text{Hess}_x u)(t,x) \right) + \nabla u(t,x) \mu(t,x)\\
    &+ f(t,x,u(t,x),\sigma^T(t,x)\nabla u(t,x)) = 0, \qquad u(T,x) = g(x),
\end{aligned}
\end{equation}
$$

using its backward stochastic differential equation (BSDE) formulation 

$$
\begin{equation}
\begin{aligned}
    u(t,X_t) - u(0,X_0) = &-\int_0^t f\bigg(s,X_s,u(s,X_s),\sigma^T(s,X_s)\nabla u(s,X_s) \bigg)ds\\
    &+ \int_0^t \bigg(\nabla u(s,X_s) \bigg)^T \sigma(s,X_s)dW_s,
\end{aligned}
\end{equation}
$$

through nonlinear Feynman-Kac with the following relation, that holds true *almost surely*

$$
\begin{equation}
    Y_t = u(t,X_t), \quad Z_t = \sigma^T (t,X_t) \nabla u(t,X_t), \qquad \forall t \in [0,T].
\end{equation}
$$

\
Three problems are implemented, `EuropeanPut`, `EuropeanCallDiffRate` and `EuropeanCallSpread`. To run, use the class name as an argument:
```
python3 deepbsde.py EuropeanCallSpread
```

Using `EuropeanCallSpread` for pricing European call spread options with different interest rates for borrowing and lending.

## Reference
TODO
