import numpy as np
import tensorflow as tf

from libs import settings


class EuropeanCallSpread:
    def __init__(self):
        # 21.299 | discounted 21.730
        self.D = 100                        # number of dimensions
        self.S = 15                         # number of time intervals
        self.T = 0.5                        # final time
        self.sigma = 0.2                    # volatility
        self.y_init = [17,18]               # initial guess for put price
        self.x_init = 100                   # stock price at time 0
        self.r1 = 0.04                      # lending interest rate
        self.r2 = 0.06                      # borrowing interest rate
        self.K1 = 120                       # strike price
        self.K2 = 150                       # strike price
        self.mu = self.r2                   # risk-neutral, drift equal rate
        self.dt = self.T/self.S             # time step size
        self.sqrt_dt = np.sqrt(self.dt)     # time step size
        self.learning_boundaries = [2500]
        self.learning_values = [7e-3, 5e-3]

    def forward_Xt(self, n_paths):
        """
        Simulate the forward process X_t
        """
        # generate random variables ~ N(0,dt)
        dW = np.random.standard_normal((n_paths, self.D, self.S)).astype(settings.DTYPE) * self.sqrt_dt
        # initialize the forward process X with initial value x
        X = np.zeros((n_paths, self.D, self.S+1), dtype=settings.DTYPE)
        X[:,:,0] = np.ones((n_paths, self.D)) * self.x_init
        # simulate the forward process
        for i in range(self.S):
            X[:,:,i+1] = X[:,:,i] + X[:,:,i]*(self.mu*self.dt + self.sigma*dW[:,:,i])
        return X, dW

    def backward_g_tf(self, t, x):
        """
        Payoff of the option, terminal condition of the PDE
        """
        # x[:,:,-1] to get the value of x at terminal time
        obj = tf.reduce_max(x[:,:,-1], axis=1)
        return tf.maximum(obj - self.K1, 0) - 2*tf.maximum(obj - self.K2, 0)

    def backward_f_tf(self, t, x, y, z):
        """
        Generator function of the PDE
        """
        term = tf.reduce_sum(z, axis=1, keepdims=True)/self.sigma
        return -self.r1*tf.maximum(y - term, 0) + self.r2*tf.minimum(y - term, 0) - term*self.mu


class EuropeanCallDiffRate:
    def __init__(self):
        # 10.99
        self.D = 1                          # number of dimensions
        self.S = 10                         # number of time intervals
        self.T = 1.0                        # final time
        self.sigma = 0.2                    # volatility
        self.y_init = [1,3]                 # initial guess for put price
        self.x_init = 100                   # stock price at time 0
        self.r1 = 0.03                      # lending interest rate
        self.r2 = 0.06                      # borrowing interest rate
        self.K = 100                        # strike price
        self.mu = self.r2                   # risk-neutral, drift equal rate
        self.dt = self.T/self.S             # time step size
        self.sqrt_dt = np.sqrt(self.dt)     # time step size
        self.learning_boundaries = [2500]
        self.learning_values = [5e-3, 5e-3]

    def forward_Xt(self, n_paths):
        """
        Simulate the forward process X_t
        """
        # generate random variables ~ N(0,dt)
        dW = np.random.standard_normal((n_paths, self.D, self.S)).astype(settings.DTYPE) * self.sqrt_dt
        # initialize the forward process X with initial value x
        X = np.zeros((n_paths, self.D, self.S+1), dtype=settings.DTYPE)
        X[:,:,0] = np.ones((n_paths, self.D)) * self.x_init
        # simulate the forward process
        for i in range(self.S):
            X[:,:,i+1] = X[:,:,i] + X[:,:,i]*(self.mu*self.dt + self.sigma*dW[:,:,i])
        return X, dW

    def backward_g_tf(self, t, x):
        """
        Payoff of the option, terminal condition of the PDE
        """
        # x[:,:,-1] to get the value of x at terminal time
        obj = tf.reduce_max(x[:,:,-1], axis=1)
        #return tf.maximum(obj - self.K, 0)
        return tf.maximum(self.K - obj, 0)

    def backward_f_tf(self, t, x, y, z):
        """
        Generator function of the PDE
        """
        term = tf.reduce_sum(z, axis=1, keepdims=True)/self.sigma
        return -self.r1*tf.maximum(y - term, 0) + self.r2*tf.minimum(y - term, 0) - term*self.mu


class EuropeanPut:
    def __init__(self):
        # Black-Scholes Price: 5.166
        self.D = 1                      # number of dimensions
        self.S = 10                     # number of time intervals
        self.T = 1.0                    # final time
        self.sigma = 0.2                # volatility
        self.y_init = [1,3]             # initial guess for put price
        self.x_init = 100               # stock price at time 0
        self.r = 0.06                   # risk-free interest rate
        self.K = 100                    # strike price
        self.mu = self.r                # risk-neutral, drift equal rate
        self.dt = self.T/self.S         # time step size
        self.sqrt_dt = np.sqrt(self.dt) # for variance of dW
        self.learning_boundaries = [2500]
        self.learning_values = [5e-3, 5e-3]

    def forward_Xt(self, n_paths):
        """
        Simulate the forward process X_t
        """
        # generate random variables ~ N(0,dt)
        dW = np.random.standard_normal((n_paths, self.D, self.S)).astype(settings.DTYPE) * self.sqrt_dt
        # initialize the forward process X with initial value x
        X = np.zeros((n_paths, self.D, self.S+1), dtype=settings.DTYPE)
        X[:,:,0] = np.ones((n_paths, self.D)) * self.x_init
        # simulate the forward process
        for i in range(self.S):
            X[:,:,i+1] = X[:,:,i] + X[:,:,i]*(self.mu*self.dt + self.sigma*dW[:,:,i])
        return X, dW

    def backward_g_tf(self, t, x):
        """
        Payoff of the option, terminal condition of the PDE
        """
        # the discounted price at terminal time
        return tf.maximum(self.K - x[:,:,-1], 0)

    def backward_f_tf(self, t, x, y, z):
        """
        Generator function of the PDE
        """
        term = tf.reduce_sum(z, axis=1, keepdims=True)/self.sigma
        return -self.r*(y - term) - term*self.mu
