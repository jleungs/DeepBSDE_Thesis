import numpy as np
import tensorflow as tf
from time import time


class DeepBSDE(tf.keras.Model):
    def __init__(self, eq):
        # initialize of tf.keras.Model
        super().__init__()
        # initialize other variables
        self.eq = eq
        self.t_space = np.linspace(0, self.eq.T, self.eq.S + 1)
        self.learning_iterations = 5000
        # optimizer for gradient descent
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                self.eq.learning_boundaries, self.eq.learning_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8)
        # initialize u(0,X_T)
        self.Y_init = tf.Variable(np.random.uniform(eq.y_init[0], eq.y_init[1], size=[1]))
        # initialize the gradient ∇u(0,X_T)
        self.Z_init = tf.Variable(np.random.uniform(-.1, .1, size=(1, eq.D)))
        # create neural networks for approximating Z_t, for all timesteps
        self.nets = [ self._create_nn() for _ in range(eq.S - 1) ]

    def _create_nn(self):
        """
        create a neural network with arcitecture:
        Input -> BN -> (Dense -> BN -> ReLU)*2 -> Dense -> BN -> Output
        """
        bn = [tf.keras.layers.BatchNormalization(
                momentum=.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5))
            for _ in range(4)]
        dense = [tf.keras.layers.Dense(
                    self.eq.D+10,
                    use_bias=False,
                    activation=None)
            for i in range(2)]
        dense.append(tf.keras.layers.Dense(
                        self.eq.D,
                        activation=None))

        net = tf.keras.Sequential()
        net.add(tf.keras.layers.Input(self.eq.D))
        net.add(bn[0])
        for i in range(len(dense) - 1):
            net.add(dense[i])
            net.add(bn[i+1])
            net.add(tf.keras.layers.ReLU())
        net.add(dense[i+1])
        net.add(bn[i+2])
        #tf.keras.utils.plot_model(net, to_file="net.png", show_shapes=True, show_layer_names=True)
        return net

    def _simulate_Y(self, X, dW):
        """
        simulate the backward process Y_N ≈ u(T, X_T) 
        """
        n_samples = X.shape[0]
        # approximate u(0,X_T)
        y = tf.ones(shape=[n_samples, 1], dtype="float64") * self.Y_init
        # approximate gradient ∇u(0,X_T)
        z = tf.ones(shape=[n_samples, 1], dtype="float64") * self.Z_init
        # approximate the backward process Y
        for i in range(self.eq.S - 1):
            t = self.t_space[i]
            # Euler-Maruyama approximation of Y at t_{i+1}
            y = y - self.eq.backward_f_tf(t, X[:,:,i], y, z)*self.eq.dt + \
                tf.reduce_sum(z * dW[:,:,i], axis=1, keepdims=True)
            # approximate Z at t_{i+1}
            z = self.nets[i](X[:,:,i+1])/self.eq.D
        # compute Y at terminal time T
        y = y - self.eq.backward_f_tf(self.t_space[-1], X[:,:,-1], y, z)*self.eq.dt + \
            tf.reduce_sum(z * dW[:,:,-1], axis=1, keepdims=True)

        return y

    def _loss_fn(self, X, dW):
        """
        function to minimize, MSE of Y_T and g(X_T)
        """
        # simulate the backward process with forward process and training data
        y_pred = self._simulate_Y(X, dW)
        # evaluate g(X_T)
        y = self.eq.backward_g_tf(self.eq.T, X)
        # Mean squared error
        loss = tf.reduce_mean(tf.square(y - y_pred))

        return loss

    def _compute_grad(self, X, dW):
        """
        Gradient of the loss function w.r.t. the network parameters theta
        """
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            loss = self._loss_fn(X, dW)
        grad = tape.gradient(loss, self.trainable_variables)

        return loss, grad

    def train(self):
        """
        train the neural network, i.e. minimize the loss function
        """
        s_time = time()
        # simulate the forward process
        X, dW = self.eq.forward_Xt(256)
        # stochastic gradient descent
        for i in range(self.learning_iterations+1):
            loss = self._train_step(X,dW)
            if i%100 == 0:
                print(f"i: {i:4} Y0: {self.Y_init.numpy()[0]:6.3f} Loss: {loss.numpy():5.3f} "\
                        f"Time: {time() - s_time:5.1f}")
            X, dW = self.eq.forward_Xt(64)

    @tf.function
    def _train_step(self, X, dW):
        loss, grad = self._compute_grad(X, dW)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return loss

