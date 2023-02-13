import tensorflow as tf
from .base import ModelBuilder


class Fishing(ModelBuilder):
    def __init__(
        self,
        T=2.,
        N_euler=10,
        dim=5,
        multiple_ctrls = False,
        r=2.*tf.ones(5),
        kappa=tf.ones((5,5)),
        X_d = tf.ones(5),
        u_m=0.1,
        u_M=1.,
        alpha=0.01*tf.ones(5),
        beta=0.1,
        sigma=0.1*tf.eye(5),
        ctrl_hidden_units = [32,32],
    ):
        super().__init__(T, N_euler, dim)
        self.r = r
        self.kappa_t = tf.transpose(kappa)
        self.X_d = X_d
        self.sigma_t = tf.transpose(sigma)
        self.u_m, self.u_M = u_m, u_M
        self.alpha, self.beta = alpha, beta
        self.ctrl_hidden_units = ctrl_hidden_units
        self.multiple_ctrls = multiple_ctrls

    
    def getCtrlModel(self):
        ctrl_model = tf.keras.Sequential([tf.keras.layers.Dense(j, activation='relu') for j in self.ctrl_hidden_units])
        ctrl_model.add(tf.keras.layers.Dense(self.dim, activation='sigmoid'))
        return ctrl_model


    def getModel(self):
        X_input = tf.keras.Input(shape=(self.dim,))
        batch_size = tf.shape(X_input)[0]
        if not self.multiple_ctrls:
            ctrl_model = self.getCtrlModel()
        J = tf.zeros((batch_size, 1))
        old_u = tf.zeros(tf.shape(X_input))

        X = 1.*X_input
        X_tab = tf.concat([X,tf.zeros(tf.shape(X))], axis=1)
        for k in range(self.N_euler):
            drift = self.h * X * (self.r - tf.matmul(X, self.kappa_t))
            noise = tf.sqrt(self.h) * X * tf.matmul(tf.random.normal(tf.shape(X)), self.sigma_t)
            if self.multiple_ctrls:
                ctrl_value = self.getCtrlModel()(X)
            else:
                ctrl_value = ctrl_model(tf.concat([X, k*self.h*tf.ones((batch_size, 1))], axis=1))
            u = self.u_m + (self.u_M - self.u_m)*ctrl_value
            X = X + drift + noise - self.h*X*u
            J = J + self.h * tf.reduce_sum(tf.square(X-self.X_d) - u*self.alpha  + (self.beta/self.h)*tf.square(u-old_u), axis=-1, keepdims=True)
            old_u = 1.*u
            X_tab = tf.concat([X_tab, tf.concat([X,u], axis=1)], axis=1)

        model = tf.keras.Model(inputs=X_input, outputs=tf.concat([J,X_tab], axis=1))

        return model





