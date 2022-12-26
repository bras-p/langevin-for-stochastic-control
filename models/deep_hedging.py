import tensorflow as tf
from .base import ModelBuilder




class DeepHedging(ModelBuilder):
    def __init__(
        self,
        T=50./365.,
        N_euler=50,
        dim=1,
        multiple_ctrls=False,
        ctrl_hidden_units=[32,32],
        ell=lambda x:x,
        a=0.3*tf.ones((1,)),
        b=0.04*tf.ones((1,)),
        sigma=2.*tf.ones((1,)),
        rho = 1.*tf.ones((1,)),
        K=1.*tf.ones((1,)),
        T_COST=0.01,
        epsilon=1e-5,
    ):
        super().__init__(T, N_euler, dim)
        self.multiple_ctrls = multiple_ctrls
        self.ctrl_hidden_units = ctrl_hidden_units
        self.ell = ell
        self.a, self.b, self.sigma, self.rho = a, b, sigma, rho
        self.K = K
        self.T_COST = T_COST
        self.epsilon = epsilon


    def getCtrlModel(self):
        ctrl_model = tf.keras.Sequential([tf.keras.layers.Dense(j, activation='relu') for j in self.ctrl_hidden_units])
        ctrl_model.add(tf.keras.layers.Dense(2*self.dim, activation='relu'))
        return ctrl_model

    def L(self,t,v):
        return (1./self.a)*(v-self.b)*(1.-tf.exp(-self.a*(self.T-t))) + self.b*(self.T-t)


    def getModel(self):
        S1_input = tf.keras.Input(shape=(self.dim,))
        V_input = tf.keras.Input(shape=(self.dim,))
        int_V = tf.zeros(tf.shape(V_input))
        batch_size = tf.shape(S1_input)[0]
        benefits = tf.zeros((batch_size, 1))
        transaction_cost = tf.zeros((batch_size, 1))

        S1 = 1.*S1_input
        V = 1.*V_input
        S2 = int_V + self.L(0.,V)
        delta = tf.zeros((batch_size, 2*self.dim))

        X_tab = tf.concat([S1, V, delta], axis=1)

        if not self.multiple_ctrls:
            ctrl_model = self.getCtrlModel()

        for k in range(self.N_euler):
            S1_old = 1.*S1
            S2_old = 1.*S2
            delta_old = 1.*delta
            if self.multiple_ctrls:
                delta = self.getCtrlModel()(tf.concat([tf.math.log(S1), V, delta_old], axis=1))
            else:
                delta = ctrl_model(tf.concat([tf.math.log(S1), V, delta_old, k*self.h*tf.ones((batch_size,1))], axis=1))
            transaction_cost = transaction_cost + tf.reduce_sum(
                self.T_COST*tf.math.abs(delta-delta_old)*tf.concat([S1, S2], axis=1),
                axis=1, keepdims=True)

            B = tf.random.normal(tf.shape(S1))
            W = self.rho*B + tf.sqrt(1.-self.rho**2)*tf.random.normal(tf.shape(S1))
            S1 = tf.clip_by_value(S1 + tf.sqrt(self.h)*tf.sqrt(V)*S1*B, self.epsilon, 100./self.epsilon)
            V = tf.keras.activations.relu(
                V + self.h*self.a*(self.b-V) + self.sigma*tf.sqrt(self.h)*tf.sqrt(V)*W)
            int_V = int_V + self.h*V
            S2 = int_V + self.L((k+1)*self.h,V)
            benefits = benefits + tf.reduce_sum(
                delta * tf.concat([S1-S1_old, S2-S2_old], axis=1),
                axis=1, keepdims=True)

            X_tab = tf.concat([X_tab, tf.concat([S1,V,delta], axis=1)], axis=1)
        
        # add the last term of the transaction cost with delta_n = 0
        transaction_cost = transaction_cost + tf.reduce_sum(
            self.T_COST*tf.math.abs(delta)*tf.concat([S1,S2], axis=1),
            axis=1, keepdims=True)

        Z = tf.reduce_sum(tf.keras.activations.relu(S1-self.K), axis=1, keepdims=True)
        w = tf.keras.layers.Dense(1, use_bias=False)(tf.ones((batch_size, 1)))
        J = w + self.ell(Z - benefits + transaction_cost - w)
        # for exponential utility: no need of w and ell and
        # J = tf.exp(lmbd*(Z - benefits + transaction_cost))

        model = tf.keras.Model(inputs=[S1_input, V_input], outputs=tf.concat([J,X_tab],axis=1))
        return model

