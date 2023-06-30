import tensorflow as tf
from .base import ModelBuilder


class OilDrilling(ModelBuilder):
    def __init__(
        self,
        T=1.,
        N_euler=10,
        dim=1,
        multiple_ctrls=False,
        mu=0.01,
        sigma=0.2,
        rho=0.01,
        epsilon=0.,
        xi_s=0.005,
        K0=5.,
        xi_e=0.01,
        qS=10.,
        U= (lambda x: x),
    ):
        super().__init__(T, N_euler, dim)
        self.multiple_ctrls = multiple_ctrls
        self.mu, self.sigma, self.rho, self.epsilon = mu, sigma, rho, epsilon
        self.xi_s, self.K0, self.xi_e, self.qS = xi_s, K0, xi_e, qS
        self.U = U
        self.c_s = (lambda x: tf.exp(xi_s*x)-1.)
        self.c_e = (lambda x: tf.exp(xi_e*x))

    
    def getCtrlModel(self):
        ctrl_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        ])
        return ctrl_model


    def apply_constraints(self, q_t, S_t):
        qv_t, qs_t, qvs_t = q_t[:,0:1], q_t[:,1:2], q_t[:,2:3]
        qv_t = max_at(qv_t, self.K0)
        qs_t = max_at(qs_t, self.K0-qv_t) # to satisfy qv_t + qs_t <= K0
        qvs_t = max_at(qvs_t, self.qS)
        qvs_t = max_at(qvs_t, S_t/self.h)
        q = tf.concat([qv_t, qs_t, qvs_t], axis=1)
        return q


    def getModel(self):
        P_input = tf.keras.Input(shape=(self.dim,))
        batch_size = tf.shape(P_input)[0]
        E_t = tf.zeros((batch_size, 1))
        S_t = tf.zeros((batch_size, 1))
        J = tf.zeros((batch_size, 1))

        P_t = 1.*P_input
        if not self.multiple_ctrls:
            ctrl_model = self.getCtrlModel()
        else:
            ctrl_models = []

        X_tab = tf.concat([P_t, E_t, S_t], axis=1)

        for k in range(self.N_euler):
            if self.multiple_ctrls:
                new_ctrl_model = self.getCtrlModel()
                ctrl_models.append(new_ctrl_model)
                q_t = new_ctrl_model()(tf.concat([P_t, E_t, S_t], axis=1))
            else:
                q_t = ctrl_model(tf.concat([P_t, E_t, S_t, k*self.h*tf.ones((batch_size, 1))], axis=1))
            q_t = self.apply_constraints(q_t, S_t)
            qv_t, qs_t, qvs_t = q_t[:,0:1], q_t[:,1:2], q_t[:,2:3]
            benefits = P_t*(qv_t + (1.-self.epsilon)*qvs_t)
            extraction_cost = (qv_t+qs_t)*self.c_e(E_t)
            storage_cost = self.c_s(S_t)
            E_t = E_t + self.h*(qv_t + qs_t)
            S_t = S_t + self.h*(qs_t - qvs_t)
            J = J - self.h*tf.exp(-self.rho*k*self.h) * self.U(benefits - extraction_cost - storage_cost)
            P_t = P_t*tf.exp((self.mu-self.sigma**2/2.)*self.h + self.sigma*tf.sqrt(self.h)*tf.random.normal((batch_size, self.dim)))

            X_tab = tf.concat([X_tab, tf.concat([P_t, E_t, S_t], axis=1)], axis=1)
        
        J = J - tf.exp(-self.rho*self.N_euler*self.h) * self.U((1.-self.epsilon)*P_t*S_t) # at the end we sell all of our stock

        model = tf.keras.Model(inputs=P_input, outputs=tf.concat([J,X_tab], axis=1))

        if not self.multiple_ctrls:
            model.ctrl_model = ctrl_model
        else:
            model.ctrl_models = ctrl_models

        return model


def max_at(q,m): # clips q at m
    return m - tf.keras.activations.relu(-q+m)