import numpy as np
import tensorflow as tf

from models.deep_hedging import DeepHedging
from experiment import Experiment
from data.constant import ConstantMultipleLoader
from optimizers.ladam import LAdam, LayerLAdam
from optimizers.lrmsprop import LRMSprop, LayerLRMSprop
from optimizers.ladadelta import LAdadelta, LayerLAdadelta


batch_size = 512
N_train=5*batch_size
N_test=25*batch_size


T = 1.
N_euler = 50
dim = 5

s0 = 1.
v0 = 0.1
# v0 = 0.04

alpha = 0.9
def ell(x):
    return (1./(1.-alpha))*tf.keras.activations.relu(x)


model_builder = DeepHedging(
    T=T,
    N_euler=N_euler,
    dim=dim,
    multiple_ctrls=False,
    ctrl_hidden_units=[32,32],
    ell=ell,
    a = 1.*tf.ones((dim,)),
    b = 0.04*tf.ones((dim,)),
    sigma = 2.*tf.ones((dim,)),
    rho = -0.7*tf.ones((dim,)),
    K = s0*tf.ones((dim,)),
    T_COST=5e-4,
)


exp_name = 'deep_hedging'
base = './results/'

# first get a dummy model to see the number of layers
model = model_builder.getModel()

EPOCHS = 200
lr_0 = 2e-3
epoch_change = 180
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[epoch_change*int(np.ceil(N_train/batch_size))], values=[lr_0, lr_0/10])
sigma_0 = 2e-3
sigma_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[epoch_change*int(np.ceil(N_train/batch_size))], values=[sigma_0, 0.])

optimizers = [
    LAdam(learning_rate=lr_schedule, sigma=0.),
    LAdam(learning_rate=lr_schedule, sigma=sigma_schedule),
    # LayerLAdam(learning_rate=lr_schedule, sigma=sigma_schedule, langevin_layers=range(int(0.3*len(model.layers)))),
]


dataloader = ConstantMultipleLoader(
    X0_list = [s0*tf.ones((dim,)), v0*tf.ones((dim,))],
    N_train = N_train,
    N_test = N_test,
    batch_size = batch_size,
)

experiment = Experiment(
    model_builder=model_builder,
    dataloader=dataloader,
    EPOCHS=EPOCHS,
    optimizers=optimizers,
    base=base
)


experiment.load_data()
experiment.run_experiment(initializer=None)
experiment.plot()

experiment.plot_traj(opt_index=1)

experiment.save_data(dir_name=exp_name +'_N{}'.format(N_euler))
experiment.save_traj(dir_name=exp_name +'_N{}'.format(N_euler))


