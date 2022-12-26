import numpy as np
import tensorflow as tf

from models.oil_drilling import OilDrilling
from experiment import Experiment
from data.constant import ConstantLoader
from optimizers.ladam import LAdam, LayerLAdam
from optimizers.lrmsprop import LRMSprop, LayerLRMSprop
from optimizers.ladadelta import LAdadelta, LayerLAdadelta


batch_size = 512
N_train=5*batch_size
N_test=25*batch_size

N_euler = 50


dim = 1
X0 = 1.*tf.ones((dim))
model_builder = OilDrilling(
    T=1,
    N_euler=N_euler,
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
)

exp_name = 'oil_drilling'
base = './'

# first get a dummy model to see the number of layers
model = model_builder.getModel()

EPOCHS = 100
lr_0 = 0.05
epoch_change = 80
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[epoch_change*int(np.ceil(N_train/batch_size))], values=[lr_0, lr_0/10])
sigma_0 = 5e-3
sigma_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[epoch_change*int(np.ceil(N_train/batch_size))], values=[sigma_0, 0.])

optimizers = [
    LAdam(learning_rate=lr_schedule, sigma=0.),
    LAdam(learning_rate=lr_schedule, sigma=sigma_schedule),
    LayerLAdam(learning_rate=lr_schedule, sigma=sigma_schedule, langevin_layers=range(int(0.3*len(model.layers)))),
]


dataloader = ConstantLoader(
    X0 = X0,
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
experiment.run_experiment(initializer='zero')
experiment.plot()

experiment.plot_traj(opt_index=1)


experiment.save_data(exp_name +'_N{}'.format(N_euler))
experiment.save_traj(exp_name)

