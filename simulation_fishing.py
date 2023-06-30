import numpy as np
import tensorflow as tf

from models.fishing import Fishing
from experiment import Experiment
from data.from_map import DataLoaderFromMap
from data.constant import ConstantLoader
from langevin_optimizers.ladam import LAdam, LayerLAdam
from langevin_optimizers.lrmsprop import LRMSprop, LayerLRMSprop
from langevin_optimizers.ladadelta import LAdadelta, LayerLAdadelta


batch_size = 512
N_train=5*batch_size
N_test=25*batch_size

N_euler = 50
dim = 5
def get_X0(batch_size):
    return tf.clip_by_value(
        tf.random.normal(shape=(batch_size, dim), mean=1., stddev=0.5),
        clip_value_min=0.2, clip_value_max=2.
    )

model_builder = Fishing(
    T=2.,
    N_euler=N_euler,
    dim=dim,
    multiple_ctrls = False,
    r=2.*tf.ones(dim),
    kappa = tf.constant([
        [1.2, -0.1, 0., 0., -0.1],
        [0.2, 1.2, 0., 0., -0.1],
        [0., 0.2, 1.2, -0.1, 0.],
        [0., 0., 0.1, 1.2, 0.],
        [0.1, 0.1, 0., 0., 1.2]
    ]),
    X_d = tf.ones(dim),
    u_m=0.1,
    u_M=1.,
    alpha=0.01*tf.ones(dim),
    beta=0.1,
    sigma=0.1*tf.eye(dim),
    ctrl_hidden_units = [128,128],
)

base = './results/'
exp_name = 'fishing'

# first get a dummy model to see the number of layers
model = model_builder.getModel()

EPOCHS = 20
lr_0 = 2e-3
epoch_change = 15
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[epoch_change*int(np.ceil(N_train/batch_size))], values=[lr_0, lr_0/10])
sigma_0 = 1e-3
sigma_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[epoch_change*int(np.ceil(N_train/batch_size))], values=[sigma_0, 0.])

optimizers = [
    LAdam(learning_rate=lr_schedule, sigma=0.),
    # LAdam(learning_rate=lr_schedule, sigma=sigma_schedule),
    # LayerLAdam(learning_rate=lr_schedule, sigma=sigma_schedule, langevin_layers=range(int(0.1*len(model.layers)))),
]


# dataloader = DataLoaderFromMap(
#     N_train = N_train,
#     N_test = N_test,
#     batch_size = batch_size,
#     get_X0 = get_X0,
# )

dataloader = ConstantLoader(
    X0 = 1.*tf.ones((dim,)),
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

experiment.plot_traj(opt_index=0)

experiment.save_data(dir_name=exp_name +'_N{}'.format(N_euler))
experiment.save_traj(dir_name=exp_name +'_N{}'.format(N_euler))


# Get the control function
print("Example of value of the control at X0 at time 0: {}"
      .format(experiment.models[optimizers[0]].ctrl_model(
          tf.concat([1.*tf.ones((1,dim)), 1.*tf.zeros((1,1))], axis=1))))

