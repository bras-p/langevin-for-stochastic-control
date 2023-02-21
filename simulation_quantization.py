import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import tensorflow_probability as tfp
tfd = tfp.distributions

from models.distorsion import KMeans
from experiment import Experiment
from data.from_map import DataLoaderFromMap
from langevin_optimizers.ladam import LAdam, LayerLAdam
from langevin_optimizers.lrmsprop import LRMSprop, LayerLRMSprop
from langevin_optimizers.ladadelta import LAdadelta, LayerLAdadelta

# https://medium.com/analytics-vidhya/gaussian-mixture-models-with-tensorflow-probability-125315891c22


batch_size = 512
N_train=5*batch_size
N_test=50*batch_size


# parameters for the distribution to quantize
# dim = 2
# N_mix = 5

# radius = 1.
# mus = tf.constant(
#     [np.array([radius*np.cos(2*k*np.pi/N_mix), radius*np.sin(2*k*np.pi/N_mix)]) for k in range(N_mix)],
#     dtype=tf.float32
# )

# sigmas = [0.5*np.ones(dim) for i in range(N_mix)]
# sigmas = tf.constant(sigmas, dtype=tf.float32)

# w = (1./N_mix)*tf.ones((N_mix))

# gaussian_mix = tfd.MixtureSameFamily(
#     mixture_distribution=tfd.Categorical(probs=w),
#     components_distribution=tfd.MultivariateNormalDiag(
#         loc = mus, 
#         scale_diag = sigmas
#     )
# )


# def get_X0(batch_size):
#     return gaussian_mix.sample(batch_size)

# plot the distribution in 2d
# xs = np.linspace(-3,3,100)
# ys = np.linspace(-3,3,100)
# xs, ys = np.meshgrid(xs, ys)

# density = gaussian_mix.prob(np.stack((xs.flatten(), ys.flatten()), axis=1)).numpy().reshape((100,100))
# plt.pcolormesh(xs, ys, density)
# plt.colorbar()
# plt.show()



# K = N_mix # nb of quantizers



# with the normal distribution
dim = 20
K = 10000

def get_X0(batch_size):
    return tf.random.normal((batch_size, dim))


model_builder = KMeans(
    K=K
    )



base = './results/'
exp_name = 'quantization'

# first get a dummy model to see the number of layers
model = model_builder.getModel()

EPOCHS = 100
lr_0 = 5e-3
epoch_change = 80
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[epoch_change*int(np.ceil(N_train/batch_size))], values=[lr_0, lr_0/10])
sigma_0 = 5e-3
sigma_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[epoch_change*int(np.ceil(N_train/batch_size))], values=[sigma_0, 0.])

optimizers = [
    # LAdam(learning_rate=lr_schedule, sigma=0.),
    LAdam(learning_rate=lr_schedule, sigma=sigma_schedule),
    # LayerLAdam(learning_rate=lr_schedule, sigma=sigma_schedule, langevin_layers=range(int(0.1*len(model.layers)))),
]


dataloader = DataLoaderFromMap(
    N_train = N_train,
    N_test = N_test,
    batch_size = batch_size,
    get_X0 = get_X0,
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


experiment.save_data(dir_name=exp_name +'_dim{}_adadelta'.format(dim))






# some results specific to the quantization
lim = 3
xs = np.linspace(-lim, lim, 100)
ys = np.linspace(-lim, lim, 100)
xs, ys = np.meshgrid(xs, ys)

model1 = experiment.models[optimizers[0]]

centroids = [model1.trainable_variables[0][k].numpy() for k in range(len(model1.trainable_variables[0].numpy()))]
centroids_x = [centroids[k][0] for k in range(len(centroids))]
centroids_y = [centroids[k][1] for k in range(len(centroids))]

# mus_x = [mus[k][0].numpy() for k in range(len(centroids))]
# mus_y = [mus[k][1].numpy() for k in range(len(centroids))]

# from scipy.spatial import Voronoi, voronoi_plot_2d
# voronoi = Voronoi(centroids)
# voronoi_plot_2d(voronoi, show_vertices=False, line_colors='red',
#                       line_width=3, line_alpha=1, point_size=0)

plt.pcolormesh(xs, ys, (2.*np.pi)**(-1)*np.exp(-0.5*(xs**2 + ys**2)))
plt.scatter(centroids_x, centroids_y, s=1., color='red')
# plt.scatter(mus_x, mus_y, color='blue')

plt.xlim([-lim,lim])
plt.ylim([-lim,lim])

plt.colorbar()
plt.show()

