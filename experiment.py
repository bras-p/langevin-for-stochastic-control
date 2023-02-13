import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import mkdir, listdir

from langevin_optimizers.base import set_langevin
import utils




class Experiment():

    def __init__(
        self,
        model_builder,
        dataloader,
        EPOCHS = 10,
        optimizers = ['adam'],
        base = './',
    ):
        self.model_builder = model_builder
        self.dataloader = dataloader
        self.EPOCHS = EPOCHS
        self.optimizers = optimizers
        self.base = base


    def load_data(self):
        self.ds, self.ds_test = self.dataloader.loadData()

    
    def getModel(self, optimizer):
        model = self.model_builder.getModel()
        model.compile(optimizer=optimizer, loss=utils.custom_loss, metrics=[utils.sq_avg])
        return model
    

    def run_experiment(self, initializer=None):
        model = self.getModel('adam')
        if initializer == 'zero':
            utils.initialize_to_zero(model)
        if initializer == 'normal':
            utils.initialize_to_normal(model, sigma=0.1)
        self.eval_train_init = model.evaluate(self.ds)
        self.eval_test_init = model.evaluate(self.ds_test)
        model.save_weights(self.base + '/checkpoints/initial_checkpoint')

        self.models = {}
        for optimizer in self.optimizers:
            self.models[optimizer] = self.getModel(optimizer)
            self.models[optimizer].evaluate(self.ds_test)
            self.models[optimizer].load_weights(self.base + '/checkpoints/initial_checkpoint')
            set_langevin(self.models[optimizer])
            self.models[optimizer].fit(self.ds, epochs=self.EPOCHS, validation_data=self.ds_test)


    def plot(self):
        # fig, (ax1, ax2) = plt.subplots(2)
        x = np.arange(self.EPOCHS+1)
        for optimizer in self.optimizers:
            y = np.array([self.eval_test_init[0]] + self.models[optimizer].history.history['val_loss'])
            y_var = np.array([self.eval_test_init[1]] + self.models[optimizer].history.history['val_sq_avg'])
            ci = 1.96*np.sqrt(y_var-y**2)/np.sqrt(self.dataloader.N_test)
            plt.plot(x, y, label='sigma='+print_schedule(optimizer.sigma))
            plt.fill_between(x, y-ci, y+ci, alpha=0.2)
        plt.legend()
        plt.show()


    def save_data(self, dir_name='experiment'):
        mkdir(self.base + dir_name)
        x = np.arange(self.EPOCHS+1)
        for k in range(len(self.optimizers)):
            # df = pd.DataFrame({'time': x, 'f':[self.eval_train_init[0]]+self.models[self.optimizers[k]].history.history['loss']})
            # df.to_csv(self.base + dir_name + '/' + str(k) + '_loss' + '.csv')
            y = np.array([self.eval_test_init[0]]+self.models[self.optimizers[k]].history.history['val_loss'])
            y_var = np.array([self.eval_test_init[1]] + self.models[self.optimizers[k]].history.history['val_sq_avg'])
            ci = 1.96*np.sqrt(y_var-y**2)/np.sqrt(self.dataloader.N_test)
            df = pd.DataFrame({'time': x, 'f':y, 'ci':ci, 'f_plus':y+ci, 'f_minus':y-ci})
            df.to_csv(self.base + dir_name + '/' + str(k) + '_v_loss' + '.csv')

        f = open(self.base + dir_name + '/' + "experiment_settings.txt", "w+")
        f.write('Model: ' + type(self.model_builder).__name__+'\n')
        f.write('Model options: ' + str(self.model_builder.__dict__)+'\n')
        f.write('Dataloader: ' + self.dataloader.__class__.__name__ + '\n')
        f.write('Data options: ' + str(self.dataloader.__dict__) +'\n')

        f.write('Learning rate: ' + print_schedule(self.optimizers[0].learning_rate)+'\n')
        f.write('EPOCHS: ' + str(self.EPOCHS)+'\n'+'\n')

        for k in range(len(self.optimizers)):
            f.write('Optimizer ' + str(k) + ' : ' + self.optimizers[k]._name + '  ')
            if hasattr(self.optimizers[k], 'sigma'):
                f.write('Sigma schedule: ' + print_schedule(self.optimizers[k].sigma) + '\n')
        f.close()

    
    def plot_traj(self, opt_index=0):
        if not hasattr(self, 'ds'):
            raise NameError('Must load_data first')
        if not hasattr(self, 'models'):
            raise NameError('Must run_experiment first')
        trained_model = self.models[self.optimizers[opt_index]]
        x, _ = next(iter(self.ds))
        # x = x[0:1]
        traj_example = trained_model(x)[0,1:]
        traj_dim = int(len(traj_example)/(self.model_builder.N_euler+1))
        self.traj_example = [np.array([
            traj_example[traj_dim*k + j] for k in range(self.model_builder.N_euler+1)]) for j in range(traj_dim)
        ]
        self.opt_index = opt_index
        for j in range(traj_dim):
            plt.plot(self.traj_example[j], label=str(j))
        plt.legend()
        plt.show()


    def save_traj(self, dir_name='trajectory'):
        if not hasattr(self, 'traj_example'):
            raise NameError('Must plot_traj first')
        traj = self.traj_example
        trajs = { 'f'+str(k): traj[k] for k in range(len(traj)) }
        trajs.update({'time':np.arange(len(traj[0]))})
        df = pd.DataFrame(trajs)
        df.to_csv(self.base + dir_name + '/' + 'traj_example' + '.csv')

        f = open(self.base + dir_name + '/' + 'traj_example_settings.txt', 'w+')
        f.write('Model: ' + type(self.model_builder).__name__+'\n')
        f.write('Model options: ' + str(self.model_builder.__dict__)+'\n')
        f.write('Dataloader: ' + self.dataloader.__class__.__name__ + '\n')
        f.write('Data options: ' + str(self.dataloader.__dict__) +'\n')

        f.write('Learning rate: ' + print_schedule(self.optimizers[0].learning_rate)+'\n')
        f.write('EPOCHS: ' + str(self.EPOCHS)+'\n')

        k = self.opt_index
        f.write('Optimizer ' + str(k) + ' : ' + self.optimizers[k]._name + '  ')
        if hasattr(self.optimizers[k], 'sigma'):
            f.write('Sigma schedule: ' + print_schedule(self.optimizers[k].sigma) + '\n')
        f.close()
        
        



def print_schedule(lr_schedule):
    if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
        return str(lr_schedule.__dict__)
    else:
        return str(lr_schedule)


def plot_data(base, dir_name):
    fs = listdir(base+dir_name)
    plt.ylabel('Test loss')
    for f in fs:
        if f[-10:-4] =='v_loss':
            df = pd.read_csv(base+dir_name +'/'+ f)
            plt.plot(df['f'])
    plt.title(dir_name)
    plt.legend()
    plt.show()


