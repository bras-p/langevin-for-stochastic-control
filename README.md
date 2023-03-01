# [Langevin algorithms for Markovian Neural Networks and Deep Stochastic control](https://arxiv.org/abs/2212.12018)

Stochastic Gradient Descent Langevin Dynamics (SGLD) algorithms, which add noise to the classic gradient descent, are known to improve the training of neural networks in some cases where the neural network is very deep.
In this paper we study the possibilities of training acceleration for the numerical resolution of stochastic control problems through gradient descent, where the control is parametrized by a neural network. If the control is applied at many discretization times then solving the stochastic control problem reduces to minimizing the loss of a very deep neural network.
We numerically show that Langevin algorithms improve the training on various stochastic control problems like hedging and resource management, and for different choices of gradient descent methods.

In this repository we give the implementation of Langevin and Layer Langevin optimizers as instances of the TensorFlow <tt>tf.keras.optimizers.Optimizer</tt> base class and we compare Langevin and non-Langevin optimizers for the training of various stochastic control problems.

The machine learning library that is used is TensorFlow.

[arXiv link](https://arxiv.org/abs/2212.12018)




## Requirements

```setup
pip install tensorflow
pip install pandas
```

Tensorflow version used:
```
tensorflow                   2.10.0
```


## Training examples

```
python simulation_fishing.py
python simulation_deep_hedging.py
python simulation_oil_drilling.py
```


## Package
The package <tt>langevin_optimizers</tt> gives Langevin optimizers as a instances of the <tt>tf.keras.optimizers.Optimizer</tt> base class. See the <tt>notebook.ipynb</tt> for more details.



## Citation
Please cite our paper if it helps your research:

	@ARTICLE{2022arXiv221212018B,
		author = {{Bras}, Pierre and {Pag{\`e}s}, Gilles},
			title = "{Langevin algorithms for Markovian Neural Networks and Deep Stochastic control}",
		journal = {arXiv e-prints},
		keywords = {Quantitative Finance - Computational Finance, Computer Science - Machine Learning, Mathematics - Optimization and Control, Statistics - Machine Learning},
			year = 2022,
			month = dec,
			eid = {arXiv:2212.12018},
			pages = {arXiv:2212.12018},
			doi = {10.48550/arXiv.2212.12018},
	archivePrefix = {arXiv},
		eprint = {2212.12018},
	primaryClass = {q-fin.CP},
		adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv221212018B},
		adsnote = {Provided by the SAO/NASA Astrophysics Data System}
	}
