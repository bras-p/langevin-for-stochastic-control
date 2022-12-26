# Langevin algorithms for Markovian Neural Networks and Deep Stochastic control

Stochastic Gradient Descent Langevin Dynamics (SGLD) algorithms, which add noise to the classic gradient descent, are known to improve the training of neural networks in some cases where the neural network is very deep.
In this paper we study the possibilities of training acceleration for the numerical resolution of stochastic control problems through gradient descent, where the control is parametrized by a neural network. If the control is applied at many discretization times then solving the stochastic control problem reduces to minimizing the loss of a very deep neural network.
We numerically show that Langevin algorithms improve the training on various stochastic control problems like hedging and resource management, and for different choices of gradient descent methods.

In this repository we give the implementation of Langevin and Layer Langevin optimizers as instances of the TensorFlow <tt>tf.keras.optimizers.Optimizer</tt> base class and we compare Langevin and non-Langevin optimizers for the training of various stochastic control problems.

The machine learning library that is used is TensorFlow.




## Requirements

```setup
pip install tensorflow
pip install pandas
```

## Training examples

```
python simulation_fishing.py
python simulation_deep_hedging.py
python simulation_oil_drilling.py
```




## Citation
Please cite our paper if it helps your research (only arXiv for now):

	@ARTICLE{2021arXiv210111557B,
	       author = {{Bras}, Pierre},
	        title = "{Convergence rates of Gibbs measures with degenerate minimum}",
	      journal = {arXiv e-prints},
	     keywords = {Mathematics - Probability},
	         year = 2021,
	        month = jan,
	          eid = {arXiv:2101.11557},
	        pages = {arXiv:2101.11557},
	archivePrefix = {arXiv},
	       eprint = {2101.11557},
	 primaryClass = {math.PR},
	       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210111557B},
	      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
	}