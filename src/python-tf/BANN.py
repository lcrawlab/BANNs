import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import warnings

class BANN_Quantitative:
	def __init__(self, layers, l_rate):
		self.n_layers = len(layers)
		self.model = self.build_model(layers, True)
		self.l_rate=l_rate
	
		def elbo(y_true, y_pred):
			return tf.losses.mean_squared_error(y_true, y_pred) + sum(self.model.losses)/K.cast(K.shape(y_true)[0], "float32")
		self.model.compile(loss=elbo, optimizer=tf.keras.optimizers.Adam(self.l_rate), metrics=["mse"])

	def build_model(self, layers, check_shapes=True):
		model = tf.keras.Sequential()
		for l in layers:
			model.add(l)
		# if check_shapes:
		# 	assert model.layers[0].input_shape[1]==self.p, "p ({}) does not match the input dimension of the first layer ({})".format(self.p, model.layers[0].input_shape[1])
		# 	assert model.layers[-1].units==1, "Output dimension of the final layer is not 1 (it is {})".format(model.layers[-1].units)
		return model
    
	def fit(self, X, y, callbacks=[], **kwargs):
		return self.model.fit(X, y, callbacks=callbacks+[tf.keras.callbacks.TerminateOnNaN()], **kwargs)

	def train(self, X, y, n_epochs, batch_size):
		return self.fit(X, y, epochs=n_epochs, batch_size=batch_size)
        
	def __call__(self):
		return self.model
    
	def var_params(self):
		G1_loc, G1_scale = [K.eval(self.G1_loc), K.eval(self.G1_scale)]
		return G1_loc, G1_scale
    
	def score(self, X, y, n_mc_samples, std=False, **kwargs):
		out = [self.model.evaluate(X, y, verbose=0, **kwargs)[1] for i in range(n_mc_samples)]
		if not np.isfinite(out).all():
			warnings.warn("There were {} non-finite results when evaluating the scores. These were removed.".format(np.isnan(out).sum()))
		if std:
			return np.nanmean(out), np.nanstd(out)
		else:
			return np.nanmean(out)
