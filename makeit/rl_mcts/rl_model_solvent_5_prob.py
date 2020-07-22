import makeit.global_config as gc
from makeit.prioritization.prioritizer import Prioritizer
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np
from makeit.utilities.io.logger import MyLogger
import math
import sys
import random
import time
import os
import makeit.utilities.io.pickle as pickle
import tensorflow as tf 
import math
import traceback

rl_model_loc = 'rl_model'
max_norm = 5

def linearND(input_, output_size, scope, reuse=False, init_bias=0.0):
	shape = input_.get_shape().as_list()
	ndim = len(shape)
	stddev = min(1.0 / math.sqrt(shape[-1]), 0.1)
	with tf.variable_scope(scope, reuse=reuse):
		W = tf.get_variable("Matrix", [shape[-1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
	X_shape = tf.gather(tf.shape(input_), list(range(ndim-1)))
	target_shape = tf.concat([X_shape, [output_size]], 0)
	exp_input = tf.reshape(input_, [-1, shape[-1]])
	if init_bias is None:
		res = tf.matmul(exp_input, W)
	else:
		with tf.variable_scope(scope, reuse=reuse):
			b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
		res = tf.matmul(exp_input, W) + b
	res = tf.reshape(res, target_shape)
	res.set_shape(shape[:-1] + [output_size])
	return res

class PolicyValueNetwork(object):

	def __init__(self):

		self.FP_len = 2048
		self.FP_rad = 2
		self.template_count = 100
		self.max_cum_prob = 1
		self.batch_size = None
		self.NK = 1000#50
		self.hidden_size = 300
		self.output_size = 163723 #61142
		self.depth = 5

		self.input_mol = tf.placeholder(tf.float32, [self.batch_size, self.FP_len])
		self.mol_hiddens = tf.nn.relu(linearND(self.input_mol, self.hidden_size, scope="encoder0", reuse=tf.AUTO_REUSE))
		for d in range(1, self.depth):
			self.mol_hiddens = tf.nn.relu(linearND(self.mol_hiddens, self.hidden_size, scope="encoder%i"%d, reuse=tf.AUTO_REUSE))

		hidden = tf.nn.relu(linearND(self.mol_hiddens, self.hidden_size, scope="prob1", reuse=tf.AUTO_REUSE))
		self.logits = linearND(hidden, self.output_size, scope="prob2", reuse=tf.AUTO_REUSE)
		self.prob = tf.nn.softmax(self.logits)

		hidden = tf.nn.relu(linearND(self.mol_hiddens, self.hidden_size, scope="value1", reuse=tf.AUTO_REUSE))
		self.value = linearND(hidden, 1, scope="value2", reuse=tf.AUTO_REUSE)

		_, self.topk = tf.nn.top_k(self.prob, k=self.NK)


	def train(self, learning_rate, weight_decay):

		self.true_prob = tf.placeholder(tf.float32, [self.batch_size, self.output_size])
		self.true_value = tf.placeholder(tf.float32, [self.batch_size, 1])

		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		#mask = tf.to_float(tf.greater(self.true_value, -self.true_value ))#0))#######02132019###01262019#####
		#self.loss = tf.losses.mean_squared_error(self.true_value, self.value)#, weights=mask)#- tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.true_prob))#, axis=0)###TODO: add l2 regularization, pay attention to the loss function
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.true_prob))#, axis=0)###TODO: add l2 regularization, pay attention to the loss function
		#found the bug in self.loss for prob on 03052019,previous loss was wrong
		#self.loss = - tf.reduce_mean(tf.reduce_sum(self.true_prob * tf.log(self.prob), axis=1))#no optimizations in value here. value is fixed to 1.

		# for var in tf.trainable_variables():
		#     if 'Matrix' in var.name:
		#         self.loss += weight_decay * tf.nn.l2_loss(var)

		self.opt_op = self.optimizer.minimize(self.loss)


	def feed_dict(self, input_mol, true_prob=None, true_value=None):
		d = {self.input_mol: input_mol}
		if true_prob is not None:
			d[self.true_prob] = true_prob
			d[self.true_value] = true_value
		return d


	def pretrain(self, weight_decay=1e-3):
		
		self.label = tf.placeholder(tf.int32, [self.batch_size,])
		self.lr = tf.placeholder(tf.float32, [])
		
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits))

		# for var in tf.trainable_variables():
		#     if 'Matrix' in var.name:
		#         self.loss += weight_decay * tf.nn.l2_loss(var)


		self.param_norm = tf.global_norm(tf.trainable_variables())
		grads_and_vars = self.optimizer.compute_gradients(self.loss)
		grads, var = zip(*grads_and_vars)
		self.grad_norm = tf.global_norm(grads)
		new_grads, _ = tf.clip_by_global_norm(grads, max_norm)
		grads_and_vars = zip(new_grads, var)
		self.opt_op = self.optimizer.apply_gradients(grads_and_vars)



		#self.opt_op = self.optimizer.minimize(self.loss)


	def pretrain_feed_dict(self, input_mol, label, lr):
		return {self.input_mol: input_mol, self.label: label, self.lr: lr}




class RLModel(object):
	'''
	Allows to prioritize the templates based on their relevance
	'''

	def __init__(self):

		self.model = PolicyValueNetwork()

		self.FP_len = 2048
		self.FP_rad = 2
		self.vars = []
		self.template_count = 100
		self.max_cum_prob = 1

		config = tf.ConfigProto()
		config.allow_soft_placement=True
		config.gpu_options.allow_growth = True
		self.session = tf.Session(config=config)
		tf.global_variables_initializer().run(session=self.session)#creates a node in the graph that will initialize all variables when it is run

		

		self.saver = tf.train.Saver(max_to_keep=None)###############
		

	def train(self, learning_rate=1e-4, weight_decay=1e-4):
		self.model.train(learning_rate, weight_decay)
		tf.global_variables_initializer().run(session=self.session)
	def train_pretrain(self, weight_decay=1e-4):
		self.model.pretrain(weight_decay)
		tf.global_variables_initializer().run(session=self.session)

	def load(self, save_path):

		restore_path = tf.train.latest_checkpoint(save_path)
		self.saver.restore(self.session, restore_path)###################

	def save(self, save_path):
		self.saver.save(self.session, save_path+"/model.ckpt")###########################33

	def update(self, smiles, true_prob, true_value, n_epochs=1):
		fps = np.array([self.smi_to_fp(smi) for smi in smiles])
		for i in range(n_epochs):
			_, loss = self.session.run([self.model.opt_op, self.model.loss], feed_dict=self.model.feed_dict(fps, true_prob, true_value))
		#self.session.run(self.model.opt_op, feed_dict=self.model.feed_dict(fps, true_prob, true_value))
		# value = self.session.run(self.model.value, feed_dict=self.model.feed_dict(fps))
		# print(value)
			print('loss is {}'.format(loss))

		return loss
	
	def update_pretrain(self, smiles, tids, n_epochs=1, lr = 1e-3):
		fps = np.array([self.smi_to_fp(smi) for smi in smiles])
		for i in range(n_epochs):
			_, loss, pnorm, gnorm = self.session.run([self.model.opt_op, self.model.loss, self.model.param_norm, self.model.grad_norm], feed_dict=self.model.pretrain_feed_dict(fps, tids, lr = lr))
		#self.session.run(self.model.opt_op, feed_dict=self.model.feed_dict(fps, true_prob, true_value))
		# value = self.session.run(self.model.value, feed_dict=self.model.feed_dict(fps))
		# print(value)
			print('loss is {}'.format(loss))
			print('pnorm is {}'.format(pnorm))
			print('gnorm is {}'.format(gnorm))

		return loss

	def mol_to_fp(self, mol):
		if mol is None:
			return np.zeros((self.FP_len,), dtype=np.float32)
		return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, self.FP_rad, nBits=self.FP_len,
															  useChirality=True), dtype=np.float32)

	def smi_to_fp(self, smi):
		if not smi:
			return np.zeros((self.FP_len,), dtype=np.float32)
		return self.mol_to_fp(Chem.MolFromSmiles(smi))

	def get_topk_from_smi(self, smi='', k=50):
		if not smi:
			return []
		mol = Chem.MolFromSmiles(smi)
		if not mol:
			return []
		return self.get_topk_from_mol(mol, k=k)

	def get_topk_from_mol(self, mol, k=50):####here fixed a bug
		prob, value = self.get_prob_value_from_mol(mol)
		indice = list(prob[:].argsort()[-k:][::-1])
		#indice = list(prob[:].argsort()[-k:][::-1])
		# return indice, prob[0], value
		#probs=prob
		#return indice, prob[-k:][::-1], value###02142019
		# prob=list(prob)
		# prob.sort()
		prob = np.sort(prob)

		return indice, prob[-k:][::-1], value
	
	def get_prob_value_from_smi(self, smi):
		mol = Chem.MolFromSmiles(smi)
		return self.get_prob_value_from_mol(mol)

	def get_logits_value_from_smi(self, smi):
		mol = Chem.MolFromSmiles(smi)
		return self.get_logits_value_from_mol(mol)

	def get_prob_value_from_mol(self, mol):###########################
		fp = self.mol_to_fp(mol).astype(np.float32).reshape((1, self.FP_len))
		prob, value = self.session.run([self.model.prob, self.model.value], feed_dict=self.model.feed_dict(fp))
		return prob[0], value[0,0]#1. #value[0,0]
		#return prob[0], value

	def get_logits_value_from_mol(self, mol):
		fp = self.mol_to_fp(mol).astype(np.float32).reshape((1, self.FP_len))
		logits, value = self.session.run([self.model.logits, self.model.value], feed_dict=self.model.feed_dict(fp))
		return logits[0], value[0,0]#1. #value[0,0]
		#return prob, value




if __name__ == '__main__':
	model = RLModel(use_tf=True)
	model.load_model()
	smis = ['CCCOCCC', 'CCCNc1ccccc1']
	for smi in smis:
		lst = model.get_topk_from_smi(smi)
		print('{} -> {}'.format(smi, lst))

	# model2 = RelevanceTemplatePrioritizer(use_tf=True)
	# model2.load_model()
	# for smi in smis:
	#     lst = model2.get_topk_from_smi(smi)
	#     print('{} -> {}'.format(smi, lst))
		
	# import time
	# time.sleep(10)
