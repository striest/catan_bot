import os
import torch
import numpy as np
import pandas as pd
import pickle
import dill

from catanbot.rl.utils.os_utils import maybe_mkdir

class Experiment:
	"""
	Wrapper around RL algorithms that drives the training and handles all the IO stuff. (i.e. making directories, saving networks, recording performance, etc.)	
	"""
	def __init__(self, algo, name, experiment_filepath = '', save_every=10, save_logs_every=100, save_best=True):
		self.algo = algo
		self.name = name
		self.base_fp = os.path.join(os.getcwd(), experiment_filepath, name)
		self.log_fp = os.path.join(self.base_fp, '_log')
		self.save_every = save_every
		self.save_logs_every = save_logs_every
		self.df = None
		self.save_best = save_best
		self.best_return = -np.inf

	def build_experiment_dir(self):
		if os.path.exists(self.base_fp):
			i = input('Directory {} already exists. input \'q\' to stop the experiment (and anything else to keep going).'.format(self.base_fp))
			if i.lower() == 'q':
				exit(0)
		maybe_mkdir(self.base_fp)
		maybe_mkdir(self.log_fp)

	def save_networks(self, dir_fp):
		maybe_mkdir(dir_fp)
		self.write_summary(dir_fp)
		for k, net in self.algo.networks.items():
			torch.save(net, os.path.join(dir_fp, '{}.cpt'.format(k)))

	def save_env(self):
		#Kinda questionable if this is a good idea, but it'll let me make videos with arbitrary envs by looking at the experiment output.
		env_fp = os.path.join(self.base_fp, 'env.cpt')
		dill.dump(self.algo.env, open(env_fp, 'wb'))

	def update_log(self):
		if self.df is None:
			self.df = self.algo.logger.dump_dataframe()
		else:
			self.df = pd.concat([self.df, self.algo.logger.dump_dataframe()], ignore_index=True)

	def save_log(self):
		if os.path.exists(os.path.join(self.log_fp, "log.csv")):
			self.df.to_csv(os.path.join(self.log_fp, "log.csv"), mode='a', header=False)
		else:
			self.df.to_csv(os.path.join(self.log_fp, "log.csv"))	
		self.df = None

	def write_summary(self, summary_fp):
		with open(os.path.join(summary_fp, '_summary'), 'w') as f:
			f.write('itr = {}\nmean ret = {}'.format(self.algo.current_epoch, self.algo.logger.get(prefix='Performance', field='Return Mean')))

	def run(self):
		self.build_experiment_dir()
		self.save_env()
		for e in range(self.algo.total_epochs):
			self.algo.train_iteration()
			self.update_log()
			if self.algo.current_epoch % self.save_every == 0:
				self.save_networks(dir_fp = os.path.join(self.base_fp, "itr_{}".format(self.algo.current_epoch)))
					
			if self.algo.current_epoch % self.save_logs_every == 0:
				self.save_log()

			if self.save_best and self.algo.logger.get(prefix='Performance', field='Return Mean') > self.best_return:
				self.best_return = self.algo.logger.get(prefix='Performance', field='Return Mean')
				self.save_networks(dir_fp = os.path.join(self.base_fp, "_best".format(self.algo.current_epoch)))
