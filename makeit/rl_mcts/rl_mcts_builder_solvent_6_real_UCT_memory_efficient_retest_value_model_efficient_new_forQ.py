from makeit.rl_mcts.rl_transformer import RLRetroTransformer
from makeit.utilities.buyable.pricer import Pricer
from multiprocessing import Process, Manager, Queue, Pool
from celery.result import allow_join_result
from pymongo import MongoClient
import rdkit.Chem as Chem 

from makeit.mcts.cost import Reset, score_max_depth, MinCost, BuyablePathwayCount
# from makeit.mcts.misc import get_feature_vec, save_sparse_tree
# from makeit.mcts.misc import value_network_training_states
from makeit.rl_mcts.nodes import Chemical, Reaction
from makeit.utilities.io.logger import MyLogger
from makeit.utilities.io import model_loader
from makeit.utilities.formats import chem_dict, rxn_dict
from makeit.rl_mcts.neuralnetwork_pred_top_slv import NeuralNetContextRecommender

from makeit.rl_mcts.rl_model_solvent_5_val_1 import RLModel as RLModel_val#####_1 no mask, otherwise mask
from makeit.rl_mcts.rl_model_solvent_5_prob import RLModel as RLModel_prob

import tensorflow as tf
import numpy as np

import makeit.global_config as gc
import Queue as VanillaQueue
import multiprocessing as mp
import cPickle as pickle 
import numpy as np
import traceback
import itertools
#import psutil
import random
import time 
import gzip 
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
rl_mcts_builder_loc = 'rl_mcts_builder'

WAITING = 0
DONE = 1

NUM_TEMPLATES = 50###confine the search space to the top 50 templates recommended by PolicyValueNetwork in pretrained model
RL_PROB_MODEL_PATH = 'makeit/models/rl_pretrain/'
#RL_PROB_MODEL_PATH = 'makeit/models/rl_prob_completely_new_scheme_56_clean_weighted_small_lr_1_failed_2/'
#RL_VAL_MODEL_PATH = 'makeit/models/rl_val_retrain_5_2/'
RL_VAL_MODEL_PATH = 'makeit/models/rl_val_retrain_round_2_6/'
#RL_VAL_MODEL_PATH = 'makeit/models/rl_val_02252019_value_only/'

#PRETRAIN_MODEL_PATH = RL_MODEL_PATH
#RL_MODEL_PATH = PRETRAIN_MODEL_PATH


def run_initial_topk_prob(smiles):
	model = RLModel_prob()
	model.load(RL_PROB_MODEL_PATH)
	indices, prob, value = model.get_topk_from_smi(smi=smiles, k=NUM_TEMPLATES)
	return (indices, prob, value)

def run_initial_topk_val(smiles):
	model=RLModel_val()
	model.load(RL_VAL_MODEL_PATH)
	indices, prob, value = model.get_topk_from_smi(smi=smiles, k=NUM_TEMPLATES)
	return (indices, prob, value)
# def run_update_prob((smiles, true_prob, true_value)):
#     model = RLModel_prob()
#     model.train()
#     model.load(RL_PROB_MODEL_PATH)
#     model.update(smiles, true_prob, true_value)
#     model.save(RL_PROB_MODEL_PATH)
#     return True
# def run_update_val((smiles, true_prob, true_value)):
#     model = RLModel_val()
#     model.train()
#     model.load(RL_VAL_MODEL_PATH)
#     model.update(smiles, true_prob, true_value)
#     model.save(RL_VAL_MODEL_PATH)
#     return True

class MCTS:
	def __init__(self, retroTransformer=None, pricer=None, 
				 max_branching=100, max_depth=10, celery=False, 
				 nproc=8, ngpus=2, mincount=25, chiral=True, max_ppg=100, 
				 mincount_chiral=10, solvent_evaluation = False):
		self.celery = celery
		self.mincount = mincount
		self.mincount_chiral = mincount_chiral
		self.max_depth = max_depth
		self.max_branching = max_branching

		self.nproc = nproc
		self.ngpus = ngpus
		self.chiral = chiral
		self.max_cum_template_prob = 1
		self.status = {}

		self.solvent_evaluation = solvent_evaluation
		if self.solvent_evaluation:
			self.cont = NeuralNetContextRecommender()
			self.cont.load_nn_model(model_path=gc.NEURALNET_CONTEXT_REC['model_path'], info_path=gc.NEURALNET_CONTEXT_REC[
					   'info_path'], weights_path=gc.NEURALNET_CONTEXT_REC['weights_path'])

		if pricer:
			self.pricer = pricer
		else:
			self.pricer = Pricer()
		self.pricer.load()

		self.reset()##reset workers, coordinators, running, pathways, pathway_count, successful_pathway_count, mincost, active_chemicals.
		## Load Transformer
		self.retroTransformer = RLRetroTransformer(mincount=self.mincount, mincount_chiral=self.mincount_chiral)
		self.retroTransformer.load(self.chiral)
		MyLogger.print_and_log('Retro synthetic transformer loaded.', rl_mcts_builder_loc)

		if self.celery:
			def expand(smiles, chem_id, queue_depth, branching):
				pass
		else:
			def expand(_id, chem_smi, template_id):
				self.expansion_queues[_id].put((_id, chem_smi, template_id))############
				self.status[(chem_smi, template_id)] = WAITING
		self.expand = expand

		###start up parallelization
		if self.celery:
			def prepare():
				pass
		else:
			def prepare():
				print 'Tree builder spinning off {} child processes'.format(self.nproc)
				for i in range(self.nproc):
					p = Process(target = self.work, args = (i,))
					self.workers.append(p)
					p.start()
		self.prepare = prepare

		if self.celery:
			def waiting_for_results():
				pass
		else:
			def waiting_for_results():
				waiting = [expansion_queue.empty()
						   for expansion_queue in self.expansion_queues]
				for results_queue in self.results_queues:
					waiting.append(results_queue.empty())
				waiting += self.idle
				return (not all(waiting))
		self.waiting_for_results = waiting_for_results

		#method to get a processed result
		if self.celery:
			def get_ready_result():
				pass
		else:
			def get_ready_result():
				for results_queue in self.results_queues:
					while not results_queue.empty():
						yield results_queue.get(timeout=0.1)
		self.get_ready_result = get_ready_result
		
		if self.celery:
			def set_initial_target(_id, smiles):
				pass
		else:
			def set_initial_target(_id, leaves):
				for leaf in leaves:
					if leaf in self.status:
						continue
					chem_smi, template_id = leaf
					self.expand(_id, chem_smi, template_id)
		self.set_initial_target = set_initial_target

		#define methods to stop working
		if self.celery:
			def stop():
				pass
		else:
			def stop():
				if not self.running:
					return
				self.done.value = 1
				for p in self.workers:
					if p and p.is_alive():
						p.terminate()
				self.running = False
		self.stop = stop
	def reset(self):
		if self.celery:
			# general parameters in celery format
			pass
		else:
			self.workers = []##a list of processes
			self.coordinator = None
			self.running = False
			
			## Queues 
			self.pathways = [0 for i in range(self.nproc)]
			self.pathway_count = 0 
			self.successful_pathway_count = 0
			self.mincost = 10000.0

			self.active_chemicals = [[] for x in range(self.nproc)]
			###########new add, reset Chemicals and Reactions################

			self.Chemicals = {}
			self.Reactions = {}

	def get_price(self, chem_smi):
		ppg = self.pricer.lookup_smiles(chem_smi, alreadyCanonical=True)
		return ppg

	def save_pathway(self, pathway):##not called elsewhere
		if self.fileName:
			with gzip.open("pathways/" + self.fileName, "a+b") as fid:
				pickle.dump(pathway, fid, pickle.HIGHEST_PROTOCOL)

	def load_crn(self):
		# if os.path.isfile("crn/"+self.fileName):
		#   try:
		#       with gzip.open("crn/"+self.fileName, "rb") as fid:
		#           Chemicals, Reactions = pickle.load(fid)
		#   except:
		#       print("WARNING: Pickle-file for {} is corrupted. Starting anew ... ".format(self.smiles))
		#       Chemicals, Reactions = {}, {}
		# else:
		Chemicals, Reactions = {}, {}
		
		self.Chemicals = Chemicals
		self.Reactions = Reactions
		
		if self.Chemicals == {}:
			return False
		else:
			return True 

	def save_crn(self):##not called elsewhere
		chem_key = tuple([self.smiles,0])
		Reset(self.Chemicals,self.Reactions)
		buyable_pathways = BuyablePathwayCount(chem_key, 
												self.max_depth, 
												self.Chemicals, 
												self.Reactions
												)
		#Reset(self.Chemicals,self.Reactions)
		mincost = MinCost(chem_key, self.max_depth, 
									self.Chemicals, self.Reactions)         
		if self.fileName:
			with gzip.open("crn/"+self.fileName, "wb") as fid:
				pickle.dump([self.Chemicals, self.Reactions], fid, pickle.HIGHEST_PROTOCOL)
		
		return mincost, buyable_pathways

	def get_solvent_score(self, rsmi_list=['CC1(C)OBOC1(C)C','Cc1ccc(Br)cc1'],psmi_list=['Cc1cccc(B2OC(C)(C)C(C)(C)O2)c1']):
		
		slv_score, slvs_list = self.cont.predict_top_solvents_from_split_smiles(rsmi_list, psmi_list)
		return slv_score, slvs_list
############important method coordinate, basically executed the majority of tree search work #########

	def coordinate(self):
		start_time = time.time()
		elapsed_time = time.time() - start_time
		next = 1
		while (elapsed_time < self.expansion_time):
			if (int(elapsed_time)/10 == next):
				next += 1
				print "Worked for {}/{} s".format(int(elapsed_time*10)/10.0, self.expansion_time)
				print "... current min-price {}".format(self.Chemicals[self.smiles].price)
				# print "... current best z {}".format(self.Chemicals[self.smiles].z)
				print "... |C| = {} |R| = {}".format(len(self.Chemicals),len(self.status))
			for (_id, chem_smi, template_id, reactants) in self.get_ready_result():
				self.status[(chem_smi, template_id)] = DONE
				R = self.Chemicals[chem_smi].reactions[template_id]
				R.waiting = False
				if len(reactants) == 0:
					R.valid = False
				else:
					for smi, indices, prob, value in reactants:
						R.reactant_smiles.append(smi)
						if smi not in self.Chemicals:
							self.Chemicals[smi] = Chemical(smi)
							self.Chemicals[smi].set_prob_value(indices, prob, value)
							ppg = self.get_price(smi)
							if ppg is not None and ppg > 0:
								self.Chemicals[smi].set_price(ppg)
					for smi in R.reactant_smiles:
						if len(R.reactant_smiles)!=0:
							#self.Chemicals[smi].scale_v=sum([self.Chemicals[smil].value for smil in R.reactant_smiles])/((len(R.reactant_smiles)+0.01)**2) ####scale_v
							self.Chemicals[smi].forQ=self.Chemicals[smi].value
					R.Q = sum([self.Chemicals[smi].value for smi in R.reactant_smiles])/(len(R.reactant_smiles)+0.01)
					if self.solvent_evaluation:
						R.s_Q, R.slvs_list = self.get_solvent_score(rsmi_list=R.reactant_smiles, psmi_list=[chem_smi])
			for _id in range(self.nproc):
				if self.expansion_queues[_id].empty() and self.results_queues[_id].empty() and self.idle[_id]:
					self.update(self.smiles, self.pathways[_id])
					self.pathways[_id] = {}
			for _id in range(self.nproc):
				if len(self.pathways[_id]) == 0:
					leaves, pathway = self.select_leaf()
					self.pathways[_id] = pathway
					self.set_initial_target(_id, leaves)

			elapsed_time = time.time() - start_time


			if self.Chemicals[self.smiles].price != -1 and self.time_for_first_path == -1:
				self.time_for_first_path = elapsed_time
				break####03222019 test only

		self.stop()

		for _id in range(self.nproc):
			self.update(self.smiles, self.pathways[_id])
			self.pathways[_id] = {}

		print "... exited prematurely."
	def work(self, i):
		with tf.device('/gpu:%d' % (i % self.ngpus)):
			self.model_prob = RLModel_prob()
			self.model_prob.load(RL_PROB_MODEL_PATH)
			self.model_val = RLModel_val()
			self.model_val.load(RL_VAL_MODEL_PATH)

		while True:
			# If done, stop
			if self.done.value:
				# print 'Worker {} saw done signal, terminating'.format(i)
				break
			
			# Grab something off the queue
			# try:
			if not self.expansion_queues[i].empty():

				self.idle[i] = False
				(_id, smiles, template_id) = self.expansion_queues[i].get(timeout=0.1)  # short timeout
				# print(_id, smiles, template_id)
				# prioritizers = (self.precursor_prioritization, self.template_prioritization)
				result = self.retroTransformer.get_outcomes(smiles, self.mincount, template_id, apply_fast_filter=True, filter_threshold=0.75)
				reactants = []
				
				if len(result) > 0:
					for smi in result[0]:
						indices, prob, value_1 = self.model_prob.get_topk_from_smi(smi=smi, k=NUM_TEMPLATES)### use relevance for prob
						indices_2, prob_2, value = self.model_val.get_topk_from_smi(smi=smi, k=NUM_TEMPLATES)###use rl for value
						reactants.append((smi, indices, prob, value))
						#print(indices, prob, value)####
					# print(_id, smiles, template_id, result)
				
				self.results_queues[_id].put((_id, smiles, template_id, reactants))
			
			# except Exception as e:
			#   print traceback.format_exc() 
			
			# time.sleep(0.01)
			self.idle[i] = True

	def UCB(self, chem_smi, c_exploration=0.2, c_solv=0.01, path=[]): #updated by xiaoxue at 12072018
		rxn_scores = []

		C = self.Chemicals[chem_smi]
		product_visits = C.visit_count#sum([C.reactions[tid].visit_count for tid in C.reactions])#C.visit_count
		max_estimate_Q = 0


		if C.if_initial_round:
			if len(C.burnable_indices_reverse) > 0:
				template_id = C.burnable_indices_reverse.pop()
				#print('template_id', template_id)
				selected_template_id = template_id
			else:
				C.if_initial_round = False
				selected_template_id = C.indices[0]

		else:

			for template_id in C.reactions:
				R = C.reactions[template_id] 
				if R.waiting or (not R.valid) or len(set(R.reactant_smiles) & set(path)) > 0:
					continue
				if R.done:
					continue
				max_estimate_Q = max(max_estimate_Q, R.Q)###
				Q_sa = R.Q + c_solv*R.s_Q##changed from - to +#####
				try:
					# ##############tuning c_exploration dynamically####################
					# if (max_estimate_Q-Q_sa)/c_exploration > 2:
					# 	c_exploration = (max_estimate_Q-Q_sa)/2
					##################################################################
					U_sa = c_exploration * np.sqrt(2* np.log(product_visits+0.01) / (R.visit_count+0.01))#####
				except:
					print(chem_smi, product_visits)
				score = Q_sa + U_sa
				# print('Qsa and Usa are:', Q_sa, U_sa)
				# print('R visit_count', R.visit_count)
				# print('product visit count', product_visits)
				rxn_scores.append((score, template_id))

		# unexpanded template
			# if (len(C.reactions) < self.max_branching and len(C.reactions) < len(C.indices)):# or chem_smi == self.smiles:
			#   template_id = C.indices[len(C.reactions)]
			#   #Q_sa = (max_estimate_Q + 0.1)###changed from - to +
			#   Q_sa = (0)
			#   ######################engineering c_exploration_new#####################################
			#   if max_estimate_Q/c_exploration > 2:
			#           c_exploration = max_estimate_Q/2
			#   #################################################################
			#   U_sa = c_exploration * np.sqrt(2*np.log(product_visits+0.1))#######
			#   score = Q_sa + U_sa
			#   # print('Qsa and Usa are:', Q_sa, U_sa)
			#   rxn_scores.append((score, template_id))

			if len(rxn_scores) > 0:
				sorted_rxn_scores = sorted(rxn_scores, key=lambda x: x[0], reverse=True)
				best_rxn_score, selected_template_id = sorted_rxn_scores[0]


			else:
				selected_template_id = None


		return selected_template_id





	def select_leaf(self, c_exploration=0.1, c_solv=0.):## tune c_exploration here #modified by xiaoxue 12102018

		#start_time = time.time()
		pathway = {}
		leaves = []
		queue = VanillaQueue.Queue()
		queue.put((self.smiles, 0, [self.smiles]))

		while not queue.empty():
			chem_smi, depth, path = queue.get()
			if depth >= self.max_depth or chem_smi in pathway:
				continue
			template_id = self.UCB(chem_smi, c_exploration=c_exploration,c_solv=c_solv, path=path)
			if template_id is None:
				continue
			pathway[chem_smi] = template_id
			C = self.Chemicals[chem_smi]
			C.visit_count += 1#VIRTUAL_LOSS
			if template_id not in C.reactions:
				C.reactions[template_id] = Reaction(chem_smi, template_id)
				R = C.reactions[template_id]
				R.visit_count += 1#VIRTUAL_LOSS
				leaves.append((chem_smi, template_id))
			else:
				R = C.reactions[template_id]
				R.visit_count += 1#VIRTUAL_LOSS
				for smi in R.reactant_smiles:
					assert smi in self.Chemicals
					# if self.Chemicals[smi].purchase_price == -1:
					if not self.Chemicals[smi].done:
						queue.put((smi, depth+1, path+[smi]))
				# if R.done:
				#     C.visit_count += R.visit_count
				#     R.visit_count += R.visit_count

		return leaves, pathway


	def update(self, chem_smi, pathway, depth=0):
		gamma=0.9#
		
		if depth == 0:
			for smi, template_id in pathway.items(): #leaf
				C = self.Chemicals[smi]
				R = C.reactions[template_id]
				# C.visit_count -= (VIRTUAL_LOSS - 1)
				# R.visit_count -= (VIRTUAL_LOSS - 1)
				# C.z = gamma * max([R.c_z for R in C.reactions])
				# C.scale_z = C.z

		if (chem_smi not in pathway) or (depth >= self.max_depth):
			return

		template_id = pathway[chem_smi]#in cooridinate start from self.smiles

		C = self.Chemicals[chem_smi]
		R = C.reactions[template_id]

		if R.waiting:
			return 

		if R.valid and (not R.done):

			R.done = all([self.Chemicals[smi].done for smi in R.reactant_smiles])

			for smi in R.reactant_smiles:
				self.update(smi, pathway, depth+1)
			

			# C.update_estimate_price(estimate_price)

			# for smi in R.reactant_smiles:
			#     self.Chemicals[smi].scale_z = sum([self.Chemicals[smi].z for smi in R.reactant_smiles])/(len(R.reactant_smiles)**2) 
			
			# R.c_z=sum([self.Chemicals[smi].scale_z for smi in R.reactant_smiles])

			# C.z = gamma * max([C.reactions[tid].c_z for tid in C.reactions])######flipped C.z and R.c_z compared to brand_new_3.py
			

			#Q = sum([self.Chemicals[smi].scale_v for smi in R.reactant_smiles])
		# ###########################################z3 important##################################################
		# 	z_list = [self.Chemicals[smi].z for smi in R.reactant_smiles]
		# 	if all([z != 0 for z in z_list]):

		# 		R.c_z=sum(z_list)/len(z_list)
		# 		if C.z < gamma * R.c_z  or C.z == 0:
		# 			C.z = gamma * R.c_z
		###############################################################3
			Q = sum([self.Chemicals[smi].forQ for smi in R.reactant_smiles])/(len(R.reactant_smiles)+0.01)
			R.update_Q(Q)
			C.update_forQ(Q)######


##################################modified price################################################
			price_list = [self.Chemicals[smi].price for smi in R.reactant_smiles]
			if all([price != -1 for price in price_list]):
				price = sum(price_list)
				R.price = price
				
				if R.price < C.price or C.price == -1:
					C.price = R.price
				
				# for smi in R.reactant_smiles:
				#     self.Chemicals[smi].z=1

				# for smi in R.reactant_smiles:
				#     self.Chemicals[smi].scale_z=sum([self.Chemicals[smi].z for smi in R.reactant_smiles])/(len(R.reactant_smiles)**2)


				
				   

		if len(C.reactions) >= self.max_branching:
			C.done = all([(R.done or (not R.valid)) for tid,R in C.reactions.items()])

		# if C.price != -1 and C.price < C.estimate_price:
		#   C.estimate_price = C.price


	def full_update(self, chem_smi, depth=0, path=[]):

		gamma=0.9###################

		C = self.Chemicals[chem_smi]
		C.pathway_count = 0

		if C.purchase_price != -1:
			C.pathway_count = 1
			return

		if depth > self.max_depth:
			return

		prefix = '    '* depth

		for template_id in C.reactions:
			R = C.reactions[template_id]
			R.pathway_count = 0
			if (R.waiting) or (not R.valid) or len(set(R.reactant_smiles) & set(path)) > 0:
				continue
			for smi in R.reactant_smiles:

				self.full_update(smi, depth+1, path+[chem_smi])
# ###########################################z3 important##################################################
# 			z_list = [self.Chemicals[smi].z for smi in R.reactant_smiles]
# 			if all([z != 0 for z in z_list]):

# 				R.c_z=sum(z_list)/len(z_list)
# 				if C.z < gamma * R.c_z  or C.z == 0:
# 					C.z = gamma * R.c_z
  #######################################z2########################################################33
			# z_list = [self.Chemicals[smi].z for smi in R.reactant_smiles]
			# if all([z != -1 for z in z_list]):

			# 	R.c_z=sum(z_list)/len(z_list)
			# 	if C.z < gamma * R.c_z  or C.z == -1:
			# 		C.z = gamma * R.c_z

 ########################################last version z1########################################################

			
			# R.c_z=sum([self.Chemicals[smi].z for smi in R.reactant_smiles])/len(R.reactant_smiles)
			# if C.z < gamma * R.c_z  or C.z == -1:
			# 	C.z = gamma * R.c_z

			
# ########################################################################################
# ########################################################################################
# 			for smi in R.reactant_smiles:
# 				self.Chemicals[smi].scale_z = sum([self.Chemicals[smi].z for smi in R.reactant_smiles])/(len(R.reactant_smiles)**2) 
			
# 			R.c_z=sum([self.Chemicals[smi].scale_z for smi in R.reactant_smiles])

# 			C.z = gamma * max([C.reactions[tid].c_z for tid in C.reactions])######flipped C.z and R.c_z compared to brand_new_3.py
 ################################################################################################33           
			price_list = [self.Chemicals[smi].price for smi in R.reactant_smiles]
			
			if all([price != -1 for price in price_list]):
				price = sum(price_list)
				
				R.price = price
			 
				if R.price < C.price or C.price == -1:
					C.price = R.price
					C.best_template = template_id
					
				R.pathway_count = np.prod([self.Chemicals[smi].pathway_count for smi in R.reactant_smiles])
				# if R.pathway_count != 0:
				#   print(prefix + '  Reac %d: '%template_id + str(R.reactant_smiles) + ' %d paths'%R.pathway_count)
				
			else:
				R.pathway_count = 0

			# print(prefix + str(R.reactant_smiles) + ' - %d' % R.pathway_count)

		C.pathway_count = sum([R.pathway_count for tid,R in C.reactions.items()])
		# if C.pathway_count != 0:
		#   print(prefix + chem_smi + ' %d paths, price: %.1f' % (C.pathway_count, C.price))


	def build_tree(self):

		self.running = True
		load_from_file = self.load_crn()
		
		p = Pool(1)
		# result = p.map(run_initial_topk, [self.smiles])
		# prob, value = result[0]
		result = p.map(run_initial_topk_prob, [self.smiles])
		indices, prob, value_1 = result[0]
		result_2 = p.map(run_initial_topk_val, [self.smiles])
		indices_2, prob_2, value = result_2[0]
		p.close()



	   
		#print(indices, prob, value)###########

		self.Chemicals[self.smiles] = Chemical(self.smiles)
		self.Chemicals[self.smiles].set_prob_value(indices, prob, value)
		# print(prob, value)

		for k in range(self.nproc):
			leaves = False
			leaf_counter = 0 
			leaves, pathway = self.select_leaf()###every time the visit count is counted, which is wrong. You need random branches at first, then paralelly expand.
			#print(leaves, pathway)############
			self.pathways[k] = pathway
			self.set_initial_target(k, leaves)
		
		# Coordinate workers.
		self.prepare()
		self.coordinate()
		
		self.full_update(self.smiles)
		C = self.Chemicals[self.smiles]

		print("Finished working.")
		print("=== find %d pathways" % C.pathway_count)
		print("=== time for fist pathway: %.2fs" % self.time_for_first_path)
		print("=== min price: %.1f" % C.price) ##
		#print("=== z_value: %.1f" % C.z)
		print("=== value: %.1f" % C.value)
		print("---------------------------")
		return #self.Chemicals, C.pathway_count, self.time_for_first_path # for training


	def tree_status(self):
		"""Summarize size of tree after expansion

		Returns:
			num_chemicals {int} -- number of chemical nodes in the tree
			num_reactions {int} -- number of reaction nodes in the tree
		"""

		num_chemicals = len(self.Chemicals)
		num_reactions = len(self.status)
		return (num_chemicals, num_reactions, [])


	# def reset(self):
	#     if self.celery:
	#         # general parameters in celery format
	#         pass
	#     else:
	#         self.workers = []
	#         self.coordinator = None
	#         self.running = False
			
	#         ## Queues 
	#         self.pathways = [0 for i in range(self.nproc)]
	#         self.pathway_count = 0 
	#         self.successful_pathway_count = 0
	#         self.mincost = 10000.0

	#     self.active_chemicals = [[] for x in range(self.nproc)]

	def get_buyable_paths(self, 
							smiles, 
							smiles_id,
							fileName = None, 
							max_depth=10, 
							expansion_time = 120,
							nproc=8, ngpus=1, mincount=25, chiral=True):

		self.reset()
		tf.reset_default_graph()

		self.smiles = smiles 
		self.smiles_id = smiles_id
		self.fileName = fileName 
		self.mincount = mincount
		self.max_depth = max_depth
		self.expansion_time = expansion_time
		self.nproc = nproc
		self.ngpus = ngpus

		self.manager = Manager()
		# specificly for python multiprocessing
		self.done = self.manager.Value('i', 0)
		# Keep track of idle workers
		self.idle = self.manager.list()
		self.workers = []#deletable
		self.coordinator = None#deletable
		self.running = False#deletable

		self.status = {}
			
		if not self.celery:
			for i in range(nproc):
				self.idle.append(True)
			if self.nproc != 1:
				self.expansion_queues = [Queue() for i in range(self.nproc)]
				self.results_queues   = [Queue() for i in range(self.nproc)]
			else:
				self.expansion_queues = [Queue()]
				self.results_queues   = [Queue()]
		self.active_chemicals = [[] for x in range(nproc)]

		self.time_for_first_path = -1

		print "Starting search for id:", smiles_id, "smiles:", smiles
		self.build_tree()
# ######################begining#######################################################################################################################
# 		def IDDFS():
# 			"""Perform an iterative deepening depth-first search to find buyable
# 			pathways.
						
# 			Yields:
# 				nested dictionaries defining synthesis trees
# 			"""
# 			for path in DLS_chem(self.smiles, depth=0, headNode=True):
# 				yield chem_dict(self.smiles, children=path, **{})


# 		def DLS_chem(chem_smi, depth, headNode=False):
# 			"""Expand at a fixed depth for the current node chem_id."""
# 			C = self.Chemicals[chem_smi]
# 			if C.purchase_price != -1:
# 				yield []

# 			if depth > self.max_depth:
# 				return

# 			for tid, R in C.reactions.items():
# 				if R.waiting or (not R.valid) or R.price == -1:
# 					continue
# 				rxn_smiles = '.'.join(sorted(R.reactant_smiles)) + '>>' + chem_smi
# 				for path in DLS_rxn(chem_smi, tid, depth):
# 					yield [rxn_dict(tid, rxn_smiles, children=path, **{})]


# 		def DLS_rxn(chem_smi, template_id, depth):
# 			"""Return children paths starting from a specific rxn_id"""
# 			C = self.Chemicals[chem_smi]
# 			R = C.reactions[template_id]

# 			rxn_list = []
# 			for smi in R.reactant_smiles:
# 				rxn_list.append([chem_dict(smi, children=path, **{}) for path in DLS_chem(smi, depth+1)])

# 			return itertools.product(rxn_list)

		
# ####################################new get unique trees#################################################
#  # Generate paths and ensure unique
# 		import hashlib
# 		import json
# 		done_trees = set()
# 		trees = []
# 		counter = 0
# 		for tree in IDDFS():
# 			hashkey = hashlib.sha1(json.dumps(
# 				tree, sort_keys=True).encode('utf-8')).hexdigest()

# 			if hashkey in done_trees:
# 				#print('Found duplicate tree...')
# 				continue

# 			done_trees.add(hashkey)
# 			trees.append(tree)
# 			counter += 1

# 			# if counter == max_trees:
# 			#     MyLogger.print_and_log('Generated {} trees (max_trees met), stopped looking for more...'.format(
# 			#         max_trees), treebuilder_loc)
# 			#     break

# 		tree_status = self.tree_status()
# 		if self.celery:
# 			self.reset()  # free up memory, don't hold tree
# 		return (tree_status, trees)

# #################################original count trees####################################################
# 		# trees = [tree for tree in IDDFS()]
# 		# return self.tree_status(), trees 
# ##########################################################################################

# #########################ending#######################################################################################################################

if __name__ == '__main__':

	import argparse

	random.seed(1)
	np.random.seed(1)
	tf.set_random_seed(1)

	parser = argparse.ArgumentParser(description='rl_mcts')
	parser.add_argument('--rl', action='store_true', default=True)#when training, change to True
	args = parser.parse_args()

	MyLogger.initialize_logFile()
	simulation_time =30. #30.
	
	smiles_id = 0
	smiles = "C1=CC(=C(C=C1F)F)C(CN2C=NC=N2)(CN3C=NC=N3)O"

	#Load tree builder 
	NCPUS = 1
	NGPUS = 1 
	print "There are {} processes available ... ".format(NCPUS)

	max_branching = 100#could be 1000, tune together with tempalte number

	Tree = MCTS(nproc=NCPUS,max_branching=max_branching, mincount=gc.RETRO_TRANSFORMS_CHIRAL['mincount'], 
		mincount_chiral=gc.RETRO_TRANSFORMS_CHIRAL['mincount_chiral'], solvent_evaluation = False)

	Tree.get_buyable_paths(smiles, smiles_id, 
										nproc=NCPUS, ngpus=NGPUS,
										expansion_time=simulation_time)

	# print(status)
	# #print(len(paths[0]))
	# #print(paths[0])
	# num_trees=0
	# for i in range(len(paths)):
	# 	num_trees=num_trees+len(paths[i])
	# print('number of trees', num_trees)
	# # for path in paths[:5]:
	# #     print(path)
	# print('Total num paths: {}'.format(len(paths)))
	#quit(1)
# # #######################################################################################################################
# #  ####################################################################################
# # #     ############################# TESTING ##############################################
# # #     ####################################################################################

######old test set##############
	# f = open(os.path.join(os.path.dirname(__file__), 'test_smiles.txt'))
	# N = 500
	# smiles_list = [line.strip().split('.')[0] for line in f]
######new test set##############	
	#f = open('makeit/data/test_set_reaxys.pkl','rb')
	f = open('makeit/data/cleaned_good_test_set_reaxys.pkl','rb')
	#f = open('makeit/data/test_set_comprised_of_seen_smiles_46000_158000.pkl','rb')
	smiles_list = pickle.load(f)
	f.close()
# #########Yujie's test set###################
# 	f = open('makeit/data/test_set_pick_out_from_Yujie_log_bak_norl.txt')
# 	smiles_list = [line.strip().split('.')[0] for line in f]
# 	f.close()
#############
	N = 1000#500
	random.shuffle(smiles_list)

	success = 0
	total = 0
	first_time = []
	pathway_count = []
	min_price = []

	# ########### STAGE 1 - PROCESS ALL CHEMICALS

	for _id, smiles in enumerate(smiles_list[:N]): 
		smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
		smiles_id=_id
		Tree.get_buyable_paths(smiles,smiles_id,
												nproc=NCPUS,
												expansion_time=simulation_time)

		Chemicals = Tree.Chemicals

		total += 1
		if Chemicals[smiles].price != -1:
			success += 1
				# first_time.append(ftime)
				# pathway_count.append(len(paths))
				# min_price.append(Chemicals[smiles].price)

		print('After looking at chemical index {}'.format(_id))
		print('Success ratio: %f (%d/%d)' % (float(success)/total, success, total)  )      
		# print('average time for first pathway: %f' % np.mean(first_time))
		# print('average number of pathways:     %f' % np.mean(pathway_count))
		# print('average minimum price:          %f' % np.mean(min_price))

# # ################################################Training#######################################################################
# 	simulation_time =120.
	
# 	fid = open('makeit/data/reaxys_limit1000000000_reaxys_v2_transforms_retro_v9_10_5.txt.data_pkl', 'rb')
# 	data = pickle.load(fid)
# 	#random.shuffle(data)
# 	smiles_list = []
# 	for r in data:
# 	  smiles_list.append(r[0])
# 	  smiles_list.append(r[1])

# 	smiles_list=[smis.strip().split('.')[0] for smis in smiles_list]
# 	#print (len(smiles_list))

# 	# f = open(os.path.join(os.path.dirname(__file__), 'test_smiles.txt'))
# 	# smiles_list = [line.strip() for line in f]
# 	# random.shuffle(smiles_list)

# 	success = 0
# 	total = 0

# 	#first_time = []
# 	#pathway_count = []
# 	#min_price = []

# 	BATCH_SIZE = 256

# 	for _id, smiles in enumerate(smiles_list[0:400]):   #used to be 1000# ["NC(=O)c1csc([N+](=O)[O-])c1"]: ['O=C(Cl)c1csc([N+](=O)[O-])c1']: #
# 	    # Chemicals, pcount, ftime = Tree.get_buyable_paths(smiles, _id, 
# 	    #                                     nproc=NCPUS,
# 	    #                                     expansion_time=simulation_time)



# 	    Tree.get_buyable_paths(smiles, _id, 
# 	                                        nproc=NCPUS,
# 	                                        expansion_time=simulation_time)

# 	    Chemicals=Tree.Chemicals##
# 	#    self.Chemicals, C.pathway_count, self.time_for_first_path
# 	#   pickle.dump(Chemicals, open('chemicals.pkl', 'wb'))

# 	# with open('chemicals.pkl', 'rb') as f:
# 	#   Chemicals = pickle.load(f)
# 	    total += 1
# 	    if Chemicals[smiles].price != -1:
# 	        success += 1
# 	        #first_time.append(ftime)
# 	        #pathway_count.append(pcount)
# 	        #min_price.append(Chemicals[smiles].price)
# 	    print('Success ratio: %f' % (float(success)/total))

# 	    if not args.rl:
# 	        continue

# 	    if Chemicals[smiles].price == -1:
# 	        continue
# #######################################################
# 	    smis = []
# 	    true_prob = []
# 	    true_value = []
# 	    tau=1####

# 	    for smi in Chemicals:
# 	        C = Chemicals[smi]
# 	        valid_reactions = [(tid,R) for tid,R in C.reactions.items() if R.valid and (not R.waiting)]
# 	        if len(valid_reactions) == 0:
# 	            continue
# 	        prob = np.zeros(163723)
# 	        product_visits = sum([(float (R.visit_count))**(1/tau) for tid,R in valid_reactions])######
# 	        if product_visits > 0:
# 	            for tid, R in valid_reactions:
# 	                prob[tid] = ((float(R.visit_count))**(1/tau)) / product_visits##########
# 	        smis.append(smi)
# 	        true_prob.append(prob)
# 	        #true_value.append(C.price)
# 	        true_value.append(C.z)

# ############################################################################################		
# 	    #print('Success ratio: %f' % (float(success)/total))
# 	    # break



# 	    if len(smis) > 20 * BATCH_SIZE:
# 	        smis = smis[-20*BATCH_SIZE:]
# 	        true_prob = true_prob[-20*BATCH_SIZE:]
# 	        true_value = true_value[-20*BATCH_SIZE:]

# 	    print('update')
# 	    print(smis)

# 	    if len(smis) < BATCH_SIZE:
# 	        _smis, _prob, _value = smis, true_prob, true_value
# 	    else:
# 	        index = np.random.choice(len(smis), BATCH_SIZE)
# 	        _smis = [smis[i] for i in index]
# 	        _prob = [true_prob[i] for i in index]
# 	        _value = [true_value[i] for i in index]

# 	    if len(smis) == 0:
# 	        continue

# 	    #print('true_prob', _prob)
# 	    #print('max true_prob', max(_prob))
# 	    print('true_value', _value)

# 	    # p = Pool(1)
# 	    # result = p.map(run_update_prob, [(_smis, np.array(_prob), np.array(_value).reshape((-1,1)))])
	    
# 	    # p.close()

# 	    p = Pool(1)
# 	    result_2 = p.map(run_update_val, [(_smis, np.array(_prob), np.array(_value).reshape((-1,1)))])
# 	    p.close()


# 	    #RL_MODEL_PATH = 'makeit/models/rl/'
# 	    #PRETRAIN_MODEL_PATH = RL_MODEL_PATH


# 	# print('average time for first pathway: %f' % np.mean(first_time))
# 	# print('average number of pathways:     %f' % np.mean(pathway_count))
# 	# print('average minimum price:          %f' % np.mean(min_price))
   
	
# 	# # if not args.rl:
# 	# #   f = open('test_smiles.txt', 'w')
# 	# #   for smiles in smiles_list[:1000]:
# 	# #       f.write(smiles + '\n')
# 	# #   f.close()

# 	# # for smi in Chemicals:
# 	# #   print(smi)
# 	# #   print(Chemicals[smi].price)
# 	# #   print([R.visit_count for tid, R in Chemicals[smi].reactions.items() if R.valid])
# 	# #   print('--------------------------------------------------')








