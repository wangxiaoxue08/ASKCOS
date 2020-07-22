from collections import OrderedDict
class Chemical(object):
	"""Represents a chemical compound."""

	def __init__(self, chem_smi):
		# Initialize lists for incoming and outgoing reactions.
		self.smiles = chem_smi
		self.reactions = OrderedDict()
		self.purchase_price = -1
		self.visit_count = 0
		self.if_initial_round = True

		# # Counter param used for the DFS search. 
		# self.estimate_price = -1         # estimated min cost
		# self.estimate_price_sum = 0.
		# self.estimate_price_cnt = 0
		self.forQ = 0##to be changed
		self.forQ_sum = 0.#
		self.forQ_cnt = 0#


		self.s_Q_sum = 0.#used to be 0.
		self.s_Q_cnt = 0#used to be 0
		self.s_Q = 1.#used to be 0.

		self.best_template = -1
		self.best_solvent_template = -1

		self.prob = None
		self.value = 0        # output of vaue network
		self.indices = []
		self.burnable_indices_reverse = []
		self.scale_v = 0 ###
		# self.sorted_id = None

		self.price = -1           # valid min cost
		self.z = 0 ###
		self.scale_z = 0###
		self.done = False

		#self.steps = -1#new 06212019, initially -1

		self.pathway_count = 0

	def __repr__(self):
		return "%s(%r)" % (self.__class__, self.smiles)

	def __str__(self):

		return "%s" % self.smiles

	def set_price(self, ppg_value):
		try:
			ppg = float(ppg_value)
			if ppg > 0:
				ppg = 1.
				self.purchase_price = ppg
				self.price = ppg
			#self.estimate_price = ppg
				self.z = 1
				self.s_Q = 0.
				self.done = True
				self.value = 1 #newly added 03272020
				#self.value = ppg #new 07252019
				#self.steps = 0 # new 06212019
		except:
			pass

	def set_prob_value(self, indices, prob, value):##################
		self.prob = prob
		# self.sorted_id = prob.argsort()[::-1]
		self.indices=indices
		self.value = value
		self.burnable_indices_reverse = indices[::-1]

		# self.update_Q(value)

	# def update_forQ(self, scale_value):
	#     self.forQ_sum += scale_value
	#     self.forQ_cnt += 1
	#     self.forQ = self.forQ_sum / self.forQ_cnt 
	def update_forQ(self, scale_value):
		self.forQ_sum += scale_value
		self.forQ_cnt += 1
		self.forQ = self.forQ_sum / self.forQ_cnt#self.visit_count
   
	def update_s_Q(self, scale_value):#not used
		self.s_Q_sum += scale_value
		self.s_Q_cnt += 1
		self.s_Q = self.s_Q_sum / self.s_Q_cnt#self.visit_count
	def reset(self):
		return

class Reaction(object):
	"""Represents a reaction."""

	def __init__(self, smiles, template_id):
		"""Initialize entry."""
		self.smiles = smiles.strip()
		self.template_id = template_id
		# self.depth  = depth 
		self.valid = True
		self.reactant_smiles = []
		self.visit_count = 0

		self.waiting = True
		self.done = False

		self.Q = 0.##to be changed
		self.Q_sum = 0.#
		self.Q_cnt = 0#

		self.s_Q = -1.##to be changed -1 to 1, used to be 0.
		self.s_Q_sum = 0.#
		self.s_Q_cnt = 0#
		self.c_z = 0.
		
		self.for_s_Q = -1.
		self.for_s_Q_sum = 0.#
		self.for_s_Q_cnt = 0#
		self.slvs_list = []

		self.price = -1
		

		self.pathway_count = 0

	def __repr__(self):
		return "%s(%r)" % (self.__class__, self.smiles)

	def __str__(self):
		return "%s" % self.smiles

	# def update_Q(self, scale_value):
	#     self.Q_sum += scale_value
	#     self.Q_cnt += 1
	#     self.Q = self.Q_sum / self.Q_cnt  

	def update_Q(self, scale_value):
		self.Q_sum += scale_value
		self.Q_cnt += 1
		self.Q = self.Q_sum / self.Q_cnt 

	def update_s_Q(self, scale_value):#not used
		self.s_Q_sum += scale_value
		self.s_Q_cnt += 1
		self.s_Q = self.s_Q_sum / self.s_Q_cnt#self.visit_count

	def update_for_s_Q(self, scale_value, summation_c_steps):##not used
		self.for_s_Q_sum += scale_value
		self.for_s_Q_cnt += 1
		adjusted_s_Q_sum = self.for_s_Q_sum / self.for_s_Q_cnt
		self.for_s_Q = (self.s_Q + adjusted_s_Q_sum)/(1+summation_c_steps)#self.visit_count

	def reset(self):
		return
		# self.price = -1 
		# self.Q = 0.1#
		# self.c_z = -1

