"""
Random forest with MapReduce

Fit phase
Random forest algorithm builds multiple decision trees with a bootstrap method on a subset of data. In each tree node, it estimates sqrt(num. of attributes)+1 randomly selected attributes (without replacement). All decision trees are merged in large ensemble.  

Predict phase
Algorithm queries as many trees as needed for reliable prediction. Firstly, it randomly chooses without replacement 15 trees. If all trees vote for the same class, it outputs prediction. If there are multiple classes predicted, it chooses 15 trees again. Algorithm calculates difference in probability between most and second most probable prediction. If difference is greater than parameter diff, it outputs prediction. If a test sample is hard to predict (difference is never higher than diff), it queries whole ensemble to make a prediction.

Reference
Similar algorithm is proposed in: Justin D Basilico, M Arthur Munson, Tamara G Kolda, Kevin R Dixon, and W Philip Kegelmeyer. Comet: A recipe for learning and using large ensembles on massive data. 

"""

def simple_init(interface, params):
	return params

def map_init(interface, params):
    """Intialize random number generator with given seed `params.seed`."""
    import random
    import numpy as np
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    return params

def map_fit(interface, state, label, inp):
	import numpy as np
	import decision_tree, measures
	
	out = interface.output(0)
	x, y, y_mapping = [], [], []
	mapping = [[] for i in range(len(state["X_meta"]))]
	 
	for row in inp:
		row = row.strip().split(state["delimiter"])
		if len(row) > 1:
			new_row = []
			for i, j in enumerate(state["X_indices"]):
				if state["X_meta"][i] == "c":
					new_row.append(float(row[j]))
				else:
					if row[j] not in mapping[i]:
						mapping[i].append(row[j])
					new_row.append(mapping[i].index(row[j]))
			x.append(new_row)
			if row[state["y_index"]] not in y_mapping:
				y_mapping.append(row[state["y_index"]])	
			y.append(y_mapping.index(row[state["y_index"]]))
	
	x,y = np.array(x), np.array(y)

	for i in range(state["trees_per_chunk"]):
		bag_indices = np.random.randint(len(x), size=(len(x)))
		
		tree = decision_tree.fit(
			x = x[bag_indices], 
			y = y[bag_indices], 
			t = state["X_meta"], 
			randomized = True, 
			max_tree_nodes = state["max_tree_nodes"], 
			leaf_min_inst = state["leaf_min_inst"], 
			class_majority = state["class_majority"],
			intervals =  state["intervals"],
			measure = measures.info_gain if state["measure"] == "info_gain" else measures.mdl,
			split_fun = measures.equal_freq_splits if state["split_fun"] == "equal_freq" else measures.random_splits)

		tree_mapped = {}
		for k,v in tree.iteritems():
			tree_mapped[k] = [None for i in range(2)]	
			for i, node in enumerate(v):
				dist_map = dict([(y_mapping[label],freq) for label, freq in node[3].iteritems()])
				split_map = set([mapping[node[1]][int(s)] for s in list(node[2])]) if node[5] == "d" else node[2]
				tree_mapped[k][i] = (node[0], node[1], split_map, dist_map, node[4],node[5])
		
		out.add("tree", tree_mapped)
	
def reduce_fit(interface, state, label, inp):	
	out = interface.output(0)
	out.add("X_names", state["X_names"])
	for i, (_, value) in enumerate(inp):
		out.add((i+1), value)

def map_predict(interface, state, label, inp):
	import decision_tree
	import numpy as np
	import random
	
	out = interface.output(0)
	for row in inp:
		row = row.strip().split(state["delimiter"])
		if len(row) > 1:
			x_id = "" if state["id_index"] == -1 else row[state["id_index"]]
			x = [(float(row[j]) if state["X_meta"][i] == "c" else row[j]) for i,j in enumerate(state["X_indices"])]
			
			tallies = {}
			predicted = False
			querry = random.sample(range(len(state["forest"])), len(state["forest"]))
			
			for i, j in enumerate(querry):
				pred = decision_tree.predict(state["forest"][j], x)
				tallies[pred] = tallies.get(pred, 0) + 1

				if (i+1) % 15 == 0:
					if len(tallies) == 1:
						out.add(x_id, (pred, 1, i+1))
						predicted = True
					else:
						probs = sorted(zip(np.true_divide(tallies.values(), i+1), tallies.keys()), reverse = True)
						diff = 1 - probs[1][0]/float(probs[0][0])
						if diff > state["diff"]:
							out.add(x_id, (probs[0][1], diff, i+1))
							predicted = True

				if predicted == True:
					break
			if predicted == False:
				out.add(x_id, (max(tallies, key=tallies.get), 0, i+1))

def fit(input, trees_per_chunk = 50, max_tree_nodes = 50, leaf_min_inst = 5, class_majority = 1, measure = "info_gain", split_fun = "equal_freq", split_intervals = 100, random_state = None, save_results = True, show = False):

	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job
	import discomll
	path = "/".join(discomll.__file__.split("/")[:-1] + ["ensemble", "core",""])

	job = Job(worker = Worker(save_results = save_results))

	job.pipeline = [
	("split", Stage("map",input_chain = input.params["input_chain"], init = map_init, process = map_fit)),
	('group_all', Stage("reduce", init = simple_init, process = reduce_fit, combine = True))]

	try:
		trees_per_chunk = int(trees_per_chunk)
		max_tree_nodes = int(max_tree_nodes)
		leaf_min_inst = int(leaf_min_inst)
		class_majority = float(class_majority)
		split_intervals = int(split_intervals)

		if trees_per_chunk <= 0 or max_tree_nodes <= 0 or leaf_min_inst <= 0 or class_majority <= 0 or split_intervals <= 0:
			raise Exception("Parameters should be greater than 0.")  
	except ValueError:
		raise Exception("Parameters should be numerical.")

	if measure not in ["info_gain", "mdl"]:
		raise Exception("measure should be set to info_gain or mdl.")
	if split_fun not in ["equal_freq", "random"]:
		raise Exception("split_fun should be set to equal_freq or random.")

	job.params = input.params
	job.params["trees_per_chunk"] = trees_per_chunk
	job.params["max_tree_nodes"] = max_tree_nodes
	job.params["leaf_min_inst"] = leaf_min_inst
	job.params["class_majority"] = class_majority
	job.params["measure"] = measure
	job.params["split_fun"] = split_fun
	job.params["intervals"] = split_intervals
	job.params['seed'] = random_state

	job.run(name = "random_forest_fit", input = input.params["data_tag"], required_files =[path+"decision_tree.py", path+"measures.py"])
	
	fitmodel_url =  job.wait(show = show)
	return {"rf_fitmodel": fitmodel_url} #return results url

def predict(input, fitmodel_url, diff = 0.3, random_state = None, save_results = True, show = False):
	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job, result_iterator
	import discomll
	path = "/".join(discomll.__file__.split("/")[:-1] + ["ensemble", "core",""])

	try:
		diff = float(diff)
		if diff < 0:
			raise Exception("Parameter diff should be >= 0.")  
	except ValueError:
		raise Exception("Parameter diff should be numerical.")

	if "rf_fitmodel" not in fitmodel_url:
		raise Exception("Incorrect fit model.")

	job = Job(worker = Worker(save_results = save_results))
	job.pipeline = [("split", Stage("map",input_chain = input.params["input_chain"], init = map_init, process = map_predict))]

	job.params = input.params
	job.params["forest"] = [v for k, v in result_iterator(fitmodel_url["rf_fitmodel"]) if k != "X_names"]
	job.params["diff"] = diff
	job.params['seed'] = random_state

	job.run(name = "random_forest_predict", input = input.params["data_tag"], required_files = [path+"decision_tree.py"])
	
	return job.wait(show = show)


























