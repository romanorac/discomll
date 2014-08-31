"""
Decision trees with MapReduce

Fit phase
Decision trees algorithm builds one decision tree on a subset of data and it estimates all attributes in every tree node.

Predict phase
Each tree votes and algorithm selects prediction with most votes.

Reference
Similar algorithm is proposed in Gongqing Wu, Haiguang Li, Xuegang Hu, Yuanjun Bi, Jing Zhang, and Xindong Wu. MRec4.5: C4. 5 ensemble classification with mapreduce.
"""

def simple_init(interface, params):
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

	tree = decision_tree.fit(
		x = np.array(x), 
		y = np.array(y), 
		t = state["X_meta"], 
		randomized = False, 
		max_tree_nodes = state["max_tree_nodes"], 
		leaf_min_inst = state["leaf_min_inst"], 
		class_majority = state["class_majority"],
		intervals = state["intervals"], 
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
	for i, (key, value) in enumerate(inp):
		out.add(key+" "+str(i+1), value)

def map_predict(interface, state, label, inp):
	import numpy as np
	import decision_tree
	
	out = interface.output(0)
	half_ensemble = round(len(state["forest"])/2.)
	
	for row in inp:
		row = row.strip().split(state["delimiter"])
		if len(row) > 1:
			x_id = "" if state["id_index"] == -1 else row[state["id_index"]]
			x = [(float(row[j]) if state["X_meta"][i] == "c" else row[j]) for i,j in enumerate(state["X_indices"])]
			
			predictions = {}
			for i, tree in enumerate(state["forest"]):
				pred = decision_tree.predict(tree, x)
				predictions[pred] = predictions.get(pred, 0) + 1 
				
				if i >= half_ensemble-1:
					prediction = max(predictions, key=predictions.get)
					value = predictions[prediction]
					if value == half_ensemble:
						break
			out.add(x_id, (prediction, value))


def fit(input, max_tree_nodes = 50, leaf_min_inst = 5, class_majority = 1, measure = "info_gain", split_fun = "equal_freq", split_intervals = 100,  save_results = True, show = False):
	
	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job
	import discomll
	path = "/".join(discomll.__file__.split("/")[:-1] + ["ensemble", "core",""])

	try:
		max_tree_nodes = int(max_tree_nodes)
		leaf_min_inst = int(leaf_min_inst)
		class_majority = float(class_majority)
		split_intervals = int(split_intervals)
		if max_tree_nodes <= 0 or leaf_min_inst <= 0 or class_majority <= 0 or split_intervals <= 0:
			raise Exception("Parameters should be greater than 0.")  
	except ValueError:
		raise Exception("Parameters should be numerical.")

	if measure not in ["info_gain", "mdl"]:
		raise Exception("measure should be set to info_gain or mdl.")
	if split_fun not in ["equal_freq", "random"]:
		raise Exception("split_fun should be set to equal_freq or random.")



	job = Job(worker = Worker(save_results = save_results))
	job.pipeline = [
	("split", Stage("map",input_chain = input.params["input_chain"], init = simple_init, process = map_fit)),
	('group_all', Stage("reduce", init = simple_init, process = reduce_fit, combine = True))]

	job.params = input.params
	job.params["max_tree_nodes"] = max_tree_nodes
	job.params["leaf_min_inst"] = leaf_min_inst
	job.params["class_majority"] = class_majority
	job.params["measure"] = measure
	job.params["split_fun"] = split_fun
	job.params["intervals"] = split_intervals

	job.run(name = "decision_trees_fit", input = input.params["data_tag"], required_files =[path+"decision_tree.py", path+"measures.py"])
	
	fitmodel_url =  job.wait(show = show)
	return {"dt_fitmodel": fitmodel_url} #return results url

def predict(input, fitmodel_url, save_results = True, show = False):
	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job, result_iterator
	import discomll
	path = "/".join(discomll.__file__.split("/")[:-1] + ["ensemble", "core",""])

	if "dt_fitmodel" not in fitmodel_url:
		raise Exception("Incorrect fit model.")

	job = Job(worker = Worker(save_results = save_results))
	job.pipeline = [("split", Stage("map",input_chain = input.params["input_chain"], init = simple_init, process = map_predict))]

	job.params = input.params
	job.params["forest"] = [v for k, v in result_iterator(fitmodel_url["dt_fitmodel"]) if k != "X_names"]
	

	job.run(name = "decision_trees_predict", input = input.params["data_tag"], required_files = [path+"decision_tree.py"])
	
	return job.wait(show = show)


























