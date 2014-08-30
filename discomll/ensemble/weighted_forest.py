
def simple_init(interface, params):
	return params

def map_init(interface, params):
    """Intialize random number generator with given seed `params.seed`."""
    import numpy as np
    import random
    
    np.random.seed(params['seed'])
    random.seed(params['seed'])
    return params

def map_fit(interface, state, label, inp):
	import numpy as np
	from itertools import permutations
	import decision_tree, measures, k_medoids

	out = interface.output(0)
	similarity_mat = {}
	x, y, y_mapping, forest, margins = [], [], [], [], []
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
	
	for t in range(state["trees_per_chunk"]):
		bag_indices = np.random.randint(len(x), size=(len(x)))
		unique = set(bag_indices)
		out_of_bag_indices = [i for i in range(len(x)) if i not in unique][:500]

		tree = decision_tree.fit(
			x = x[bag_indices], 
			y = y[bag_indices], 
			t = state["X_meta"], 
			randomized = True, 
			max_tree_nodes = state["max_tree_nodes"], 
			leaf_min_inst = state["leaf_min_inst"], 
			class_majority = state["class_majority"],
			intervals = state["intervals"],
			measure = measures.info_gain if state["measure"] == "info_gain" else measures.mdl,
			split_fun = measures.equal_freq_splits if state["split_fun"] == "equal_freq" else measures.random_splits)
		
		#calculate margins
		tree_margins, leafs_grouping = {}, {}
		for j in out_of_bag_indices:
			leaf, margin = decision_tree.predict(tree, x[j], y[j])
			tree_margins[j] = margin
			if leaf in leafs_grouping:
				leafs_grouping[leaf].append(j)
			else:
				leafs_grouping[leaf] = [j]	
		margins.append(tree_margins)
		
		for k, v in leafs_grouping.iteritems():
			for cx, cy in permutations(v,2): 
				if cx in similarity_mat:
					similarity_mat[cx][cy] = similarity_mat[cx].get(cy, 0) - 1
				else:
					similarity_mat[cx] = {cy: -1}
		
		tree_mapped = {}
		for k,v in tree.iteritems():
			tree_mapped[k] = [None for i in range(2)]	
			for i, node in enumerate(v):
				dist_map = dict([(y_mapping[label],freq) for label, freq in node[3].iteritems()])
				split_map = set([mapping[node[1]][int(s)] for s in list(node[2])]) if node[5] == "d" else node[2]
				tree_mapped[k][i] = (node[0], node[1], split_map, dist_map, node[4],node[5])
		forest.append(tree_mapped)
	
	min_elements = []
	for k, v in similarity_mat.iteritems():
		min_id = min(similarity_mat[k], key = similarity_mat[k].get) 
		min_elements.append((similarity_mat[k][min_id], min_id))
	min_elements = sorted(min_elements)
	
	if state["k"] == "sqrt":
		k = int(np.sqrt(len(x[0]))) + 1
	elif state["k"] == "square": 
		k = len(np.unique(y)) * len(np.unique(y))
	
	cidx = set()
	counter = 0
	while counter < len(min_elements) and len(cidx) < k:
		cidx.add(min_elements[counter][1])
		counter += 1
	
	inds, medoids_i = k_medoids.fit(similarity_mat, len(x), list(cidx))
	sample_ids = np.array(similarity_mat.keys())
	medoids_i = [sample_ids[i] for i in medoids_i]

	clusters = [sample_ids[np.where(inds == i)[0]] for i in np.unique(inds)]
	medoids = x[medoids_i].tolist() #set medoids without sample identifier

	cont, disc = [], []
	for i in range(len(medoids)):
		cont.append([medoids[i][j] for j in range(len(medoids[i])) if state["X_meta"][j] == "c"])
		disc.append([mapping[j][int(medoids[i][j])] for j in range(len(medoids[i])) if state["X_meta"][j] == "d"])
	medoids = [np.array(cont), np.array(disc)]

	stats = [[] for i in range(len(medoids_i))] 
	for i in range(len(forest)): #for every tree in forest
		for num, cluster in enumerate(clusters):
			#calculate average margin for cluster
			values = [margins[i][sample_id] for sample_id in cluster if int(sample_id) in margins[i]]
			if values != []:
				avg = np.average(values)
				forest[i]["margin" + str(num)] = avg
				stats[num].append(avg)
			
	stats = [np.median(value) for value in stats]
	gower_range = np.array([np.ptp(x[:,i]) for i in range(len(state["X_meta"])) if state["X_meta"][i] == "c"])
	gower_range[gower_range == 0] = 1e-9
	out.add("model", (forest, medoids, stats, gower_range))


def reduce_fit(interface, state, label, inp):	
	out = interface.output(0)
	out.add("X_names", state["X_names"])
	out.add("X_meta", state["X_meta"])
	for i, (_, value) in enumerate(inp):
		out.add((i+1), value)

def map_predict(interface, state, label, inp):
	import decision_tree
	import numpy as np
	
	out = interface.output(0)
	
	for row in inp:
		row = row.strip().split(state["delimiter"])
		if len(row) > 1:
			x_id = "" if state["id_index"] == -1 else row[state["id_index"]]
			
			x, cont, disc = [], [],[]
			for i,j in enumerate(state["X_indices"]):
				if state["X_meta"][i] == "c":
					x.append(float(row[j]))
					cont.append(float(row[j]))
				else:
					x.append(row[j])
					disc.append(row[j])
			cont, disc = np.array(cont), np.array(disc)

			similarities = []
			for i, medoids in enumerate(state["medoids"]):
				gower = 0 if len(cont) == 0 else np.sum(1 - np.true_divide(np.abs(cont - medoids[0]), state["gower_ranges"][i]), axis = 1)
				gower += 0 if len(disc) == 0 else np.sum(disc == medoids[1], axis = 1)
				similarities += zip(np.round(1 - gower/float(len(x)), 4), [(i,j) for j in range(len(x))])

			similarities = sorted(similarities)
			similar_medoids = [similarities[0][1]]
			for sim in similarities[1:]:
				if similarities[0][0] == sim[0]:
					similar_medoids.append(sim[1])
				else:
					break

			global_predictions = {}
			for i,j in similar_medoids:
				predictions = {}
				margin = "margin"+str(j)
				for tree in state["forest"][i]:
					if margin in tree and tree[margin] >= state["stats"][i][j]:
						pred = decision_tree.predict(tree, x)
						predictions[pred] = predictions.get(pred, []) + [tree[margin]]

				
				for k, v in predictions.iteritems(): 
					predictions[k] = np.average(v) * len(v)
				
				max_pred = max(predictions, key = predictions.get)
				if max_pred not in global_predictions:
					global_predictions[max_pred] = predictions[max_pred]
				elif predictions[max_pred] > global_predictions[max_pred]:
					global_predictions[max_pred] = predictions[max_pred]

			out.add(x_id, (max(global_predictions, key = global_predictions.get),))			
	
def fit(input, trees_per_chunk = 50, max_tree_nodes = 50, leaf_min_inst = 5, class_majority = 1, k = "sqrt", measure = "info_gain", split_fun = "equal_freq", split_intervals = 100, random_state = None, save_results = True, show = False):
	
	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job
	import discomll
	
	path = "/".join(discomll.__file__.split("/")[:-1] + ["ensemble", "core",""])

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

	job = Job(worker = Worker(save_results = save_results))
	job.pipeline = [
	("split", Stage("map",input_chain = input.params["input_chain"], init = map_init, process = map_fit)),
	('group_all', Stage("reduce", init = simple_init, process = reduce_fit, combine = True))]

	job.params = input.params
	job.params["trees_per_chunk"] = trees_per_chunk
	job.params["max_tree_nodes"] = max_tree_nodes
	job.params["leaf_min_inst"] = leaf_min_inst
	job.params["class_majority"] = class_majority
	job.params["measure"] = measure
	job.params["split_fun"] = split_fun
	job.params["intervals"] = split_intervals
	job.params["k"] = k
	job.params['seed'] = random_state

	job.run(name = "weighted_forest_fit", input = input.params["data_tag"], required_files =[path+"decision_tree.py", path+"measures.py", path + "k_medoids.py"])
	
	fitmodel_url =  job.wait(show = show)
	return {"wrf_fitmodel": fitmodel_url} #return results url



def predict(input, fitmodel_url, save_results = True, show = False):
	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job, result_iterator
	import discomll
	path = "/".join(discomll.__file__.split("/")[:-1] + ["ensemble", "core",""])

	job = Job(worker = Worker(save_results = save_results))
	job.pipeline = [("split", Stage("map",input_chain = input.params["input_chain"], init = simple_init, process = map_predict))]

	if "wrf_fitmodel" not in fitmodel_url:
		raise Exception("Incorrect fit model.")
    
	job.params = input.params
	fit_model = [v for k, v in result_iterator(fitmodel_url["wrf_fitmodel"]) if k not in ["X_names", "X_meta"]]
	job.params["forest"] = [e[0] for e in fit_model]
	job.params["medoids"] = [e[1] for e in fit_model]
	job.params["stats"] = [e[2] for e in fit_model]
	job.params["gower_ranges"] = [e[3] for e in fit_model]

	job.run(name = "weighted_forest_predict", input = input.params["data_tag"], required_files = [path+"decision_tree.py"])
	
	return job.wait(show = show)
























