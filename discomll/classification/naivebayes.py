"""
Naive Bayes with MapReduce

Algorithm calculates multinomial distribution for discrete features and Gaussian distribution for numerical features. The output of algorithm is consistent with implementation of Naive Bayes classifier in Orange and scikit-learn.

Reference:
MapReduce version of algorithm is proposed by Cheng-Tao Chu; Sang Kyun Kim, Yi-An Lin, YuanYuan Yu, Gary Bradski, Andrew Ng, and Kunle Olukotun. "Map-Reduce for Machine Learning on Multicore". NIPS 2006.   
"""

def simple_init(interface, params):
	return params

def map_fit(interface, state, label, inp):
	"""
	Function counts occurrences of feature values for every row in given data chunk. For continuous features it returns number of values and it calculates mean and variance for every feature. 
	For discrete features it counts occurrences of labels and values for every feature. It returns occurrences of pairs: label, feature index, feature values.  

	"""
	import numpy as np
	combiner = {} #combiner used for joining of intermediate pairs
	out = interface.output(0) #all outputted pairs have the same output label
	
	for row in inp: #for every row in data chunk
		row = row.strip().split(state["delimiter"]) #split row
		if len(row) > 1: #check if row is empty
			for i, j in enumerate(state["X_indices"]): #for defined features
				if row[j] not in state["missing_vals"]: #check missing values
					#creates a pair - label, feature index
					pair = row[state["y_index"]] + state["delimiter"] + str(j) 
					
					if state["X_meta"][i] == "c": #continuous features
						if pair in combiner:
							#convert to float and store value
							combiner[pair].append(np.float32(row[j])) 
						else:
							combiner[pair] = [np.float32(row[j])]    
					
					else: #discrete features
						#add feature value to pair
						pair += state["delimiter"]+row[j] 
						#increase counts of current pair
						combiner[pair] = combiner.get(pair, 0) + 1 
					
					#increase label counts
					combiner[row[state["y_index"]]] = combiner.get(row[state["y_index"]], 0) + 1
	
	for k, v in combiner.iteritems(): #all pairs in combiner are output
		if len(k.split(state["delimiter"])) == 2: #continous features  
			#number of elements, partial mean and variance
			out.add(k, (np.size(v), np.mean(v, dtype=np.float32), np.var(v, dtype=np.float32)))  
		else: #discrete features and labels
			out.add(k, v)
	
	
def reduce_fit(interface, state, label, inp):
	"Function separates aggregation of continuous and discrete features. For continuous features it aggregates partially calculated means and variances and returns them. For discrete features it aggregates pairs and returns them. Pairs with label occurrences are used to calculate prior probabilities"
	from disco.util import kvgroup #function for grouping values by key
	import numpy as np
	
	out = interface.output(0) #all outputted pairs have the same output label
	
	#model of naive Bayes stores label names, sum of all label occurrences and pairs (feature index, feature values) for discrete features which are needed to optimize predict phase. 
	fit_model = {"y_labels":[], "y_sum":0, "iv" : set()} 
	combiner = {} #combiner maintains correct order of means and variances.
	means, variances = [], []
	k_prev = ""
	
	for key, value in kvgroup(inp): #input pairs are sorted and grouped by key
		k_split = key.split(state["delimiter"])#pair is split
		
		if len(k_split) == 3: #discrete features
				#store pair (feature index, feature value)
				fit_model["iv"].add(tuple(k_split[1:])) 
				#aggregate and output occurrences of a pair
				out.add(tuple(k_split), sum(value)) 

		elif len(k_split) == 2: #continuous features
			
			#if label is different than previous. This enables calculation of all variances and means for every feature for current label. 
			if k_split[0] != k_prev and k_prev != "": 
				mean,var = zip(*[combiner[key] for key in sorted(combiner.keys())])
				means.append(mean)
				variances.append(var)
			
			#number of elements, partial mean, partial variance.
			n_a = mean_a = var_a =  0 
			#code aggregates partially calculated means and variances
			for n_b, mean_b, var_b in value:
				n_ab = n_a + n_b
				var_a = ((n_a * var_a + n_b * var_b)/float(n_ab)) + (n_a * n_b * ((mean_b - mean_a) / float(n_ab))**2)
				mean_a = (n_a * mean_a + n_b * mean_b)/float(n_ab)
				n_a = n_ab
			
			#maintains correct order of statistics for every feature
			combiner[int(k_split[1])] = (mean_a , var_a + 1e-9) 
			k_prev = k_split[0]
		
		else: #aggregates label occurrences
			fit_model[key] = np.sum(value)
			fit_model["y_sum"] += fit_model[key] #sum of all label occurrences
			fit_model["y_labels"].append(key)
	
	#if statistics for continuous features were not output in last iteration
	if len(means) > 0: 
		mean,var = zip(*[combiner[key] for key in sorted(combiner.keys())])
		out.add("mean", np.array(means + [mean], dtype=np.float32))
		variances = np.array(variances + [var], dtype=np.float32)
		out.add("var", variances)
		out.add("var_log", np.log(np.pi * variances))	

	#calculation of prior probabilities
	prior = [fit_model[y_label]/float(fit_model["y_sum"]) for y_label in fit_model["y_labels"]]
	out.add("prior", np.array(prior, dtype=np.float32))
	out.add("prior_log", np.log(prior))
	out.add("iv",list(fit_model["iv"]))
	out.add("y_labels", fit_model["y_labels"])
	

def map_predict(interface, state, label, inp):
	"""
	Function makes a predictions of samples with given model. It calculates probabilities with multinomial and Gaussian distribution.
	"""
	import numpy as np
	out = interface.output(0)
	
	continuous = [j for i,j in enumerate(state["X_indices"]) if state["X_meta"][i] == "c"] #indices of continuous features 
	discrete = [j for i,j in enumerate(state["X_indices"]) if state["X_meta"][i] == "d"] #indices of discrete features  

	cont = True if len(continuous) > 0 else False #enables calculation of Gaussian probabilities
	disc = True if len(discrete) > 0 else False #enables calculation of multinomial probabilities.

	for row in inp:
		row = row.strip().split(state["delimiter"])
		if len(row) > 1: #if row is empty 
			#set id of a sample
			x_id = "" if state["id_index"] == -1 else row[state["id_index"]] 
			#initialize prior probability for all labels	
			probs = state["fit_model"]["prior_log"]	
			
			if cont: #continuous features
				x = np.array([(0 if row[j] in state["missing_vals"] else float(row[j])) for j in continuous]) #sets selected features of the sample
				#Gaussian distribution
				probs = probs - 0.5 * np.sum(np.true_divide((x - state["fit_model"]["mean"])**2, state["fit_model"]["var"]) + state["fit_model"]["var_log"], axis=1)
			
			if disc: #discrete features
				#multinomial distribution
				probs = probs + np.sum([(0 if row[i] in state["missing_vals"] else state["fit_model"].get((str(i), row[i]), np.zeros(1))) for i in discrete], axis = 0)

			# normalize by P(x) = P(f_1, ..., f_n)
			log_prob_x = np.log(np.sum(np.exp(probs)))
			probs = np.exp(np.array(probs) - log_prob_x)
			#Predicted label is the one with highest probability
			y_predicted = max(zip(probs, state["fit_model"]["y_labels"]))[1]
			out.add(x_id, (y_predicted, probs.tolist()))

def fit(input, save_results = True, show = False):
	"""
	Function builds a model for Naive Bayes. It executes multiple map functions and one reduce function which aggregates intermediate results and returns a model.

	Parameters
	----------
	input - dataset object with input urls and other parameters
	save_results - save results to ddfs
	show - show info about job execution

	Returns
	-------
	Urls of fit model results on ddfs
	"""
	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job
	
	#define a job and set save of results to ddfs
	job = Job(worker = Worker(save_results = save_results)) 
	
	#job parallelizes mappers, sorts intermediate pairs and joins them with one reducer 
	job.pipeline = [
	("split", Stage("map",input_chain = input.params["input_chain"], init = simple_init, process = map_fit)),
	('group_all', Stage("reduce", init = simple_init, process = reduce_fit, sort = True, combine = True))]

	job.params = input.params #job parameters (dataset object)
	#define name of a job and input data urls
	job.run(name = "naivebayes_fit", input = input.params["data_tag"]) 
	fitmodel_url =  job.wait(show = show)
	return {"naivebayes_fitmodel": fitmodel_url} #return results url


def predict(input, fitmodel_url, m = 1, save_results = True, show = False):
	"""
	Function starts a job that makes predictions to input data with a given model 

	Parameters
	----------
	input - dataset object with input urls and other parameters
	fitmodel_url - model created in fit phase
	m - m estimate is used with discrete features
	save_results - save results to ddfs
	show - show info about job execution

	Returns
	-------
	Urls of predictions on ddfs  
	"""
	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job, result_iterator
	import numpy as np
	
	try:
		m = float(m)
	except ValueError:
		raise Exception("Parameter m should be numerical.")

	if "naivebayes_fitmodel" in fitmodel_url:
		#fit model is loaded from ddfs
		fit_model = dict((k,v) for k,v in result_iterator(fitmodel_url["naivebayes_fitmodel"]))
		if len(fit_model["y_labels"]) < 2:
			print "There is only one class in training data."
			return []
	else:
		raise Exception("Incorrect fit model.")
		
	if input.params["X_meta"].count("d") > 0: #if there are discrete features in the model
		#code calculates logarithms to optimize predict phase as opposed to calculation by every mapped.
		np.seterr(divide='ignore')
		for iv in fit_model["iv"]:
			dist = [fit_model.pop((y,) + iv, 0) for y in fit_model["y_labels"]]
			fit_model[iv] = np.nan_to_num(np.log(np.true_divide(np.array(dist) + m * fit_model["prior"], np.sum(dist) + m ))) - fit_model["prior_log"]
		del(fit_model["iv"])

	#define a job and set save of results to ddfs
	job = Job(worker = Worker(save_results = save_results))
	
	#job parallelizes execution of mappers
	job.pipeline = [
	("split", Stage("map",input_chain = input.params["input_chain"], init = simple_init, process = map_predict))]

	job.params = input.params #job parameters (dataset object)
	job.params["fit_model"] = fit_model
	#define name of a job and input data urls
	job.run(name = "naivebayes_predict", input = input.params["data_tag"]) 
	results = job.wait(show = show)
	return results




















