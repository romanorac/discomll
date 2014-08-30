"""
Logistic regression with MapReduce

Algorithm builds a model with continuous features and predicts binary target variable (1, 0). Learning is done by fitting theta parameters to the training data where the likelihood function is optimized by using Newton-Raphson to update theta parameters. The output of algorithm is consistent with implementation of logistic regression classifier in Orange.

Reference:
MapReduce version of algorithm is proposed by Cheng-Tao Chu; Sang Kyun Kim, Yi-An Lin, YuanYuan Yu, Gary Bradski, Andrew Ng, and Kunle Olukotun. "Map-Reduce for Machine Learning on Multicore". NIPS 2006.   
"""

def simple_init(interface, params):
	return params

def map_fit(interface, state, label, inp):
	"""
	Function calculates sigmoid function (g) for every sample. With g it calculates part of Hessian matrix and gradient, aggregates and output them. It also calculates J function which is needed for checking the convergence of parameters theta. 
	"""
	import numpy as np
	out = interface.output(0)

	H, J, grad = 0, 0, 0
	for row in inp:
		row = row.strip().split(state["delimiter"]) #split row
		if len(row) > 1: #check if row is empty
			#add intercept term to every sample
			x = np.array([1] + [(0 if v in state["missing_vals"] else float(v)) for i, v in enumerate(row) if i in state["X_indices"]])
			#map label value to 0 or 1. If label does not match set error 
			y = 0 if state["y_map"][0] == row[state["y_index"]] else 1 if state["y_map"][1] == row[state["y_index"]] else "Error"
			
			g = 1./(1 + np.exp(-np.dot(x, state["thetas"])))#sigmoid function
			grad += x * (g - y) #gradient
			H += np.multiply(np.outer(x, x), g * (1 - g))
			#H += np.outer(x, x) * g * (1 - g) #Hessian matrix  
			J -= np.log(g) if y == 1 else np.log(1 - g) #J cost function
	out.add("grad", grad)
	out.add("J", J)
	for i, row in enumerate(H):
		out.add(i, row)

def reduce_fit(interface, state, label, inp):
	import numpy as np
	out = interface.output(0)
	J, grad = 0,0
	H = [0 for i in range(len(state["X_indices"])+1)]
	
	for k, v in inp:
		if k == "grad": 
			grad += v #aggregate gradient
		elif k == "J": 
			J += v #aggregate J cost function
		else:
			H[k] += v #aggregate Hessian matrix
	
	#calculate new values of thetas with current thetas, Hessian and gradient
	out.add("thetas", state["thetas"] - np.linalg.lstsq(H, grad)[0])
	out.add("J", J) 

def map_predict(interface, state, label, inp):
	import numpy as np
	out = interface.output(0)

	for row in inp:
		row = row.strip().split(state["delimiter"])
		if len(row) > 1:
			#set id of current sample	
			x_id = "" if state["id_index"] == -1 else row[state["id_index"]]
			#add intercept term
			x = np.array([1] + [(0 if v in state["missing_vals"] else float(v)) for i, v in enumerate(row) if i in state["X_indices"]])
			#calculate probability with sigmoid function
			prob = 1./(1 + np.exp(-np.dot(x, state["thetas"])))
			probs = [1 - prob, prob] #probability for label 0 and 1
			#select label with highest probability
			y = max(zip(probs, state["y_map"]))[1]
			out.add(x_id, (y, probs))


def fit(input, alpha = 1e-8, max_iterations = 10, save_results = True, show = False):
	"""
	Function starts a job for calculation of theta parameters

	Parameters
	----------
	input - dataset object with input urls and other parameters
	alpha - convergence value
	max_iterations - define maximum number of iterations 
	save_results - save results to ddfs
	show - show info about job execution

	Returns
	-------
	Urls of fit model results on ddfs
	"""
	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job,result_iterator
	import numpy as np

	if input.params["y_map"] == []:
		raise Exception("Logistic regression requires a target label mapping parameter.")
	try:
		alpha = float(alpha)
		max_iterations = int(max_iterations)
		if max_iterations < 1:
			raise Exception("Parameter max_iterations should be greater than 0.")
	except ValueError:
		raise Exception("Parameters should be numerical.")

	#initialize thetas to 0 and add intercept term
	thetas = np.zeros(len(input.params["X_indices"]) + 1)
	
	J = [0] #J cost function values for every iteration
	for i in range(max_iterations):
		job = Job(worker = Worker(save_results = save_results))
		#job parallelizes mappers and joins them with one reducer 
		job.pipeline = [
		("split", Stage("map",input_chain = input.params["input_chain"], init = simple_init, process = map_fit)),
		('group_all', Stage("reduce", init = simple_init, process = reduce_fit, combine = True))]
		
		job.params = input.params #job parameters (dataset object)
		job.params["thetas"] = thetas #every iteration set new thetas
		job.run(name = "logreg_fit_iter_%d"%(i+1), input = input.params["data_tag"])

		fitmodel_url = job.wait(show = show)
		for k,v in result_iterator(fitmodel_url): 
			if k == "J": #
				J.append(v) #save value of J cost function
			else:
				thetas = v #save new thetas
		if np.abs(J[-2] - J[-1]) < alpha: #check for convergence
			if show:
				print("Converged at iteration %d" % (i+1))
			break

	return {"logreg_fitmodel": fitmodel_url} #return results url


def predict(input, fitmodel_url, save_results = True, show = False):
	"""
	Function starts a job that makes predictions to input data with a given model

	Parameters
	----------
	input - dataset object with input urls and other parameters
	fitmodel_url - model created in fit phase
	save_results - save results to ddfs
	show - show info about job execution

	Returns
	-------
	Urls with predictions on ddfs
	"""
	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job, result_iterator

	if input.params["y_map"] == []:
		raise Exception("Logistic regression requires a target label mapping parameter.")
	if "logreg_fitmodel" not in fitmodel_url:
		raise Exception("Incorrect fit model.")

	job = Job(worker = Worker(save_results = save_results))
	#job parallelizes execution of mappers
	job.pipeline = [
	("split", Stage("map",input_chain = input.params["input_chain"], init = simple_init, process = map_predict))]
	
	job.params = input.params #job parameters (dataset object)
	job.params["thetas"] = [v for k,v in result_iterator(fitmodel_url["logreg_fitmodel"]) if k == "thetas"][0] #thetas are loaded from ddfs

	job.run(name = "logreg_predict", input = input.params["data_tag"])
	results = job.wait(show = show)
	return results


					   


























