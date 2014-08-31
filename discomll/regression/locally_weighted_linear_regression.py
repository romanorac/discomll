"""
Locally weighted linear regression with MapReduce

Reference:
MapReduce version of algorithm is proposed by Cheng-Tao Chu; Sang Kyun Kim, Yi-An Lin, YuanYuan Yu, Gary Bradski, Andrew Ng, and Kunle Olukotun. "Map-Reduce for Machine Learning on Multicore". NIPS 2006.   

"""


def simple_init(interface, params):
	return params

def map_fit(interface, state, label, inp):
	import numpy as np
	combiner = dict([(k, [0,0]) for k in state["samples"].keys()])
	out = interface.output(0)

	for row in inp:
		row = row.strip().split(state["delimiter"])
		if len(row) > 1:
			x = np.array([1] + [(0 if v in state["missing_vals"] else float(v)) for i, v in enumerate(row) if i in state["X_indices"]])
			y = float(row[state["y_index"]])

			for test_id, x1 in state["samples"].iteritems():
				w = np.exp(np.dot(-(x1 - x).transpose(),x1 - x)/state["tau"])
				combiner[test_id][0] += w * np.outer(x, x)
				combiner[test_id][1] += w * x * y

	for k, v in combiner.iteritems():
		out.add(k+state["delimiter"]+"b", v[1])
		for i in range(len(v[0])):
			out.add(k+state["delimiter"]+"A"+state["delimiter"]+ str(i), v[0][i])
		
		

def map_predict(interface, state, label, inp):
	import numpy as np
	out = interface.output(0)

	for row in inp:
		row = row.strip().split(state["delimiter"])
		if len(row) > 1:
			x = np.array([1] + [(0 if v in state["missing_vals"] else float(v)) for i, v in enumerate(row) if i in state["X_indices"]])
			out.add(row[state["id_index"]], x)

def reduce_fit(interface, state, label, inp):
	import numpy as np
	from disco.util import kvgroup
	
	out = interface.output(0)
	A = [0 for i in range(len(state["X_indices"])+1)]
	for k, v in kvgroup(inp):
		ksplit = k.split(state["delimiter"])
		if ksplit[1] == "A":
			A[int(ksplit[2])] = np.sum(v)
		else:			
			b = np.sum(v)
			thetas = np.linalg.lstsq(A,b)[0]
			out.add(ksplit[0], (np.dot(state["samples"][ksplit[0]],thetas), thetas.tolist())) 
			A = [0 for i in range(len(state["X_indices"])+1)]

def _fit_predict(fit_data, samples, tau, save_results, show):
	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job
	job = Job(worker = Worker(save_results = save_results))
	
	job.pipeline = [
	("split", Stage("map",input_chain = fit_data.params["input_chain"], init = simple_init, process = map_fit)),
	('group_all', Stage("reduce", init = simple_init, process = reduce_fit, sort = True ,combine = True))]

	job.params = fit_data.params
	job.params["tau"] = tau
	job.params["samples"] = samples

	job.run(name = "lwlr_fit_predict", input = fit_data.params["data_tag"])
	return job.wait(show = show)

def fit_predict(training_data, fitting_data, tau = 1, samples_per_job = 0, save_results = True, show = False):
	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job, result_iterator
	import numpy as np
	from disco.core import Disco

	try:
		tau = float(tau)
		if tau <= 0:
			raise Exception("Parameter tau should be >= 0.")  
	except ValueError:
		raise Exception("Parameter tau should be numerical.")
		
	if fitting_data.params["id_index"] == -1:
		raise Exception("Predict data should have id_index set.")
		return {}
 
	job = Job(worker = Worker(save_results = save_results))
	job.pipeline = [
	("split", Stage("map",input_chain = fitting_data.params["input_chain"], init = simple_init, process = map_predict))]
	job.params = fitting_data.params
	job.run(name = "lwlr_read_data", input = fitting_data.params["data_tag"])
	
	samples = {}
	results = []
	tau = float(2*tau**2)
	counter = 0
	
	for test_id, x in result_iterator(job.wait(show = show)):
		if samples_per_job == 0:
			if len(x) <= 100:
				samples_per_job = 100
			else:
				samples_per_job= len(x) * -25/900. + 53

		samples[test_id] = x
		if counter == samples_per_job:
			print counter, len(x)
			results.append(_fit_predict(training_data, samples, tau, save_results, show))
			
			samples = {}
		counter+=1

	if len(samples) > 0:
		results.append(_fit_predict(training_data, samples, tau, save_results, show))

	ddfs = Disco().ddfs
	results = [list(ddfs.blobs(tag))[0] for tag in results] 
	return results


































