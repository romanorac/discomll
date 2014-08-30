def simple_init(interface, params):
	return params

def map_fit(interface, state, label, inp):
	combiner = {} 
	file_name = str(inp.input)
	out = interface.output(0) 

	for row in inp: #for every row in data chunk
		row = row.strip().split(state["delimiter"]) #split row
		if len(row) > 1: #check if row is empty
			y = row[state["y_index"]]
			combiner[y] = combiner.get(y, 0) + 1
	
	suma = sum(combiner.values())
	for k,v in combiner.iteritems():
		combiner[k] = round(v/float(suma),4)
	
	out.add(file_name, combiner)
	
def reduce_fit(interface, state, label, inp):
	out = interface.output(0) #all outputted pairs have the same output label
	
	for key, value in inp: 
		out.add(key, value)

def measure(input, save_results = True, show = False):
	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job
	
	#define a job and set save of results to ddfs
	job = Job(worker = Worker(save_results = save_results)) 
	
	job.pipeline = [
	("split", Stage("map",input_chain = input.params["input_chain"], init = simple_init, process = map_fit)),
	('group_all', Stage("reduce", init = simple_init, process = reduce_fit, combine = True))]

	job.params = input.params #job parameters (dataset object)
	
	job.run(name = "Distribution", input = input.params["data_tag"]) 
	return job.wait(show = show) #return results url


















