

def simple_init(interface, params):
	return params

def map_predictions(interface, state, label, inp):
	out = interface.output(0) #all outputted pairs have the same output label
	
	for row in inp: #for every row in data chunk	
		out.add(row[0], row[1][0])

def map_test_data(interface, state, label, inp):
	out = interface.output(0) #all outputted pairs have the same output label

	for row in inp: #for every row in data chunk
		if type(row) == str:
			row = row.strip().split(state["delimiter"]) #split row
			if len(row) > 1: #check if row is empty
				out.add(row[state["id_index"]], (row[state["y_index"]],))
			
	
def reduce_ca(interface, state, label, inp):
	from disco.util import kvgroup #function for grouping values by key
	out = interface.output(0) #all outputted pairs have the same output label
	
	number_of_samples, correct_predictions = 0, 0
	
	for key, value in kvgroup(inp): #input pairs are sorted and grouped by key
		value = list(value)
		if value[0] == value[1]:
			correct_predictions += 1
		number_of_samples += 1

	out.add("CA", correct_predictions/float(number_of_samples))

def reduce_mse(interface, state, label, inp):
	from disco.util import kvgroup #function for grouping values by key
	out = interface.output(0) #all outputted pairs have the same output label
	
	number_of_samples, mse = 0, 0
	
	for key, value in kvgroup(inp): #input pairs are sorted and grouped by key
		value = list(value)
		mse += (float(value[0]) - float(value[1]))**2
		number_of_samples += 1

	out.add("MSE", mse/float(number_of_samples))
	

def measure(test_data, predictions, measure = "ca" , save_results = True, show = False):
	from disco.worker.pipeline.worker import Worker, Stage
	from disco.core import Job, result_iterator
	from disco.worker.task_io import task_input_stream, chain_reader
	
	if measure not in ["ca", "mse"]:
		raise Exception("measure should be ca or mse.")
	if test_data.params["id_index"] == -1:
		raise Exception("ID index should be defined.")

	if predictions == []:
		return "No predictions", None

	#define a job and set save of results to ddfs
	job = Job(worker = Worker(save_results = save_results)) 
	
	job = Job(worker = Worker(save_results = save_results)) 
	job.pipeline = [("split", Stage("map",input_chain = test_data.params["input_chain"], init = simple_init, process = map_test_data))]
	
	job.params = test_data.params 
	job.run(name = "ma_parse_testdata", input = test_data.params["data_tag"]) 
	parsed_testdata = job.wait(show = show)

	reduce_proces = reduce_ca if measure == "ca" else reduce_mse

	job = Job(worker = Worker(save_results = save_results)) 
	job.pipeline = [("split", Stage("map", init = simple_init, input_chain = [task_input_stream,chain_reader], process = map_predictions)),
	('group_all', Stage("reduce", init = simple_init, process = reduce_proces, sort = True, combine = True))]

	job.run(name = "ma_measure_accuracy", input =  parsed_testdata + predictions) 

	measure, acc = [(measure, acc) for measure, acc in result_iterator(job.wait(show = show))][0]
	return  measure, acc
















