from discomll import dataset
from discomll.regression import locally_weighted_linear_regression
from discomll.utils import accuracy
from disco.core import result_iterator


training_data = dataset.Data(data_tag = ["test:regression_data1","test:regression_data2"],
			data_type = "chunk",
			id_index = 0, 
			X_indices = [0],
			y_index = 1)

fitting_data = dataset.Data(data_tag = ["test:regression_data_test1","test:regression_data_test2"],
			data_type = "chunk",
			id_index = 0, 
			X_indices = [0],
			y_index = 1) 
 
#fit fitting data to training data
results = locally_weighted_linear_regression.fit_predict(training_data, fitting_data, tau = 10)

#output results
for k,v in result_iterator(results):
	print k,v


