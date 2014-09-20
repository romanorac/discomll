import numpy as np
import Orange
import discomll
from discomll import dataset
path = "/".join(discomll.__file__.split("/")[:-2] + ["discomll", "datasets", ""])

def breastcancer_disc_orange(replication = 2):
	test_samples = 100
	data = Orange.data.Table("breast-cancer-wisconsin")
	train_data = data[:-test_samples]
	test_data = data[-test_samples:]
	
	for j in range(replication-1):
		for i in range(len(train_data)):
			train_data.append(train_data[i])
		for i in range(len(test_data)):
			test_data.append(test_data[i])

	return train_data, test_data

def breastcancer_cont_orange(replication = 2):
	test_samples = 100
	data = Orange.data.Table("breast-cancer-wisconsin-cont")
	train_data = data[:-test_samples]
	test_data = data[-test_samples:]
	
	for j in range(replication-1):
		for i in range(len(train_data)):
			train_data.append(train_data[i])
		for i in range(len(test_data)):
			test_data.append(test_data[i])

	return train_data, test_data

def breastcancer_disc_discomll(replication = 2):
	data_tag = ["test:breast_cancer_disc" for i in range(replication)]

	train_data = dataset.Data(data_tag = data_tag,
									data_type = "chunk",
									X_indices = xrange(1,10),
									X_meta = ["d" for i in range(9)],
									id_index = 0, 
									y_index = 10,
									delimiter = ",",
									y_map = ["2","4"])

	data_tag = ["test:breast_cancer_disc_test" for i in range(replication)]
	test_data = dataset.Data(data_tag = data_tag,
									data_type = "chunk",
									X_indices = xrange(1,10),
									X_meta = ["d" for i in range(9)],
									id_index = 0,
									y_index = 10,
									delimiter = ",",
									y_map = ["2","4"],
									missing_vals=["?"])
	
	return train_data, test_data

def breastcancer_cont_discomll(replication = 2):
	data_tag = ["test:breast_cancer_cont" for i in range(replication)]
	train_data = dataset.Data(data_tag = data_tag,
								data_type = "chunk",
								X_indices = xrange(0,9),
								X_meta = ["c" for i in range(9)],
								y_index = 9,
								delimiter = ",",
								y_map = ["benign", "malign"])

	data_tag = ["test:breast_cancer_cont_test" for i in range(replication)]
	test_data = dataset.Data(data_tag = data_tag,
								data_type = "chunk",
								X_indices = xrange(0,9),
								X_meta = ["c" for i in range(9)], 
								y_index = 9,
								delimiter = ",",
								y_map = ["benign", "malign"])
	
	return train_data, test_data

def breastcancer_cont(replication = 2):
	f = open(path + "breast_cancer_wisconsin_cont.txt", "r")
	data = np.loadtxt(f, delimiter = ",",dtype=np.string0)
	x_train = np.array(data[:,range(0,9)])
	y_train = np.array(data[:,9])
	for j in range(replication-1):
		x_train = np.vstack([x_train, data[:,range(0,9)]])	
		y_train  = np.hstack([y_train, data[:,9]])
	x_train = np.array(x_train, dtype=np.float)
	
	f = open(path +  "breast_cancer_wisconsin_cont_test.txt")
	data = np.loadtxt(f, delimiter = ",",dtype=np.string0)
	x_test = np.array(data[:,range(0,9)])
	y_test = np.array(data[:,9])
	for j in range(replication-1):
		x_test = np.vstack([x_test, data[:,range(0,9)]])	
		y_test  = np.hstack([y_test, data[:,9]])
	x_test = np.array(x_test, dtype=np.float)

	return x_train, y_train, x_test, y_test

def breastcancer_disc(replication = 2):
	f = open(path +  "breast_cancer_wisconsin_disc.txt")
	data = np.loadtxt(f, delimiter = ",")
	x_train = data[:,range(1,10)]
	y_train = data[:,10]
	for j in range(replication-1):
		x_train = np.vstack([x_train, data[:,range(1,10)]])
	 	y_train =  np.hstack([y_train, data[:,10]])
	
	f = open(path +  "breast_cancer_wisconsin_disc_test.txt")
	data = np.loadtxt(f, delimiter = ",")
	x_test = data[:,range(1,10)]
	y_test = data[:,10]
	for j in range(replication-1):
		x_test = np.vstack([x_test, data[:,range(1,10)]])
		y_test = np.hstack([y_test,data[:,10]])

	return x_train, y_train, x_test, y_test


def ex4_orange(replication = 2):
	f = open(path + "ex4.txt")
	data = np.loadtxt(f, delimiter = ",")

	features = [Orange.feature.Continuous("atr1"), Orange.feature.Continuous("atr2")]
	classattr = Orange.feature.Continuous("class")

	domain = Orange.data.Domain(features + [classattr])
	train_data = Orange.data.Table(domain)
	for j in range(replication):
		for row in data:
			train_data.append(row.tolist())
		
	return train_data

def ex4_discomll(replication = 2):

	data_tag = ["test:ex4" for i in range(replication)]
	data = dataset.Data(data_tag = data_tag,
						data_type = "chunk",
						X_indices = xrange(0,2),
						y_index = 2,
						y_map = ["0.0000000e+00","1.0000000e+00"])
	return data

def iris(replication = 2):
	f = open(path + "iris.txt")
	data = np.loadtxt(f, delimiter = ",", dtype=np.string0)
	x_train = np.array(data[:,range(0,4)], dtype=np.float)
	y_train  = data[:, 4]

	for j in range(replication-1):
		x_train = np.vstack([x_train, data[:,range(0,4)]])
	 	y_train =  np.hstack([y_train, data[:,4]])
	x_train = np.array(x_train, dtype=np.float)

	f = open(path + "iris_test.txt")
	data = np.loadtxt(f, delimiter = ",", dtype=np.string0)
	x_test = np.array(data[:,range(0,4)], dtype=np.float)
	y_test = data[:, 4]
	
	for j in range(replication-1):
		x_test = np.vstack([x_test, data[:,range(0,4)]])
		y_test = np.hstack([y_test, data[:,4]])
	x_test = np.array(x_test, dtype=np.float)

	return x_train, y_train, x_test, y_test

def iris_discomll(replication = 2):
	data_tag = ["test:iris" for i in range(replication)]	
	train_data = dataset.Data(data_tag = data_tag,
							data_type = "chunk",
							X_indices = xrange(0,4),
							X_meta = ["c" for i in xrange(0,4)], 
							y_index = 4,
							delimiter = ",")

	data_tag = ["test:iris_test" for i in range(replication)]
	
	test_data = dataset.Data(data_tag = data_tag,
							data_type = "chunk",
							X_indices = xrange(0,4), 
							X_meta = ["c" for i in xrange(0,4)],
							y_index = 4,
							delimiter = ",")

	return train_data, test_data

def regression_data():
	f = open(path + "regression_data1.txt")
	data = np.loadtxt(f, delimiter = ",")
	x1 = np.insert(data[:,0].reshape(len(data),1), 0, np.ones(len(data)), axis=1)
	y1 = data[:,1]
	f = open(path + "regression_data2.txt")
	data = np.loadtxt(f, delimiter = ",")
	x2 = np.insert(data[:,0].reshape(len(data),1), 0, np.ones(len(data)), axis=1)
	y2 = data[:,1]
	x1 = np.vstack((x1,x2))
	y1 = np.hstack((y1,y2))

	f = open(path + "regression_data_test1.txt")
	data = np.loadtxt(f, delimiter = ",")
	x1_test = np.insert(data[:,0].reshape(len(data),1), 0, np.ones(len(data)), axis=1)
	y1_test = data[:,1]
	f = open(path +  "regression_data_test2.txt")
	data = np.loadtxt(f, delimiter = ",")
	x2_test = np.insert(data[:,0].reshape(len(data),1), 0, np.ones(len(data)), axis=1)
	y2_test = data[:,1]
	x1_test = np.vstack((x1_test,x2_test))
	y1_test = np.hstack((y1_test,y2_test))
	return x1, y1, x1_test, y1_test

def regression_data_discomll():
	train =  dataset.Data(data_tag = ["test:regression_data1","test:regression_data2"],
				data_type = "chunk",
				id_index = 0, 
				X_indices = [0],
				X_meta = ["c"],
				y_index = 1)

	test = dataset.Data(data_tag = ["test:regression_data_test1","test:regression_data_test2"],
				data_type = "chunk",
				id_index = 0, 
				X_indices = [0],
				X_meta = ["c"],
				y_index = 1) 
	return train, test

def ex3(replication = 2):
	f = open(path +  "ex3.txt")
	train_data = np.loadtxt(f, delimiter = ",")
	f = open(path +  "ex3_test.txt")
	test_data = np.loadtxt(f, delimiter = ",")
	
	x_train = np.insert(train_data[:,(0,1)], 0, np.ones(len(train_data)), axis=1)
	y_train = train_data[:,2]
	x_test = np.insert(test_data[:,(0,1)], 0, np.ones(len(test_data)), axis=1)
	y_test = test_data[:,2]

	for i in range(replication-1):
		x_train = np.vstack((x_train, np.insert(train_data[:,(0,1)], 0, np.ones(len(train_data)), axis=1)))
		y_train = np.hstack((y_train, train_data[:,2]))

		x_test = np.vstack((x_test, np.insert(test_data[:,(0,1)], 0, np.ones(len(test_data)), axis=1)))
		y_test = np.hstack((y_test, test_data[:,2]))

	return x_train, y_train, x_test, y_test

def ex3_discomll(replication = 2):
	data_tag = ["test:ex3" for i in range(replication)]
	train_data = dataset.Data(data_tag = data_tag,
						data_type = "chunk", 
						X_indices = [0,1],
						y_index = 2)

	data_tag = ["test:ex3_test" for i in range(replication)]
	test_data = dataset.Data(data_tag = data_tag,
						data_type = "chunk",
						X_indices = [0,1],
						y_index = 2)

	return train_data, test_data


