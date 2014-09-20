
def chunk_testdata():
	import discomll 
	from disco import ddfs
	path = "/".join(discomll.__file__.split("/")[:-2] + ["discomll", "datasets", ""])

	tags_chunk = ["test:breast_cancer_cont", "test:breast_cancer_cont_test", "test:breast_cancer_disc", "test:breast_cancer_disc_test", "test:ex3", "test:ex3_test", "test:ex4", "test:iris", "test:iris_test","test:regression_data1","test:regression_data2", "test:regression_data_test1","test:regression_data_test2"]
	
	filenames_chunk = ["breast_cancer_wisconsin_cont.txt", "breast_cancer_wisconsin_cont_test.txt", "breast_cancer_wisconsin_disc.txt", "breast_cancer_wisconsin_disc_test.txt", "ex3.txt", "ex3_test.txt", "ex4.txt", "iris.txt", "iris_test.txt", "regression_data1.txt","regression_data2.txt","regression_data_test1.txt","regression_data_test2.txt"]
	
	ddfs = ddfs.DDFS()
	for i in range(len(tags_chunk)):
		f = open(path + filenames_chunk[i], "r")
		print f.name
		ddfs.chunk(tags_chunk[i], [f.name])















