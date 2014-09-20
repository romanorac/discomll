from discomll import dataset
from discomll.ensemble import random_forest
from discomll.utils import model_view
from disco.core import result_iterator
from discomll.utils import accuracy

train = dataset.Data(data_tag = [["http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"]],
					id_index = 0,         
                    X_indices = xrange(1,10),
                    X_meta = "http://ropot.ijs.si/data/datasets_meta/breastcancer_meta.csv",
                    y_index = 10,
                    delimiter = ",") 

fit_model = random_forest.fit(train, trees_per_chunk = 50, max_tree_nodes = 50, leaf_min_inst = 5, class_majority = 1, measure = "info_gain", split_fun = "equal_freq", split_intervals = 100)
print model_view.output_model(fit_model)

#predict training dataset
predictions = random_forest.predict(train, fit_model) 

#output results
for k,v in result_iterator(predictions):
    print k, v[0]

#measure accuracy
ca = accuracy.measure(train, predictions)
for k,v in result_iterator(ca):
    print k, v

