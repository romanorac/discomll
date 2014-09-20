from discomll import dataset
from discomll.ensemble import decision_trees
from discomll.utils import model_view
from disco.core import result_iterator

train = dataset.Data(data_tag = [["http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"]],         
                    X_indices = xrange(0,4),
                    X_meta = "http://ropot.ijs.si/data/datasets_meta/iris_meta.csv",
                    y_index = 4,
                    delimiter = ",") 

fit_model = decision_trees.fit(train, max_tree_nodes = 50, leaf_min_inst = 5, class_majority = 1, measure = "info_gain", split_fun = "equal_freq", split_intervals = 100)

print model_view.output_model(fit_model)

#predict training dataset
predictions = decision_trees.predict(train, fit_model) 

#output results
for k,v in result_iterator(predictions):
    print k, v[0]



