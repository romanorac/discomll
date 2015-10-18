from disco.core import result_iterator

from discomll import dataset
from discomll.ensemble import forest_distributed_decision_trees
from discomll.utils import model_view

train = dataset.Data(data_tag=[["http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"]],
                     X_indices=xrange(0, 4),
                     X_meta="http://ropot.ijs.si/data/datasets_meta/iris_meta.csv",
                     y_index=4,
                     delimiter=",")

fit_model = forest_distributed_decision_trees.fit(train, trees_per_chunk=1, bootstrap=False, max_tree_nodes=50,
                                                  min_samples_leaf=2, min_samples_split=1, class_majority=1,
                                                  separate_max=True, measure="info_gain", accuracy=1, random_state=None,
                                                  save_results=True)

print model_view.output_model(fit_model)

# predict training dataset
predictions = forest_distributed_decision_trees.predict(train, fit_model)

# output results
for k, v in result_iterator(predictions):
    print k, v[0]
