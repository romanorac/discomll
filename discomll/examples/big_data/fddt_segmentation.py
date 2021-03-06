from discomll import dataset
from discomll.ensemble import forest_distributed_decision_trees

train = dataset.Data(data_tag=["http://ropot.ijs.si/data/segmentation/train/xaaaaa.gz",
                               "http://ropot.ijs.si/data/segmentation/train/xaaabj.gz"],
                     data_type="gzip",
                     generate_urls=True,
                     X_indices=range(2, 21),
                     id_index=0,
                     y_index=1,
                     X_meta=["c" for i in range(2, 21)],
                     delimiter=",")

test = dataset.Data(data_tag=["http://ropot.ijs.si/data/segmentation/test/xaaaaa.gz",
                              "http://ropot.ijs.si/data/segmentation/test/xaaabj.gz"],
                    data_type="gzip",
                    generate_urls=True,
                    X_indices=range(2, 21),
                    id_index=0,
                    y_index=1,
                    X_meta=["c" for i in range(2, 21)],
                    delimiter=",")

fit_model = forest_distributed_decision_trees.fit(train, trees_per_chunk=1, bootstrap=True, max_tree_nodes=50,
                                                  min_samples_leaf=10, min_samples_split=5, class_majority=1,
                                                  separate_max=True, measure="info_gain", accuracy=1, random_state=None,
                                                  save_results=True)
predict_url = forest_distributed_decision_trees.predict(test, fit_model)
print predict_url
