from discomll import dataset
from discomll.ensemble import decision_trees

train = dataset.Data(data_tag = ["http://ropot.ijs.si/data/segmentation/train/xaaaaa.gz","http://ropot.ijs.si/data/segmentation/train/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            X_indices = range(2,21),
                            id_index = 0,
                            y_index = 1,
                            X_meta = ["c" for i in range(2,21)],
                            delimiter = ",")

test = dataset.Data(data_tag = ["http://ropot.ijs.si/data/segmentation/test/xaaaaa.gz","http://ropot.ijs.si/data/segmentation/test/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            X_indices = range(2,21),
                            id_index = 0,
                            y_index = 1,
                            X_meta = ["c" for i in range(2,21)],
                            delimiter = ",")

fit_model = decision_trees.fit(train, max_tree_nodes = 50, leaf_min_inst = 5, class_majority=1)
predict_url = decision_trees.predict(test, fit_model)
print predict_url



























