from discomll import dataset
from discomll.ensemble import random_forest


train = dataset.Data(data_tag = ["seg_train1","seg_train2"],#[["http://ropot.ijs.si/data/segmentation/train/xaaaaa.gz"]],
                            data_type = "chunk",
                            #generate_urls = True,
                            X_indices = range(2,21),
                            id_index = 0,
                            y_index = 1,
                            X_meta = ["c" for i in range(2,21)],
                            delimiter = ",")

test = dataset.Data(data_tag = ["seg_test1","seg_test2"],#[["http://ropot.ijs.si/data/segmentation/test/xaaaaa.gz"]],#,"http://ropot.ijs.si/data/segmentation/test/xaaabj.gz"],
                            data_type = "chunk",
                            #generate_urls = True,
                            X_indices = range(2,21),
                            id_index = 0,
                            y_index = 1,
                            X_meta = ["c" for i in range(2,21)],
                            delimiter = ",")

fit_model = random_forest.fit(train, trees_per_chunk = 50, max_tree_nodes = 50, leaf_min_inst = 5, class_majority = 1)

#fit_model = {"rf_fitmodel":["tag://disco:results:rand om_forest_fit@58f:3398f:f1dca"]}
predictions = random_forest.predict(test, fit_model, diff = 1)
print predictions

