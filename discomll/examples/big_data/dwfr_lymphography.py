from discomll import dataset
from discomll.ensemble import distributed_weighted_forest_rand

train = dataset.Data(data_tag=["http://ropot.ijs.si/data/lymphography/train/xaaaaa.gz",
                               "http://ropot.ijs.si/data/lymphography/train/xaaabj.gz"],
                     data_type="gzip",
                     generate_urls=True,
                     X_indices=range(2, 20),
                     id_index=0,
                     y_index=1,
                     X_meta=["d", "d", "d", "d", "d", "d", "d", "d", "c", "c", "d", "d", "d", "d", "d", "d", "d", "c"],
                     delimiter=",")

test = dataset.Data(data_tag=["http://ropot.ijs.si/data/lymphography/test/xaaaaa.gz",
                              "http://ropot.ijs.si/data/lymphography/test/xaaabj.gz"],
                    data_type="gzip",
                    generate_urls=True,
                    X_indices=range(2, 20),
                    id_index=0,
                    y_index=1,
                    X_meta=["d", "d", "d", "d", "d", "d", "d", "d", "c", "c", "d", "d", "d", "d", "d", "d", "d", "c"],
                    delimiter=",")

fit_model = distributed_weighted_forest_rand.fit(train, trees_per_chunk=3, max_tree_nodes=50, min_samples_leaf=10,
                                                 min_samples_split=5, class_majority=1, measure="info_gain",
                                                 num_medoids=10, accuracy=1, separate_max=True, random_state=None,
                                                 save_results=True)
predict_url = distributed_weighted_forest_rand.predict(test, fit_model)
print predict_url
