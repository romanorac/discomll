from discomll import dataset
from discomll.clustering import kmeans


train = dataset.Data(data_tag = ["http://ropot.ijs.si/data/linear/train/xaaaaa.gz", "http://ropot.ijs.si/data/linear/train/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            X_indices = range(1,22),
                            id_index = 0,
                            delimiter = ",")

test = dataset.Data(data_tag = ["http://ropot.ijs.si/data/linear/test/xaaaaa.gz","http://ropot.ijs.si/data/linear/test/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            X_indices = range(1,22),
                            id_index = 0,
                            delimiter = ",")

fit_model = kmeans.fit(train, n_clusters = 5, max_iterations = 10, random_state = 0)
predictions = kmeans.predict(test, fit_model)
print predictions
