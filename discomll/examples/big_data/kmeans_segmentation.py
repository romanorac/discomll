from discomll import dataset
from discomll.clustering import kmeans

train = dataset.Data(data_tag = ["http://ropot.ijs.si/data/segmentation/train/xaaaaa.gz","http://ropot.ijs.si/data/segmentation/train/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            X_indices = range(2,21),
                            id_index = 0,
                            X_meta = ["c" for i in range(2,21)],
                            delimiter = ",")

test = dataset.Data(data_tag = ["http://ropot.ijs.si/data/segmentation/test/xaaaaa.gz","http://ropot.ijs.si/data/segmentation/test/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            X_indices = range(2,21),
                            id_index = 0,
                            X_meta = ["c" for i in range(2,21)],
                            delimiter = ",")


#fit_model = kmeans.fit(train, n_clusters = 7, max_iterations = 10, random_state = 0)
fit_model = ["tag://disco:results:kmeans_iter_10@57f:2a0a9:a7d64"]
predictions = kmeans.predict(test, fit_model)
print predictions
