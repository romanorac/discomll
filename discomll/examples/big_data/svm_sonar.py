from discomll import dataset
from discomll.classification import linear_proximal_svm

train = dataset.Data(data_tag = ["http://ropot.ijs.si/data/sonar/train/xaaaaa.gz", "http://ropot.ijs.si/data/sonar/train/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            X_indices = range(1,61),
                            id_index = 0,
                            y_index = 61,
                            X_meta = ["c" for i in range(1,61)],
                            y_map = ["R","M"],
                            delimiter = ",")

test = dataset.Data(data_tag = ["http://ropot.ijs.si/data/sonar/test/xaaaaa.gz","http://ropot.ijs.si/data/sonar/test/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            X_indices = range(1,61),
                            id_index = 0,
                            y_index = 61,
                            X_meta = ["c" for i in range(1,61)],
                            y_map = ["R","M"],
                            delimiter = ",")

fit_model = linear_proximal_svm.fit(train)
predictions = linear_proximal_svm.predict(test, fit_model)
print predictions