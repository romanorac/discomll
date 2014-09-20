from discomll import dataset
from discomll.regression import linear_regression

train = dataset.Data(data_tag = ["http://ropot.ijs.si/data/linear/train/xaaaaa.gz", "http://ropot.ijs.si/data/linear/train/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            X_indices = range(1,21),
                            id_index = 0,
                            y_index = 21,
                            delimiter = ",")

test = dataset.Data(data_tag = ["http://ropot.ijs.si/data/linear/test/xaaaaa.gz","http://ropot.ijs.si/data/linear/test/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            X_indices = range(1,21),
                            id_index = 0,
                            y_index = 21,
                            delimiter = ",")

fit_model = linear_regression.fit(train)
predictions = linear_regression.predict(test, fit_model)
print predictions