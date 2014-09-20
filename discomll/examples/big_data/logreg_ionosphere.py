from discomll import dataset
from discomll.classification import logistic_regression

train = dataset.Data(data_tag = ["http://ropot.ijs.si/data/ionosphere/train/xaaaaa.gz", "http://ropot.ijs.si/data/ionosphere/train/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            id_index = 0,
                            X_indices = range(1,35),
                            X_meta = ["c" for i in range(1,35)], 
                            y_index = 35,
                            delimiter = ",",
                            y_map = ["b","g"])

test = dataset.Data(data_tag = ["http://ropot.ijs.si/data/ionosphere/test/xaaaaa.gz","http://ropot.ijs.si/data/ionosphere/test/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            id_index = 0,
                            X_indices = range(1,35),
                            X_meta = ["c" for i in range(1,35)], 
                            y_index = 35,
                            delimiter = ",",
                            y_map = ["b","g"])

fit_model = logistic_regression.fit(train, max_iterations= 18, alpha = 1)
predictions = logistic_regression.predict(test, fit_model)
print predictions