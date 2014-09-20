from discomll import dataset
from discomll.regression import locally_weighted_linear_regression

train = dataset.Data(data_tag = ["http://ropot.ijs.si/data/fraction/train/xaaaaa.gz", "http://ropot.ijs.si/data/fraction/train/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            X_indices = range(1,14),
                            id_index = 0,
                            y_index = 14,
                            delimiter = ",")

test = dataset.Data(data_tag = [["http://ropot.ijs.si/data/fraction/test/xaaaaa.gz"]],
                            data_type = "gzip",
                            X_indices = range(1,14),
                            id_index = 0,
                            y_index = 14,
                            delimiter = ",")

predictions = locally_weighted_linear_regression.fit_predict(train, test, num_to_predict = 2, tau = 10)
print predictions









