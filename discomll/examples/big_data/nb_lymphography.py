from discomll import dataset
from discomll.classification import naivebayes

train = dataset.Data(data_tag = ["http://ropot.ijs.si/data/lymphography/train/xaaaaa.gz", "http://ropot.ijs.si/data/lymphography/train/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            X_indices = range(2,20),
                            id_index = 0,
                            y_index = 1,
                            X_meta = ["d","d","d","d","d","d","d","d","c","c","d","d","d","d","d","d","d","c"],
                            delimiter = ",")

test = dataset.Data(data_tag = ["http://ropot.ijs.si/data/lymphography/test/xaaaaa.gz","http://ropot.ijs.si/data/lymphography/test/xaaabj.gz"],
                            data_type = "gzip",
                            generate_urls = True,
                            X_indices = range(2,20),
                            id_index = 0,
                            y_index = 1,
                            X_meta = ["d","d","d","d","d","d","d","d","c","c","d","d","d","d","d","d","d","c"],
                            delimiter = ",")

fit_model = naivebayes.fit(train)
predictions = naivebayes.predict(test, fit_model)
print predictions