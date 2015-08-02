from discomll.utils import accuracy
from discomll import dataset
from disco.core import result_iterator

"""
test_data = dataset.Data(data_tag = [["http://ropot.ijs.si/data/segmentation/test/xaaaaa.gz"]],#,"http://ropot.ijs.si/data/segmentation/test/xaaabj.gz"],
                            data_type = "gzip",
                            #generate_urls = True,
                            X_indices = range(2,21),
                            id_index = 0,
                            y_index = 1,
                            X_meta = ["c" for i in range(2,21)],
                            delimiter = ",")
"""
test_data = dataset.Data(data_tag = ["seg_test1","seg_test2"],#[["http://ropot.ijs.si/data/segmentation/test/xaaaaa.gz"]],#,"http://ropot.ijs.si/data/segmentation/test/xaaabj.gz"],
                            data_type = "chunk",
                            #generate_urls = True,
                            X_indices = range(2,21),
                            id_index = 0,
                            y_index = 1,
                            X_meta = ["c" for i in range(2,21)],
                            delimiter = ",")

predictions_url = [u'tag://disco:results:random_forest_predict@58f:513bf:a8a3']
accuracy = accuracy.measure(test_data, predictions_url, measure = "ca")
print accuracy

 