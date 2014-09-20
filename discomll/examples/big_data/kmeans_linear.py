from discomll import dataset
from discomll.clustering import kmeans


train_data = dataset.Data(data_tag = ["test:breast_cancer_cont"],
                            data_type = "chunk",
                            X_indices = xrange(0,9),
                            y_index = 9,
                            delimiter = ",")


test_data = dataset.Data(data_tag = ["test:breast_cancer_cont_test"],
                                   data_type = "chunk",
                                   X_indices = xrange(0,9),
                                   y_index = 9,
                                   delimiter = ",")

fit_model = kmeans.fit(train, n_clusters = 5, max_iterations = 10, random_state = 0)
predictions = kmeans.predict(test, fit_model)
print predictions
