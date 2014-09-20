from discomll import dataset
from discomll.clustering import kmeans
from discomll.utils import model_view
from disco.core import result_iterator

#define training dataset
train = dataset.Data(data_tag = ["test:breast_cancer_cont"],
                     data_type = "chunk", #define data source - chunk data on ddfs
                     X_indices = xrange(0,9), #define attribute indices
                     y_index = 9, #define class index
                     delimiter = ",")

#define test dataset
test = dataset.Data(data_tag = ["test:breast_cancer_cont_test"], 
                     data_type = "chunk", #define data source - chunk data on ddfs
                     X_indices = xrange(0,9), #define attribute indices
                     y_index = 9, #define class index
                     delimiter = ",") 

#fit model on training dataset
fit_model = kmeans.fit(train, n_clusters = 2, max_iterations = 5, random_state = 0)

#output model
model = model_view.output_model(fit_model)
print model

#predict test dataset
predictions = kmeans.predict(test, fit_model)

#output results
for k,v in result_iterator(predictions):
    print k,v

