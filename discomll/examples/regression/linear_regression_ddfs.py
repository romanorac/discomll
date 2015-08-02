from discomll import dataset 
from discomll.regression import linear_regression
from discomll.utils import model_view
from disco.core import result_iterator

#define training dataset
train = dataset.Data(data_tag = ["test:ex3"],
					data_type = "chunk", 
					X_indices = [0,1],
					y_index = 2)

#define test dataset
test = dataset.Data(data_tag = ["test:ex3_test"],
					data_type = "chunk",
					X_indices = [0,1],
					y_index = 2)

#fit model on training dataset
fit_model = linear_regression.fit(train)

#output model
model = model_view.output_model(fit_model)
print model

#predict test dataset
predictions = linear_regression.predict(test, fit_model)

#output results
for k,v in result_iterator(predictions):
    print k, v[0]









