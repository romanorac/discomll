from discomll import dataset
from discomll.classification import logistic_regression
from discomll.utils import model_view

#define training dataset
train = dataset.Data(data_tag = ["test:ex4"],
                    data_type = "chunk",
                    X_indices = xrange(0,2),
                    y_index = 2,
                    y_map = ["0.0000000e+00","1.0000000e+00"])

#fit model on training dataset
fit_model = logistic_regression.fit(train)

#output model
model = model_view.output_model(fit_model)
print model 


