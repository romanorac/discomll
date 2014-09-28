import unittest
import numpy as np
import datasets
from disco.core import result_iterator

class Tests_Regression(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import chunk_testdata
        from disco import ddfs
        ddfs = ddfs.DDFS()

        if not ddfs.exists("test:ex3"):
            print "Chunking test datasets to DDFS..."
            chunk_testdata.chunk_testdata()
    
    def test_lwlr(self):
        #python -m unittest tests_regression.Tests_Regression.test_lwlr
        import locally_weighted_linear_regression as lwlr1
        from discomll.regression import locally_weighted_linear_regression as lwlr2

        x_train, y_train, x_test, y_test = datasets.regression_data()
        train_data, test_data = datasets.regression_data_discomll()

        lwlr1 = lwlr1.Locally_Weighted_Linear_Regression()
        taus = [1, 10, 25]
        sorted_indices = np.argsort([str(el) for el in x_test[:,1].tolist()])
        
        for tau in taus:            
            thetas1,estimation1 = lwlr1.fit(x_train, y_train, x_test, tau = tau)
            thetas1,estimation1 = np.array(thetas1)[sorted_indices],np.array(estimation1)[sorted_indices]
            
            results = lwlr2.fit_predict(train_data, test_data, tau = tau )
            thetas2, estimation2 = [], []
            
            for x_id, (est, thetas)  in result_iterator(results):                
                estimation2.append(est)
                thetas2.append(thetas)
            
            self.assertTrue(np.allclose(thetas1, thetas2, atol = 1e-8))
            self.assertTrue(np.allclose(estimation1, estimation2, atol = 1e-3))
    
    def test_lin_reg(self):
        #python -m unittest tests_regression.Tests_Regression.test_lin_reg
        from sklearn import linear_model
        from discomll.regression import linear_regression

        x_train, y_train, x_test, y_test = datasets.ex3()
        train_data, test_data = datasets.ex3_discomll()

        lin_reg = linear_model.LinearRegression() # Create linear regression object
        lin_reg.fit(x_train, y_train) # Train the model using the training sets
        thetas1 = [lin_reg.intercept_] + lin_reg.coef_[1:].tolist()
        prediction1 = lin_reg.predict(x_test)

        thetas_url = linear_regression.fit(train_data)
        thetas2 = [v for k,v in result_iterator(thetas_url["linreg_fitmodel"])]
        results = linear_regression.predict(test_data, thetas_url)
        prediction2 = [v[0] for k,v in result_iterator(results)]
       
        self.assertTrue(np.allclose(thetas1, thetas2))
        self.assertTrue(np.allclose(prediction1, prediction2))
            
if __name__ == '__main__':
    unittest.main()




















