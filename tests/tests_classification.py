import unittest
import numpy as np
import datasets
import Orange
from disco.core import result_iterator

class Tests_Classification(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import chunk_testdata
        from disco import ddfs
        ddfs = ddfs.DDFS()

        if not ddfs.exists("test:ex3"):
            print "Chunking test datasets to DDFS..."
            chunk_testdata.chunk_testdata()

    def test_naivebayes_breastcancer(self):
        #python -m unittest tests_classification.Tests_Classification.test_naivebayes_breastcancer
        from discomll.classification import naivebayes
        train_data1, test_data1 = datasets.breastcancer_disc_orange()
        train_data2, test_data2 = datasets.breastcancer_disc_discomll()

        for m in range(3):            
            learner = Orange.classification.bayes.NaiveLearner(m = m)
            classifier = learner(train_data1)
            predictions1 = [classifier(inst, Orange.classification.Classifier.GetBoth) for inst in test_data1]
            predictions1_target = [v[0].value for v in predictions1]
            predictions1_probs = [v[1].values() for v in predictions1]

            fitmodel_url = naivebayes.fit(train_data2)
            predictions_url = naivebayes.predict(test_data2, fitmodel_url, m = m)
            predictions2_target = []
            predictions2_probs = []
            for k, v in result_iterator(predictions_url):
                predictions2_target.append(v[0])
                predictions2_probs.append(v[1])

            self.assertListEqual(predictions1_target, predictions2_target)
            self.assertTrue(np.allclose(predictions1_probs, predictions2_probs))
    
    def test_naivebayes_breastcancer_cont(self):
        #python -m unittest tests_classification.Tests_Classification.test_naivebayes_breastcancer_cont
        from sklearn.naive_bayes import GaussianNB
        from discomll.classification import naivebayes
        
        
        x_train, y_train, x_test, y_test = datasets.breastcancer_cont(replication = 1)
        train_data, test_data = datasets.breastcancer_cont_discomll(replication = 1)
        
        clf = GaussianNB()
        probs_log1 = clf.fit(x_train, y_train).predict_proba(x_test)
        
        fitmodel_url = naivebayes.fit(train_data)
        prediction_url = naivebayes.predict(test_data, fitmodel_url)
        probs_log2 = [v[1] for _, v in result_iterator(prediction_url)]

        self.assertTrue(np.allclose(probs_log1, probs_log2, atol = 1e-8))
    
    
    def test_log_reg_thetas(self):
        #python tests_classification.py Tests_Classification.test_log_reg_thetas
        from discomll.classification import logistic_regression

        train_data1 = datasets.ex4_orange()
        train_data2 = datasets.ex4_discomll()
        
        lr = Orange.classification.logreg.LogRegFitter_Cholesky(train_data1)
        thetas1 = lr[1]

        thetas_url  = logistic_regression.fit(train_data2)
        thetas2 = [v for k,v in result_iterator(thetas_url["logreg_fitmodel"]) if k == "thetas"]

        self.assertTrue(np.allclose(thetas1, thetas2))
    
    def test_log_reg(self):
        #python tests_classification.py Tests_Classification.test_log_reg
        from discomll.classification import logistic_regression

        train_data1, test_data1 = datasets.breastcancer_cont_orange()
        train_data2, test_data2  = datasets.breastcancer_cont_discomll()

        learner = Orange.classification.logreg.LogRegLearner(fitter=Orange.classification.logreg.LogRegFitter_Cholesky)
        classifier =learner(train_data1)
        thetas1 = classifier.beta

        predictions1 = []
        probabilities1 = []
        for inst in test_data1:
            target, probs = classifier(inst, Orange.classification.Classifier.GetBoth)
            predictions1.append(target.value)
            probabilities1.append(probs.values())

        thetas_url  = logistic_regression.fit(train_data2, alpha = 1e-8, max_iterations = 10)
        thetas2 = [v for k,v in result_iterator(thetas_url["logreg_fitmodel"]) if k == "thetas"]
        results_url = logistic_regression.predict(test_data2, thetas_url)

        predictions2 = []
        probabilities2 = []
        for k,v in result_iterator(results_url):
            predictions2.append(v[0])
            probabilities2.append(v[1])
        self.assertTrue(np.allclose(thetas1, thetas2))
        self.assertTrue(np.allclose(probabilities1, probabilities2, atol = 1e-5))
        self.assertListEqual(predictions1, predictions2)
    

    """
    def test_svm1(self):
        #python -m unittest tests_classification.Tests_Classification.test_svm1
        from sklearn import svm
        from discomll.classification import linear_proximal_svm
        
        data = load_dataset.Data()
        _, X_train, y_train, _, X_pred, y_pred = data.load_breast_cancer_disc()
        train_data, test_data = data.load_breast_cancer_disc_ddfs()

        clf = svm.SVC()
        clf.fit(X_train, y_train)
        predictions1 = clf.predict(X_pred)

        fit_model = linear_proximal_svm.fit(train_data)
        predictions2 = linear_proximal_svm.predict(test_data, fit_model)

        for i,(k,v) in enumerate(predictions2):
            if float(v[0]) != predictions1[i]:
                print k, v, predictions1[i]
    """
    """
    clf = svm.SVC()
    clf.fit(X_pred, y_pred)
    predictions1 = clf.predict(X_train)

    fit_model = linear_proximal_svm.fit(test_data)
    predictions2 = linear_proximal_svm.predict(train_data, fit_model)

    for i,(k,v) in enumerate(predictions2):
        if float(v[0]) != predictions1[i]:
            print k, v, predictions1[i]
    """


if __name__ == '__main__':
    unittest.main()
