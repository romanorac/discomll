import unittest
import numpy as np
import datasets
import random
from disco.core import result_iterator

class Tests_Clustering(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        import chunk_testdata
        from disco import ddfs
        ddfs = ddfs.DDFS()

        if not ddfs.exists("test:ex3"):
            print "Chunking test datasets to DDFS..."
            chunk_testdata.chunk_testdata()

    def test_kmeans_iris(self):
        #python -m unittest tests_clustering.Tests_Clustering.test_kmeans_iris
        from discomll.clustering import kmeans
        from sklearn.cluster import KMeans

        max_iter = 10
        clusters = 3
        random_seed = 0

        x_train, y_train, x_test, y_test  = datasets.iris()
        train_data, test_data = datasets.iris_discomll()
        
        sk_kmeans = KMeans(n_clusters=clusters, max_iter = max_iter, n_init=1, random_state=random_seed).fit(x_train)
        centroids1 = sk_kmeans.cluster_centers_
        #predictions1 = sk_kmeans.predict(x_test)
        
        centroids_url = kmeans.fit(train_data,
                            n_clusters = clusters,
                            max_iterations = max_iter,
                            random_state = random_seed)
        
        
        predictions_url = kmeans.predict(test_data, centroids_url)
        #predictions2 = [v[1] for k,v in result_iterator(predictions_url)]

        centroids2 = [v["x"] for k,v in result_iterator(centroids_url["kmeans_fitmodel"])]
        centroids2[0], centroids2[2] = centroids2[2], centroids2[0]
        self.assertTrue(np.allclose(centroids1, centroids2))

        
    

    def test_kmeans_breastcancer(self):
        #python -m unittest tests_clustering.Tests_Clustering.test_kmeans_breastcancer
        from discomll.clustering import kmeans
        from sklearn.cluster import KMeans
        
        max_iter = 10
        clusters = 2
        random_seed = 2
 
        x_train, _, x_test, _  = datasets.breastcancer_disc()
        train_data, test_data = datasets.breastcancer_disc_discomll()

        kmeans2 = KMeans(n_clusters=clusters, max_iter=max_iter, n_init=1, random_state=random_seed).fit(x_train)
        centroids1 = kmeans2.cluster_centers_
        predictions1 = kmeans2.predict(x_test)

        centroids_url = kmeans.fit(train_data,
                            n_clusters = clusters,
                            max_iterations = max_iter,
                            random_state = random_seed)

        predictions_url = kmeans.predict(test_data, centroids_url)
        predictions2 = [v[0] for k,v in result_iterator(predictions_url)]
        centroids2 = [v["x"] for k,v in result_iterator(centroids_url["kmeans_fitmodel"])]

        centroids2[0], centroids2[1] = centroids2[1], centroids2[0]

        self.assertTrue(np.allclose(centroids1, centroids2))
    


if __name__ == '__main__':
    unittest.main()
