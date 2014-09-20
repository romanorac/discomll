# discomll #

Disco Machine Learning Library (discomll) is a python package for machine learning with MapReduce paradigm. It works with Disco framework for distributed computing. discomll is suited for analysis of large datasets as it offers classification, regression and clustering algorithms. 

## Algorithms ##
Classification algorithms
- naive Bayes - discrete and continuous features, 
- linear proximal SVM - continuous features, binary target,
- logistic regression - continuous features, binary target,
- decision trees - discrete and continuous features, 
- random forest - discrete and continuous features,
- weighted forest - discrete and continuous features,

Clustering algorithms
- k-means - continuous features,

Regression algorithms:
- linear regression - continuous features, continuous target,
- locally weighted linear regression - continuous features, continuous target,

## Features of discomll ##
discomll works with following data sources:
- datasets on Disco Distributed File System,
- text or gziped data accessible via file server.

discomll enables multiple settings for a dataset:
- defining training and test dataset with multiple data sources,
- feature selection,
- feature type specification,
- parsing of data,
- handling of missing values,
- generating URLs.

discomll enables:
- evaluation of the accuracy,
- class distribution views,
- algorithm model views.

## Installing ##
Prerequisites
- Disco 0.5.1 or newer,
- numpy should be installed on all worker nodes.

```bash
pip install discomll
```

## Performance analysis ##
In [performance analisys](http://1drv.ms/1qj6680), we compare speed and accuracy of discomll algorithms with scikit and Knime. We measure speedups of discomll algorithms with 1, 3, 6 and Disco workers. 

## Try it now ##
The ClowdFlows comes with discomll pre-installed and it can process big batch data using visual programming. ClowdFlows is an open sourced cloud based platform for composition, execution, and sharing of interactive machine learning and data mining workflows. 

- [Decision trees - lymphography dataset](http://clowdflows.org/workflow/2727/)
- [Random forest - segmentation dataset](http://clowdflows.org/workflow/2731/)
- [Naive Bayes - lymphography dataset](http://clowdflows.org/workflow/2729/)

## Additional info ##
Write me at orac.roman@gmail.com.




