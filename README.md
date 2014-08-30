# discomll #

Disco Machine Learning Library (discomll) is a python package for machine learning with MapReduce paradigm. It works with Disco framework for distributed computing. discomll is suited for analysis of large datasets as it offers classification, regression and clustering algorithms. 

## Algorithms ##
Classification algorithms
- naive Bayes - discrete and continuous features, 
- random forest - discrete and continuous features, 
- decision trees - discrete and continuous features, 
- linear proximal SVM - continuous features, binary target,
- logistic regression - continuous features, binary target,

Clustering algorithms
- k-means - continuous features,

regression algorithms:
- linear regression - continuous features, continuous target,
- locally weighted linear regression - continuous features, continuous target,

## Features of discomll ##
discomll works with following data sources:
- data on Disco Distributed File System (DDFS),
- text or gziped data accessible via file server.

discomll enables multiple settings for a dataset:
- defining training and test dataset with multiple data sources,
- feature selection,
- feature type specification,
- parsing of data,
- handling of missing values,
- generating URLs. 

## Installing ##
Prerequisites
- Disco 0.5.1,
- numpy should be installed on all cluster nodes,
- scikit-learn and Orange are optional (needed for unit tests).

To install, download package and run
```bash
python setup.py install
```

To run unit tests: 
```bash
python setup.py test
```

## Questions ##
For any additional info, write me at orac.roman@gmail.com




