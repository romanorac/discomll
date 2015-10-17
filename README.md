# discomll #

Disco Machine Learning Library (discomll) is a python package for machine learning with MapReduce paradigm. It works with Disco framework for distributed computing. discomll is suited for analysis of large datasets as it offers classification, regression and clustering algorithms.

## Algorithms ##
Classification algorithms
- naive Bayes - discrete and continuous features, 
- linear SVM - continuous features, binary target,
- logistic regression - continuous features, binary target,
- forest of distributed decision trees - discrete and continuous features,
- distributed random forest - discrete and continuous features,
- distributed weighted forest (experimental) - discrete and continuous features,
- distributed weighted forest rand (experimental) - discrete and continuous features,

Clustering algorithms
- k-means - continuous features,

Regression algorithms
- linear regression - continuous features, continuous target,
- locally weighted linear regression - continuous features, continuous target,

Utilities
- evaluation of the accuracy,
- distribution views,
- model views.

## Features of discomll ##
discomll works with following data sources:
- datasets on the Disco Distributed File System,
- text or gziped datasets accessible via file server.

discomll enables multiple settings for a dataset:
- multiple data sources,
- feature selection,
- feature type specification,
- parsing of data,
- handling of missing values.

## Installing ##
Prerequisites
- Disco 0.5.4,
- numpy should be installed on all worker nodes,
- orange and scikit-learn are used in unit tests.

```bash
pip install discomll
```

## Performance analysis ##
In [performance analysis](http://1drv.ms/1qj6680), we compare speed and accuracy of discomll algorithms with scikit and Knime. We measure speedups of discomll algorithms with 1, 3, 6 and 9 Disco workers.

## Performance analysis 2##
In [second performance analysis](http://1drv.ms/1FYORb8), we compare accuracy of distributed ensemble algorithms with scikit-learn algorithms. We train the model on whole dataset with distributed algorithms and on a subset with single core algorithms. We show that distributed ensembles achieve similar accuracy as single core algorithms. 

## Try it now ##
You can try discomll algorithms on the ClowdFlows platform. ClowdFlows is an open sourced cloud based platform for composition, execution, and sharing of interactive machine learning and data mining workflows. For instruction see the [User Guide.](https://onedrive.live.com/redir?resid=C695DFFBD3161AEA!161&authkey=!AERQJpsxOqkLykI&ithint=file%2cpdf)

![alt tag](https://github.com/romanorac/discomll/blob/master/big_data_workflow.png)
 
Public workflows:

- [naive Bayes - lymphography dataset,](http://clowdflows.org/workflow/2729/)
- [naive Bayes - segmentation dataset,](http://clowdflows.org/workflow/2788/)
- [logistic regression - sonar dataset,](http://clowdflows.org/workflow/2801/)
- [logistic regression - ionosphere dataset,](http://clowdflows.org/workflow/2802/)
- [linear SVM - sonar dataset,](http://clowdflows.org/workflow/2799/)
- [linear SVM - ionosphere dataset,](http://clowdflows.org/workflow/2800/)
- [forest of distributed decision trees - lymphography dataset,](http://clowdflows.org/workflow/2727/)
- [forest of distributed decision trees - segmentation dataset,](http://clowdflows.org/workflow/2796/)
- [distributed random forest - lymphography dataset,](http://clowdflows.org/workflow/2789/)
- [distributed random forest - segmentation dataset,](http://clowdflows.org/workflow/2731/)
- [distributed weighted forest rand - lymphography dataset,](http://clowdflows.org/workflow/2797/)
- [distributed weighted forest rand - segmentation dataset,](http://clowdflows.org/workflow/2798/)
- [k-means - linear dataset,](http://clowdflows.org/workflow/2812/)
- [k-means - segmentation dataset,](http://clowdflows.org/workflow/2811/)
- [linear regression - linear dataset,](http://clowdflows.org/workflow/2815/)
- [linear regression - fraction dataset,](http://clowdflows.org/workflow/2816/)

## Release notes ##
### version 0.1.4.1 (Released 17/oct/2015) ###
 - model view fixed for ensembles,
 - bug fixes in examples and tests.

### version 0.1.4 (Released 11/oct/2015) ###
 - distributed weighted forest Rand was added. Algorithm is similar to distributed weighted forest, but it uses randomly selected medoids.
 - improvements of algorithms, especially ensembles,
 - performance analysis 2.




