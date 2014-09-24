# discomll #

Disco Machine Learning Library (discomll) is a python package for machine learning with MapReduce paradigm. It works with Disco framework for distributed computing. discomll is suited for analysis of large datasets as it offers classification, regression and clustering algorithms. Examples are available in discomll example directory.

## Algorithms ##
Classification algorithms
- naive Bayes - discrete and continuous features, 
- linear SVM - continuous features, binary target,
- logistic regression - continuous features, binary target,
- decision trees - discrete and continuous features, 
- random forest - discrete and continuous features,
- weighted forest - discrete and continuous features,

Clustering algorithms
- k-means - continuous features,

Regression algorithms
- linear regression - continuous features, continuous target,
- locally weighted linear regression - continuous features, continuous target,

Utilities
- evaluation of the accuracy,
- class distribution views,
- algorithm model views.

## Features of discomll ##
discomll works with following data sources:
- datasets on Disco Distributed File System,
- text or gziped data accessible via file server.

discomll enables multiple settings for a dataset:
- defining training and test dataset with multiple data sources,
- feature selection,
- feature type specification,
- parsing of data,
- handling of missing values.

## Installing ##
Prerequisites
- Disco 0.5.1 or newer,
- numpy should be installed on all worker nodes.

```bash
pip install discomll
```

## Performance analysis ##
In [performance analisys](http://1drv.ms/1qj6680), we compare speed and accuracy of discomll algorithms with scikit and Knime. We measure speedups of discomll algorithms with 1, 3, 6 and 9 Disco workers.

## Try it now ##
You can try discomll algorithms on the ClowdFlows platform. ClowdFlows is an open sourced cloud based platform for composition, execution, and sharing of interactive machine learning and data mining workflows.

![alt tag](https://github.com/romanorac/discomll/blob/master/big_data_workflow.png)
 
Public workflows:

- [naive Bayes - lymphography dataset,](http://clowdflows.org/workflow/2729/)
- [naive Bayes - segmentation dataset,](http://clowdflows.org/workflow/2788/)
- [logistic regression - sonar dataset,](http://clowdflows.org/workflow/2801/)
- [logistic regression - ionosphere dataset,](http://clowdflows.org/workflow/2802/)
- [linear SVM - sonar dataset,](http://clowdflows.org/workflow/2799/)
- [linear SVM - ionosphere dataset,](http://clowdflows.org/workflow/2800/)
- [decision trees - lymphography dataset,](http://clowdflows.org/workflow/2727/)
- [decision trees - segmentation dataset,](http://clowdflows.org/workflow/2796/)
- [random forest - lymphography dataset,](http://clowdflows.org/workflow/2789/)
- [random forest - segmentation dataset,](http://clowdflows.org/workflow/2731/)
- [weighted forest - lymphography dataset,](http://clowdflows.org/workflow/2797/)
- [weighted forest - segmentation dataset,](http://clowdflows.org/workflow/2798/)
- [k-means - linear dataset,](http://clowdflows.org/workflow/2812/)
- [k-means - segmentation dataset,](http://clowdflows.org/workflow/2811/)
- [linear regression - linear dataset,](http://clowdflows.org/workflow/2815/)
- [linear regression - fraction dataset,](http://clowdflows.org/workflow/2816/)

## Additional info ##
Write me at orac.roman@gmail.com.




