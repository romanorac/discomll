"""
Module measures contains: 
    function for calculating information gain
    function for calculating minimum description length
    heuristic for searching best binary split of nominal values
    function for equal frequency label discretization of numerical values
    function for random discretization of numerical values
"""

import numpy as np
import math
from collections import Counter

def mdl(x, y, ft, accuracy, separate_max):
    return mdl_nominal(x, y, separate_max) if ft == "d" else mdl_numeric(x, y, accuracy)

def info_gain(x, y, ft, accuracy, separate_max):
    return info_gain_nominal(x, y, separate_max) if ft == "d" else info_gain_numeric(x, y, accuracy)

def nominal_splits(x, y, x_vals, y_dist, separate_max):
    """
    Function uses heuristic to find best binary split of nominal values. Heuristic is described in (1) and it is originally defined for binary classes. We extend it to work with multiple classes by comparing label with least samples to others.

    x: numpy array - nominal feature
    y: numpy array - label
    x_vals: numpy array - unique nominal values of x
    y_dist: dictionary - distribution of labels

    Reference:
    (1) Classification and Regression Trees by Breiman, Friedman, Olshen, and Stone, pages 101- 102. 
    """
    #select a label with least samples
    y_val = max(y_dist, key=y_dist.get) if separate_max else min(y_dist, key=y_dist.get)

    prior = y_dist[y_val]/float(len(y)) #prior distribution of selected label 

    values, dist, splits = [],[], []
    for x_val in x_vals: #for every unique nominal value
        dist.append(Counter(y[x == x_val])) #distribution of labels at selected nominal value
        splits.append(x_val)
        suma = sum([prior * dist[-1][y_key] for y_key in y_dist.keys()])
        #estimate probability
        values.append(prior * dist[-1][y_val]/float(suma))
    indices = np.array(values).argsort()[::-1]

    #distributions and splits are sorted according to probabilities
    return np.array(dist)[indices], np.array(splits)[indices].tolist()

def h(values):
    """
    Function calculates entropy.

    values: list of integers
    """
    ent = np.true_divide(values, np.sum(values))
    return -np.sum(np.multiply(ent, np.log2(ent)))

def info_gain_nominal(x, y, separate_max):
    """
    Function calculates information gain for discrete features. If feature is continuous it is firstly discretized.

    x: numpy array - numerical or discrete feature
    y: numpy array - labels
    ft: string - feature type ("c" - continuous, "d" - discrete)
    split_fun: function - function for discretization of numerical features
    """
    x_vals = np.unique(x) #unique values
    if len(x_vals) < 3: #if there is just one unique value
        return None
    y_dist = Counter(y) #label distribution
    h_y = h(y_dist.values()) #class entropy
    
    #calculate distributions and splits in accordance with feature type
    
    dist, splits = nominal_splits(x, y, x_vals, y_dist, separate_max)
    
    indices, repeat = (range(1, len(dist)), 1) if len(dist) < 50 else (range(1, len(dist), len(dist)/10), 3)
    interval= len(dist)/10

    max_ig, max_i, iteration = 0, 1, 0
    while iteration < repeat:    
        for i in indices:
            dist0 = np.sum([el for el in dist[:i]]) #iter 0: take first distribution
            dist1 = np.sum([el for el in dist[i:]]) #iter 0: take the other distributions without first
            coef = np.true_divide([np.sum(dist0.values()),np.sum(dist1.values())], len(y))
            ig = h_y - np.dot(coef, [h(dist0.values()), h(dist1.values())]) #calculate information gain
            if ig > max_ig: 
                max_ig, max_i = ig, i #store index and value of maximal information gain
        iteration+=1
        if repeat > 1:
            interval = int(interval * 0.5)
            if max_i in indices  and interval > 0:
                middle_index = indices.index(max_i)
            else:
                break
            min_index = middle_index if middle_index == 0 else middle_index - 1
            max_index = middle_index if middle_index == len(indices)-1 else middle_index + 1 
            indices = range(indices[min_index], indices[max_index], interval)

    #store splits of maximal information gain in accordance with feature type
    return float(max_ig),  [splits[:max_i], splits[max_i:]]

def info_gain_numeric(x, y, accuracy):
    x_unique = list(np.unique(x))
    if len(x_unique) == 1:
        return None
    indices = x.argsort() #sort numeric attribute
    x, y = x[indices], y[indices] #save sorted features with sorted labels
    
    right_dist = np.bincount(y)
    dummy_class = np.array([len(right_dist)])
    class_indices = right_dist.nonzero()[0]
    right_dist = right_dist[class_indices]
    left_dist = np.zeros(len(class_indices))

    diffs = np.nonzero(y[:-1] != y[1:])[0] + 1 #different neighbor classes have value True
    if accuracy > 0:
        diffs = np.array([diffs[i] for i in range(1, len(diffs)) if diffs[i]-diffs[i-1] > accuracy], dtype=np.int32) if len(diffs) > 15 else diffs
    intervals = np.array((np.concatenate(([0],diffs[:-1])), diffs)).T
    if len(diffs) < 2: 
        return None


    max_ig, max_i, max_j = 0, 0, 0
    prior_h = h(right_dist) #calculate prior entropy
    
    for i, j in intervals:            
        dist = np.bincount(np.concatenate((dummy_class, y[i:j])))[class_indices]
        left_dist += dist
        right_dist -= dist
        coef = np.true_divide((np.sum(left_dist), np.sum(right_dist)), len(y)) 
        ig = prior_h - np.dot(coef, [h(left_dist[left_dist.nonzero()]), h(right_dist[right_dist.nonzero()])])
        if ig > max_ig:
            max_ig, max_i, max_j = ig, i, j

    if x[max_i] == x[max_j]:
        ind = x_unique.index(x[max_i])
        mean = np.float32(np.mean((x_unique[1 if ind == 0 else ind-1], x_unique[ind])))
    else:
        mean = np.float32(np.mean((x[max_i], x[max_j])))
    
    return float(max_ig), [mean, mean]

def multinomLog2(selectors):
    """
    Function calculates logarithm 2 of a kind of multinom.

    selectors: list of integers
    """

    ln2 = 0.69314718055994528622
    noAll = sum(selectors)
    lgNf = math.lgamma(noAll+1.0)/ln2 #log2(N!)

    lgnFac = []
    for selector in selectors:
        if selector == 0 or selector == 1:
            lgnFac.append(0.0)
        elif selector == 2:
            lgnFac.append(1.0)
        elif selector == noAll:
            lgnFac.append(lgNf)
        else:
            lgnFac.append(math.lgamma(selector+1.0)/ln2)
    return lgNf - sum(lgnFac)

def calc_mdl(yx_dist, y_dist):
    """
    Function calculates mdl with given label distributions. 

    yx_dist: list of dictionaries - for every split it contains a dictionary with label distributions
    y_dist: dictionary - all label distributions
    
    Reference:
    Igor Kononenko. On biases in estimating multi-valued attributes. In IJCAI, volume 95, pages 1034-1040, 1995.
    """
    prior = multinomLog2(y_dist.values())
    prior += multinomLog2([len(y_dist.keys())-1, sum(y_dist.values())])
    
    post = 0
    for x_val in yx_dist:
        post += multinomLog2([x_val.get(c, 0) for c in y_dist.keys()])      
        post += multinomLog2([len(y_dist.keys())-1, sum(x_val.values())])
    return (prior - post)/float(sum(y_dist.values()))

def mdl_nominal(x, y, separate_max):
    """
    Function calculates minimum description length for discrete features. If feature is continuous it is firstly discretized.

    x: numpy array - numerical or discrete feature
    y: numpy array - labels
    """
    x_vals = np.unique(x) #unique values
    if len(x_vals) == 1: #if there is just one unique value
        return None
    
    y_dist = Counter(y) #label distribution
    #calculate distributions and splits in accordance with feature type
    dist, splits = nominal_splits(x, y, x_vals, y_dist, separate_max)
    prior_mdl = calc_mdl(dist, y_dist)

    max_mdl, max_i = 0, 1
    for i in range(1, len(dist)):
        #iter 0: take first distribution
        dist0_x = [el for el in dist[:i]]
        dist0_y = np.sum(dist0_x)
        post_mdl0 = calc_mdl(dist0_x, dist0_y)

        #iter 0: take the other distributions without first
        dist1_x = [el for el in dist[i:]]
        dist1_y = np.sum(dist1_x)
        post_mdl1 = calc_mdl(dist1_x, dist1_y)

        coef = np.true_divide([sum(dist0_y.values()),sum(dist1_y.values())], len(x))
        mdl_val = prior_mdl - np.dot(coef, [post_mdl0, post_mdl1]) #calculate mdl
        if mdl_val > max_mdl:
            max_mdl, max_i = mdl_val, i 
    
    #store splits of maximal mdl in accordance with feature type
    split = [splits[:max_i], splits[max_i:]]
    return (max_mdl,  split)

def mdl_numeric(x, y, accuracy):
    x_unique = list(np.unique(x))
    if len(x_unique) == 1:
        return None
    indices = x.argsort() #sort numeric attribute
    x, y = x[indices], y[indices] #save sorted features with sorted labels
    
    right_dist = np.bincount(y)
    dummy_class = np.array([len(right_dist)])
    class_indices = right_dist.nonzero()[0]
    right_dist = right_dist[class_indices]
    left_dist = np.zeros(len(class_indices))
    y_dist = Counter(dict(zip(class_indices, right_dist)))

    diffs = np.nonzero(y[:-1] != y[1:])[0] + 1 #different neighbor classes have value True
    if accuracy > 0:
        diffs = np.array([diffs[i] for i in range(1, len(diffs)) if diffs[i]-diffs[i-1] > accuracy], dtype=np.int32) if len(diffs) > 15 else diffs
    intervals = np.array((np.concatenate(([0],diffs[:-1])), diffs)).T
    
    dist = [Counter(dict(zip(class_indices,np.bincount(np.concatenate((dummy_class, y[i:j])))[class_indices]))) for i, j in intervals]

    prior_mdl = calc_mdl(dist, y_dist)
    max_mdl, max_i = 0, 0
    
    for i in range(1, len(dist)):
        #iter 0: take first distribution        
        dist0_x = dist[:i]
        dist0_y = np.sum(dist0_x)
        post_mdl0 = calc_mdl(dist0_x, dist0_y)

        #iter 0: take the other distributions without first
        dist1_x = dist[i:]
        dist1_y = np.sum(dist1_x) 
        post_mdl1 = calc_mdl(dist1_x, dist1_y)
        coef = np.true_divide([sum(dist0_y.values()),sum(dist1_y.values())], len(x))

        mdl_val = prior_mdl - np.dot(coef, [post_mdl0, post_mdl1]) #calculate mdl
        if mdl_val > max_mdl:
            max_mdl, max_i = mdl_val, i
    
    max_i, max_j = intervals[max_i]
    if x[max_i] == x[max_j]:
        ind = x_unique.index(x[max_i])
        mean = np.mean((x[1 if ind == 0 else ind-1], x[ind]))
    else:
        mean = np.mean((x[max_i], x[max_j]))
    return (max_mdl,  [mean, mean])

















