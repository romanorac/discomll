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
import random
import sys

def random_splits(x, y, intervals):
	"""
	Function makes discretization of continuous feature. It randomly splits continuous feature if len(x) > intervals. If there is less samples, it uses equal_freq_splits function. Split is mean value calculated from two distinct neighbor values.
	
	x: numpy array - continuous feature
	y: numpy array - labels
	intervals: int - number of intervals   
	"""

	if len(x) > intervals: #if there is more samples than intervals 
		indices = x.argsort() #sort continuous feature
		x, y = x[indices],y[indices] #save sorted values with sorted labels

		prev_i = 0
		dist, splits = [],[-np.inf] #distribution of labels, split values (inf is dummy value)
		#select randomly without replacement number of instances which will represent splits
		indices = sorted(random.sample(range(1, len(x)), intervals)) 
		for i in indices:
			mean = np.mean((x[prev_i], x[i])) #mean value between previous split and current
			if mean != splits[-1]: #avoid duplicates
				#calculate distribution of labels from previous split to current
				dist.append(Counter(y[prev_i:i])) 
				splits.append(mean)
				prev_i = i
		dist[-1] += Counter(y[prev_i:]) #distribution of labels from current split to the end
		splits = splits[1:] #remove dummy value from the split

	else: #if there is less samples than intervals, make splits with equal label frequency
		dist, splits = equal_freq_splits(x, y, intervals)
	
	return dist, splits

def equal_freq_splits(x, y, intervals):
	"""
	Function makes discretization of continuous features. It splits continuous feature with equal label frequency if len(samples) > intervals. Otherwise it takes unique values and it makes intervals/3 number of splits. Split is mean value calculated from two distinct neighbor values
	
	x: numpy array - continuous feature
	y: numpy array - labels
	intervals: int - number of intervals   
	"""
	indices = x.argsort() #sort continuous feature
	x, y = x[indices],y[indices] #save sorted features with sorted labels
	x_vals =  list(np.unique(x)) #sorted unique values (x was sorted before)
	prev_i = 0
	dist, splits = [], [-np.inf] #distribution of labels, split values (-inf dummy value)
	
	if len(x_vals) < intervals: #if there is less unique values than intervals s
		#make interval/3 number of split - it is not desirable to make a split for every distinct value
		for i in range(1, len(y)):
			if y[i-1] != y[i]:
				index = x_vals.index(x[i])
				split = np.mean((x_vals[1 if index == 0 else index-1],x_vals[index])) 
				if split > splits[-1]: #for distinct splits
					dist.append(Counter(y[prev_i: i])) #calculate label distribution
					splits.append(split)
					prev_i = i

	else: #equal frequency discretization
		freq = len(y)/intervals #interval
		for i in range(1,intervals):
			index = x_vals.index(x[i*freq])
			split = np.mean((x_vals[1 if index == 0 else index-1],x_vals[index]))  
			if split > splits[-1]: #for distinct splits
				dist.append(Counter(y[prev_i: i*freq])) #calculate label distribution
				splits.append(split)
				prev_i=i*freq


	dist[-1] += Counter(y[prev_i:]) #distribution of labels from current split to the end
	return dist, splits[1:]

def nominal_splits(x, y, x_vals, y_dist):
	"""
	Function uses heuristic to find best binary split of nominal values. Heuristic is described in (1) and it is originally defined for binary classes. We extend it to work with multiple classes by comparing label with least samples to others.

	x: numpy array - nominal feature
	y: numpy array - label
	x_vals: numpy array - unique nominal values of x
	y_dist: dictionary - distribution of labels

	Reference:
	(1) Classification and Regression Trees by Breiman, Friedman, Olshen, and Stone, pages 101- 102. 
	"""

	y_val = min(y_dist, key=y_dist.get) #select a label with least samples
	prior = y_dist[y_val]/float(len(y)) #prior distribution of selected label 

	values, dist, splits = [],[], []
	for x_val in x_vals: #for every unique nominal value
		dist.append(Counter(y[x == x_val])) #distribution of labels at selected nominal value
		splits.append(x_val)
		suma = sum([prior * dist[-1][y_key] for y_key in y_dist.keys()])
		#estimate probability
		values.append(prior * dist[-1][y_val]/float(suma))
	indices = np.array(values).argsort()

	#distributions and splits are sorted according to probabilities
	return np.array(dist)[indices], np.array(splits)[indices].tolist()

def h(values):
	"""
	Function calculates entropy.

	values: list of integers
	"""
	ent = np.true_divide(values, np.sum(values))
	ent[ent==0] = 1e-10
	return -np.sum(np.multiply(ent, np.log2(ent)))

def info_gain(x, y, ft, split_fun, intervals):
	"""
	Function calculates information gain for discrete features. If feature is continuous it is firstly discretized.

	x: numpy array - numerical or discrete feature
	y: numpy array - labels
	ft: string - feature type ("c" - continuous, "d" - discrete)
	split_fun: function - function for discretization of numerical features
	"""
	x_vals = np.unique(x) #unique values
	if len(x_vals) == 1: #if there is just one unique value
		return None
	y_dist = Counter(y) #label distribution
	h_y = h(y_dist.values()) #class entropy
	
	#calculate distributions and splits in accordance with feature type
	dist, splits = nominal_splits(x, y, x_vals, y_dist) if ft == "d" else split_fun(x, y, intervals)
	
	max_ig, max_i = 0, 1 	
	for i in range(1, len(dist)):
		dist0 = np.sum([el for el in dist[:i]]) #iter 0: take first distribution
		dist1 = np.sum([el for el in dist[i:]]) #iter 0: take the other distributions without first
		coef = np.true_divide([np.sum(dist0.values()),np.sum(dist1.values())], len(y))
		ig = h_y - np.dot(coef, [h(dist0.values()), h(dist1.values())]) #calculate information gain
		if ig > max_ig: 
			max_ig, max_i = ig, i #store index and value of maximal information gain
	
	#store splits of maximal information gain in accordance with feature type
	split = [splits[:max_i], splits[max_i:]] if ft == "d" else [float(splits[max_i-1]), float(splits[max_i-1])]
	#print split
	return (float(max_ig),  split)

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

def mdl(x, y, ft, split_fun, intervals):
	"""
	Function calculates minimum description length for discrete features. If feature is continuous it is firstly discretized.

	x: numpy array - numerical or discrete feature
	y: numpy array - labels
	ft: string - feature type ("c" - continuous, "d" - discrete)
	split_fun: function - function for discretization of numerical features
	"""
	x_vals = np.unique(x) #unique values
	if len(x_vals) == 1: #if there is just one unique value
		return None
	
	y_dist = Counter(y) #label distribution
	#calculate distributions and splits in accordance with feature type
	dist, splits = nominal_splits(x, y, x_vals, y_dist) if ft == "d" else split_fun(x, y, intervals)
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
	split = [splits[:max_i], splits[max_i:]] if ft == "d" else [float(splits[max_i-1]), float(splits[max_i-1])]
	return (float(max_mdl),  split)
















