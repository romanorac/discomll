"""
Decision tree algorithm

Algorithm builds a binary decision tree. It expands nodes in priority order, where priority is set by measure (information gain or minimum description length).

"""

import numpy as np
from collections import Counter
from operator import itemgetter
import Queue
import random

def rand_indices(x, rand_attr):
	"""
	Function randomly selects features without replacement. It used with random forest. Selected features must have more than one distinct value.
	x: numpy array - dataset
	rand_attr - parameter defines number of randomly selected features
	"""
	loop = True
	indices = range(len(x[0]))

	while loop:
		loop = False
		#randomly selected features without replacement
		rand_list = random.sample(indices, rand_attr) 
		for i in rand_list:
			if len(np.unique(x[:,i])) == 1:
				loop = True
				indices.remove(i)
				if len(indices) == rand_attr-1:
					return -1 #all features in dataset have one distinct value
				break
	return rand_list

def fit(x, y, t, randomized, max_tree_nodes, min_samples_leaf, min_samples_split, class_majority, measure, accuracy, separate_max):
	"""
	Function builds a binary decision tree with given dataset and it expand tree nodes in priority order. Tree model is stored in a dictionary and it has following structure: 
	{parent_identifier: [(child_identifier, highest_estimated_feature_index , split_value, distribution_of_labels, depth, feature_type)]}

	x: numpy array - dataset with features
	y: numpy array - dataset with labels
	t: list - features types
	randomized: boolean - if True, algorithm estimates sqrt(num_of_features)+1 randomly selected features each iteration. If False, it estimates all features in each iteration.
	max_tree_nodes: integer - number of tree nodes to expand. 
	min_samples_leaf: float - minimal number of samples in leafs.
	class_majority: float - purity of the classes in leafs.
	measure: measure function - information gain or mdl.
	split_fun: split function - discretization of continuous features can be made randomly or with equal label frequency.
	"""

	depth = 0 #depth of the tree
	node_id = 1 #node identifier 

	#conditions of continuous and discrete features. 
	operation = {"c": (np.less_equal, np.greater), "d": (np.in1d, np.in1d)}
	tree = {0:[(node_id, -1 ,"", dict(Counter(y)), depth, "")]} #initialize tree model	
	mapping = range(len(x[0])) #global features indices
	rand_attr = int(np.ceil(np.sqrt(len(x[0])))) #sqrt(num_of_features) is estimated if randomized == True. If randomized == False, all indices are estimated at each node.
	recalculate = True
	while recalculate:
		recalculate = False
        	est_indices = rand_indices(x, rand_attr) if randomized else range(len(x[0])) 
		#estimate indices with given measure
		est = [measure(x[:,i], y, t[i], accuracy, separate_max) for i in est_indices]
		try:
			max_est, split = max(est) #find highest estimated split
		except:
			recalculate = True

	best = est_indices[est.index((max_est, split))] #select feature index with highest estimate
	
	queue = Queue.PriorityQueue() #initialize priority queue
	#put datasets in the queue
	queue.put((max_est, (node_id, x,y, mapping, best, split, depth)))

	while not queue.empty() and len(tree)*2 < max_tree_nodes: 
		_, (parent_id, x, y, mapping, best, split, depth) = queue.get()
		
		#features indices are mapped due to constantly changing subsets of data 
		best_map = mapping[best] 

		for j in range(2): #for left and right branch of the tree
			selection = range(len(x[0])) #select all indices for the new subset
			new_mapping = [i for i in mapping] #create a mapping of indices for a new subset
			
			if t[best_map] == "d" and len(split[j]) == 1:
				#if feature is discrete with one value in split, we cannot split it further. 
				selection.remove(best) #remove feature from new dataset
				new_mapping.remove(best_map) #remove mapping of feature

			#select rows of new dataset that satisfy condition (less than, greater than or in)
			indices = operation[t[best_map]][j](x[:,best],split[j]).nonzero()[0]
			#create new subsets of data
			sub_x, sub_y = x[indices.reshape(len(indices),1), selection], y[indices]
			if j==0 and (len(sub_y) < min_samples_leaf or len(x)-len(sub_y) < min_samples_leaf):
				break
			
			node_id += 1 #increase node identifier
			y_dist = Counter(sub_y) #distribution of labels in the new node

			#connect child node with its parent and update tree model
			tree[parent_id] = tree.get(parent_id, []) + [(node_id, best_map, set(split[j]) if t[best_map] == "d" else split[j], dict(y_dist), depth+1, t[best_map])]

			#select new indices for estimation
			est_indices = rand_indices(sub_x, rand_attr) if randomized and len(sub_x[0]) > rand_attr else range(len(sub_x[0]))
			#check label majority
			current_majority = y_dist[max(y_dist, key = y_dist.get)]/float(len(sub_y))
			
			#if new node satisfies following conditions it can be further split
			if current_majority < class_majority and len(sub_y) > min_samples_split and est_indices != -1: 
				#estimate selected indices
				est = [measure(sub_x[:,i], sub_y, t[new_mapping[i]], accuracy, separate_max) for i in est_indices]
				try:
					max_est, new_split = max(est) #find highest estimated split
				except:
					continue
				#select feature index with highest estimate
				new_best = est_indices[est.index((max_est, new_split))]
				#put new datasets in the queue with inverse value of estimate (priority order)
				queue.put((max_est*-1*len(sub_y),(node_id, sub_x, sub_y, new_mapping, new_best, new_split, depth+1)))
	return tree

def predict(tree, x, y = [], dist=False):
	"""
	Function makes a prediction of one sample with a tree model. If y label is defined it returns node identifier and margin.

	tree: dictionary - tree model
	x: numpy array - one sample from the dataset
	y: string, integer or float - sample label
	"""

	#conditions of continuous and discrete features
	node_id = 1 #initialize node identifier as first node under the root
	while 1:
		nodes = tree[node_id]

		if nodes[0][5] == "c":
			if x[nodes[0][1]] <= nodes[0][2]:
				index, node_id = 0, nodes[0][0] #set identifier of child node
			else:
				index, node_id = 1, nodes[1][0] #set identifier of child node
		else:
			if x[nodes[0][1]] in nodes[0][2]:
				index, node_id = 0, nodes[0][0] #set identifier of child node

			elif x[nodes[1][1]] in nodes[1][2]:
				index, node_id = 1, nodes[1][0] #set identifier of child node

			else:
				#value is not in left or right branch. Get label distributions of left and right child
				#sum labels distribution to get parent label distribution
				node_id = str(nodes[0][0]) + "," + str(nodes[1][0])
				index, nodes = 0, [[0,0,0,{ k: nodes[0][3].get(k, 0) + nodes[1][3] .get(k, 0) for k in set(nodes[0][3]) | set(nodes[1][3] )}]]

		if node_id in tree.keys(): #check if tree can be traversed further
			continue

		if dist:
			suma = sum(nodes[index][3].values())
			return Counter({k:v/float(suma) for k, v in nodes[index][3].iteritems()})
		
		prediction = max(nodes[index][3], key = nodes[index][3].get)
		if y == []:
			return prediction
		
		probs = sorted(zip(nodes[index][3].keys(), np.true_divide(nodes[index][3].values(), np.sum(nodes[index][3].values()))), key = itemgetter(1), reverse = True)
		if prediction == y:
			margin = probs[0][1] - probs[1][1] if len(probs) > 1 else 1
		else:
			margin = dict(probs).get(y, 0) - probs[0][1]
		return node_id, margin












