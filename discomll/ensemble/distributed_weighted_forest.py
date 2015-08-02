"""
Weighted forest with MapReduce

Weighted forest is a novel ensemble algorithm. 

Fit phase
Weighted forest algorithm builds multiple decision trees with a bootstrap method on a subset of data. In each tree node, it estimates sqrt(num. of attributes)+1 randomly selected attributes (without replacement). It uses decision tree to predict out-of-bag samples. For each prediction of an out-of-bag sample, it measures margin (classifier confidence in prediction) and leaf identifier that outputs prediction. Algorithm uses similarity matrix, where it stores similarities for each out-of-bag sample that was predicted with the same leaf. We assume that samples are similar, if the same leaf predicts them multiple times in multiple decision trees. 
After algorithm builds all decision trees, it passes similarity matrix to k-medoids algorithm. Similarity matrix presents distances between test samples. We set parameter k as sqrt(num. of attributes)+1. k-medoids algorithm outputs medoids, which are test samples in the cluster centers of the dataset. Medoids are actual samples in a dataset, unlike centroids which are centers of clusters. Algorithm measures average margin for all samples that are in the cluster of certain medoid. It saves the average margin of a decision tree in its model. Algorithm uses this scores as weights of decision trees in predict phase.
Algorithm builds a forest on each subset of the data and it merges them in large ensemble. Each forest has its own medoids.

Predict phase 
Algorithm selects a forest (or more, if it finds equal similarities with medoids in multiple forests), that contain most similar medoid with a test sample. Then it uses the same procedure like with small data. Algorithm calculates Gower similarity coefficient with a test sample and every medoid. Only decision trees with high margin on similar test samples output prediction and algorithm stores decision tree margin for each prediction. Algorithm calculates final values for each prediction: (number of margins) * avg(margins) and it selects prediction with highest value.

"""

def simple_init(interface, params):
    return params

def map_init(interface, params):
    """Intialize random number generator with given seed `params.seed`."""
    import numpy as np
    import random
    np.random.seed(params['seed'])
    random.seed(params['seed'])
    
    return params

def map_fit(interface, state, label, inp):
    import numpy as np
    from itertools import permutations
    import decision_tree, measures, k_medoids

    out = interface.output(0)
    x, y, margins, forest = [], [], [], []
    attr_mapping, y_mapping, similarity_mat = {}, {}, {}
    missing_vals_attr = set()

    for row in inp:
        row = row.strip().split(state["delimiter"])
        if len(row) > 1:
            new_row = []
            for i, j in enumerate(state["X_indices"]):
                if row[j] in state["missing_vals"]:
                    new_row.append(row[j])
                    missing_vals_attr.add(i)
                elif state["X_meta"][i] == "c":
                    new_row.append(float(row[j]))
                else:
                    if row[j] not in attr_mapping:
                        attr_mapping[row[j]] = len(attr_mapping)
                    new_row.append(attr_mapping[row[j]])
            x.append(new_row)
            
            if row[state["y_index"]] not in y_mapping:
                y_mapping[row[state["y_index"]]] = len(y_mapping)
            y.append(y_mapping[row[state["y_index"]]])
    if len(y_mapping) == 1:
        print "Warning: Only one class in the subset!"
        return

    fill_in_values = []
    attr_mapping = {v:k for k,v in attr_mapping.iteritems()}
    y_mapping = {v:k for k,v in y_mapping.iteritems()}
    if len(missing_vals_attr) > 0:
        for i in range(len(state["X_indices"])):
            if state["X_meta"][i] == "c":
                value = np.average([sample[i] for sample in x if type(sample[i]) == float])
                fill_in_values.append(value)
            else:
                value = np.bincount([sample[i] for sample in x if type(sample[i]) == int]).argmax()
                fill_in_values.append(attr_mapping[value])
            if i in missing_vals_attr:
                for j in range(len(x)):
                    if x[j][i] in state["missing_vals"]:
                        x[j][i] = value
    x,y = np.array(x), np.array(y)
    
    iteration = 0
    while len(forest) < state["trees_per_chunk"]:
        if iteration == state["trees_per_chunk"]*2:
            return
        bag_indices = np.random.randint(len(x), size=(len(x)))
        unique = set(bag_indices)
        out_of_bag_indices = [i for i in range(len(x)) if i not in unique][:500]
        iteration+=1

        if len(np.unique(y[bag_indices])) == 1:
            continue

        tree = decision_tree.fit(
            x = x[bag_indices], 
            y = y[bag_indices], 
            t = state["X_meta"], 
            randomized = True, 
            max_tree_nodes = state["max_tree_nodes"], 
            min_samples_leaf=state["min_samples_leaf"], 
            min_samples_split=state["min_samples_split"],
            class_majority = state["class_majority"], 
            measure = measures.info_gain if state["measure"] == "info_gain" else measures.mdl,
            accuracy=state["accuracy"],
            separate_max=state["separate_max"])

        if len(tree) < 2:
            continue
        #calculate margins
        tree_margins, leafs_grouping = {}, {}
        for j in out_of_bag_indices:
            leaf, margin = decision_tree.predict(tree, x[j], y[j])
            tree_margins[j] = margin
            if leaf in leafs_grouping:
                leafs_grouping[leaf].append(j)
            else:
                leafs_grouping[leaf] = [j]  
        margins.append(tree_margins)
        
        for k, v in leafs_grouping.iteritems():
            for cx, cy in permutations(v,2): 
                if cx in similarity_mat:
                    similarity_mat[cx][cy] = similarity_mat[cx].get(cy, 0) - 1
                else:
                    similarity_mat[cx] = {cy: -1}
        
        tree_mapped = {}
        for k,v in tree.iteritems():
            tree_mapped[k] = [None for i in range(2)]   
            for i, node in enumerate(v):
                dist_map = dict([(y_mapping[label],freq) for label, freq in node[3].iteritems()])
                split_map = set([attr_mapping[int(s)] for s in list(node[2])]) if node[5] == "d" else node[2]
                tree_mapped[k][i] = (node[0], node[1], split_map, dist_map, node[4],node[5])
        forest.append(tree_mapped)
    
    min_elements = []
    for k, v in similarity_mat.iteritems():
        min_id = min(similarity_mat[k], key = similarity_mat[k].get) 
        min_elements.append((similarity_mat[k][min_id], min_id))
    min_elements = sorted(min_elements)
    
    if state["k"] == "sqrt":
        k = int(np.sqrt(len(x[0]))) + 1
    elif state["k"] == "square": 
        k = len(np.unique(y)) * len(np.unique(y))
    
    cidx = set()
    counter = 0
    while counter < len(min_elements) and len(cidx) < k:
        cidx.add(min_elements[counter][1])
        counter += 1
    
    inds, medoids_i = k_medoids.fit(similarity_mat, len(x), list(cidx))
    sample_ids = np.array(similarity_mat.keys())
    medoids_i = [sample_ids[i] for i in medoids_i]

    clusters = [sample_ids[np.where(inds == i)[0]] for i in np.unique(inds)]
    medoids = x[medoids_i].tolist() #set medoids without sample identifier

    cont, disc = [], []
    for i in range(len(medoids)):
        cont.append([medoids[i][j] for j in range(len(medoids[i])) if state["X_meta"][j] == "c"])
        disc.append([attr_mapping[int(medoids[i][j])] for j in range(len(medoids[i])) if state["X_meta"][j] == "d"])
    medoids = [np.array(cont), np.array(disc)]

    stats = [[] for i in range(len(medoids_i))] 
    for i in range(len(forest)): #for every tree in forest
        for num, cluster in enumerate(clusters):
            #calculate average margin for cluster
            values = [margins[i][sample_id] for sample_id in cluster if int(sample_id) in margins[i]]
            if values != []:
                avg = np.average(values)
                forest[i]["margin" + str(num)] = avg
                stats[num].append(avg)
            
    stats = [np.median(value) for value in stats]
    gower_range = np.array([np.ptp(x[:,i]) for i in range(len(state["X_meta"])) if state["X_meta"][i] == "c"])
    gower_range[gower_range == 0] = 1e-9
    out.add("model", (forest, medoids, stats, gower_range))
    out.add("fill_in_values", fill_in_values)


def reduce_fit(interface, state, label, inp):   
    import numpy as np
    out = interface.output(0)
    out.add("X_names", state["X_names"])

    forest, medoids, stats, gower_ranges, group_fillins = [],[],[],[],[]
    for i, (k, value) in enumerate(inp):
        if k == "model":
            forest.append(value[0])
            medoids.append(value[1])
            stats.append(value[2])
            gower_ranges.append(value[3])
        elif len(value) > 0:
            group_fillins.append(value)
    out.add("forest", forest)
    out.add("medoids", medoids)
    out.add("stats", stats)
    out.add("gower_ranges", gower_ranges)

    fill_in_values = []
    if len(group_fillins) > 0:
        for i, type in enumerate(state["X_meta"]):
            if type == "c":
                fill_in_values.append(np.average([sample[i] for sample in group_fillins]))
            else:
                fill_in_values.append(np.bincount([sample[i] for sample in group_fillins]).argmax())
    out.add("fill_in_values", fill_in_values)


def map_predict(interface, state, label, inp):
    import decision_tree
    import numpy as np
    
    out = interface.output(0)
    fill_in_values = state["fill_in_values"]
    coeff = state["coeff"]
    
    for row in inp:
        row = row.strip().split(state["delimiter"])
        if len(row) > 1:
            x_id = "" if state["id_index"] == -1 else row[state["id_index"]]
            
            x, cont, disc = [], [], []
            for i,j in enumerate(state["X_indices"]):
                if row[j] in state["missing_vals"]:
                    row[j] = fill_in_values[i]

                if state["X_meta"][i] == "c":
                    x.append(float(row[j]))
                    cont.append(float(row[j]))
                else:
                    x.append(row[j])
                    disc.append(row[j])
            cont, disc = np.array(cont), np.array(disc)

            similarities = []
            for i, medoids in enumerate(state["medoids"]):
                gower = 0 if len(cont) == 0 else np.sum(1 - np.true_divide(np.abs(cont - medoids[0]), state["gower_ranges"][i]), axis = 1)
                gower += 0 if len(disc) == 0 else np.sum(disc == medoids[1], axis = 1)
                similarities += zip(np.round(1 - gower/float(len(x)), 4), [(i,j) for j in range(len(x))])

            similarities = sorted(similarities)
            threshold = similarities[0][0] * (1+coeff)
            similar_medoids = [similarities[0][1]]
            pos = 1
            while pos < len(similarities) and similarities[pos][0] <= threshold:
                similar_medoids.append(similarities[pos][1])
                pos+=1

            global_predictions = {}
            for i,j in similar_medoids:
                predictions = {}
                margin = "margin"+str(j)
                for tree in state["forest"][i]:
                    if margin in tree and tree[margin] >= state["stats"][i][j]:
                        pred = decision_tree.predict(tree, x)
                        predictions[pred] = predictions.get(pred, []) + [tree[margin]]

                
                for k, v in predictions.iteritems(): 
                    predictions[k] = np.average(v) * len(v)
                
                max_pred = max(predictions, key = predictions.get)
                if max_pred not in global_predictions:
                    global_predictions[max_pred] = predictions[max_pred]
                elif predictions[max_pred] > global_predictions[max_pred]:
                    global_predictions[max_pred] = predictions[max_pred]

            out.add(x_id, (max(global_predictions, key = global_predictions.get),))         
    
def fit(input, trees_per_chunk=3, max_tree_nodes=None, min_samples_leaf=10, min_samples_split=5, class_majority=1, measure="info_gain", k="sqrt", accuracy=1, random_state=None, separate_max=True, save_results=True, show=False):
    
    from disco.worker.pipeline.worker import Worker, Stage
    from disco.core import Job
    import discomll
    
    path = "/".join(discomll.__file__.split("/")[:-1] + ["ensemble", "core",""])

    try:
        trees_per_chunk = int(trees_per_chunk)
        max_tree_nodes = int(max_tree_nodes) if max_tree_nodes != None else max_tree_nodes
        min_samples_leaf = int(min_samples_leaf)
        min_samples_split = int(min_samples_split)
        class_majority = float(class_majority)
        separate_max = separate_max

        if trees_per_chunk <= 0 or min_samples_leaf <= 0 or min_samples_split <= 0 or class_majority <= 0 or accuracy < 0:
            raise Exception("Parameters should be greater than 0.")  
    except ValueError:
        raise Exception("Parameters should be numerical.")

    if measure not in ["info_gain", "mdl"]:
        raise Exception("measure should be set to info_gain or mdl.")

    job = Job(worker = Worker(save_results = save_results))
    job.pipeline = [
    ("split", Stage("map",input_chain = input.params["input_chain"], init = map_init, process = map_fit)),
    ('group_all', Stage("reduce", init = simple_init, process = reduce_fit, combine = True))]

    job.params = input.params
    job.params["trees_per_chunk"] = trees_per_chunk
    job.params["max_tree_nodes"] = max_tree_nodes
    job.params["min_samples_leaf"] = min_samples_leaf
    job.params["min_samples_split"] = min_samples_split
    job.params["class_majority"] = class_majority
    job.params["measure"] = measure
    job.params["accuracy"] = accuracy
    job.params["k"] = k
    job.params['seed'] = random_state
    job.params['separate_max'] = separate_max

    job.run(name = "distributed_weighted_forest_fit", input = input.params["data_tag"], required_files =[path+"decision_tree.py", path+"measures.py", path + "k_medoids.py"])
    
    fitmodel_url =  job.wait(show = show)
    return {"dwf_fitmodel": fitmodel_url} #return results url


def predict(input, fitmodel_url, coeff = 0.5, save_results = True, show = False):
    from disco.worker.pipeline.worker import Worker, Stage
    from disco.core import Job, result_iterator
    import discomll
    path = "/".join(discomll.__file__.split("/")[:-1] + ["ensemble", "core",""])

    job = Job(worker = Worker(save_results = save_results))
    job.pipeline = [("split", Stage("map",input_chain = input.params["input_chain"], init = simple_init, process = map_predict))]

    if "dwf_fitmodel" not in fitmodel_url:
        raise Exception("Incorrect fit model.")

    try:
        coeff = float(coeff)
        if coeff < 0:
            raise Exception("Parameter coeff should be greater than 0.")  
    except ValueError:
        raise Exception("Parameter coeff should be numerical.")
    
    job.params = input.params
    job.params["coeff"] = coeff
    for k, v in result_iterator(fitmodel_url["dwf_fitmodel"]):
        job.params[k] = v

    if len(job.params["forest"]) == 0:
        print "Warning: There is no decision trees in forest"
        return []

    job.run(name = "distributed_weighted_forest_predict", input = input.params["data_tag"], required_files = [path+"decision_tree.py"])
    
    return job.wait(show = show)
























