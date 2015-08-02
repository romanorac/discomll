"""
Random forest with MapReduce

Fit phase
Random forest algorithm builds multiple decision trees with a bootstrap method on a subset of data. In each tree node, it estimates sqrt(num. of attributes)+1 randomly selected attributes (without replacement). All decision trees are merged in large ensemble.  

Predict phase
Algorithm queries as many trees as needed for reliable prediction. Firstly, it randomly chooses without replacement 15 trees. If all trees vote for the same class, it outputs prediction. If there are multiple classes predicted, it chooses 15 trees again. Algorithm calculates difference in probability between most and second most probable prediction. If difference is greater than parameter diff, it outputs prediction. If a test sample is hard to predict (difference is never higher than diff), it queries whole ensemble to make a prediction.

Reference
Similar algorithm is proposed in: Justin D Basilico, M Arthur Munson, Tamara G Kolda, Kevin R Dixon, and W Philip Kegelmeyer. Comet: A recipe for learning and using large ensembles on massive data. 

"""

def simple_init(interface, params):
    return params

def map_init(interface, params):
    """Intialize random number generator with given seed `params.seed`."""
    import random
    import numpy as np
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    return params

def map_fit(interface, state, label, inp):
    import numpy as np
    import decision_tree, measures
    from collections import Counter 
    
    out = interface.output(0)
    num_samples = sum([1 for row in inp if len(row.strip().split(state["delimiter"])) > 1])
    missing_vals_attr = set()
    
    for counter in range(state["trees_per_chunk"]):
        bag_indices = Counter(np.random.randint(num_samples, size=(num_samples)))
        attr_mapping, y_mapping = {}, {}
        x, y, fill_in_values = [], [], []
        row_num = 0
        for row in inp:
            row = row.strip().split(state["delimiter"])
            if len(row) > 1:
                while bag_indices[row_num] > 0:
                    new_row = []
                    for i, j in enumerate(state["X_indices"]):
                        if row[j] in state["missing_vals"]:
                            new_row.append(row[j])
                            missing_vals_attr.add(i)
                        elif state["X_meta"][i] == "c":
                            new_row.append(row[j])
                        else:
                            if row[j] not in attr_mapping:
                                attr_mapping[row[j]] = len(attr_mapping)
                            new_row.append(attr_mapping[row[j]])
                    x.append(np.array(new_row, dtype=np.float32))
                    if row[state["y_index"]] not in y_mapping:
                        y_mapping[row[state["y_index"]]] = len(y_mapping)
                    y.append(y_mapping[row[state["y_index"]]])
                    bag_indices[row_num]-=1
                row_num+=1

        attr_mapping = {v:k for k,v in attr_mapping.iteritems()}
        y_mapping = {v:k for k,v in y_mapping.iteritems()}

        if len(y_mapping) == 1:
            print "Warning: Only one class in the subset!"
            return

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
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.uint16)
        
        tree = decision_tree.fit(
            x = x, 
            y = y, 
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
        print "tree was build"
        tree_mapped = {}
        for k,v in tree.iteritems():
            tree_mapped[k] = [None for i in range(2)]   
            for i, node in enumerate(v):
                dist_map = dict([(y_mapping[label],freq) for label, freq in node[3].iteritems()])
                split_map = set([attr_mapping[int(s)] for s in list(node[2])]) if node[5] == "d" else node[2]
                tree_mapped[k][i] = (node[0], node[1], split_map, dist_map, node[4],node[5])
        out.add("tree", tree_mapped)
        out.add("fill_in_values", fill_in_values)
    
def reduce_fit(interface, state, label, inp):   
    import numpy as np  
    out = interface.output(0)
    out.add("X_names", state["X_names"])

    forest = []
    group_fillins = []
    for i, (k, value) in enumerate(inp):
        if k == "tree":
            forest.append(value)
        elif len(value) > 0:
            group_fillins.append(value)
    out.add("forest", forest)

    fill_in_values = []
    if len(group_fillins) > 0:
        for i, type in enumerate(state["X_meta"]):
            if type == "c":
                fill_in_values.append(np.average([sample[i] for sample in group_fillins]))
            else:
                fill_in_values.append(np.bincount([sample[i] for sample in group_fillins]).argmax())
    out.add("fill_in_values", fill_in_values)

def map_predict_voting(interface, state, label, inp):
    import decision_tree
    
    out = interface.output(0)
    fill_in_values = state["fill_in_values"]

    for row in inp:
        row = row.strip().split(state["delimiter"])
        predicted = False
        if len(row) > 1:
            x_id = "" if state["id_index"] == -1 else row[state["id_index"]]
            x = [(fill_in_values[j] if row[j] in state["missing_vals"] else float(row[j]) if state["X_meta"][i] == "c" else row[j]) for i,j in enumerate(state["X_indices"])]

            tallies = {}
            for tree in state["forest"]:
                pred = decision_tree.predict(tree, x)
                tallies[pred] = tallies.get(pred, 0) + 1
                if any(e > int(len(state["forest"])/2.) for e in tallies.values()):
                    prediction = max(tallies, key=tallies.get)
                    out.add(x_id, (prediction, tallies[prediction]))
                    predicted = True
                    break
            if not predicted:
                prediction = max(tallies, key=tallies.get)
                out.add(x_id, (prediction, tallies[prediction]))


def map_predict_dist(interface, state, label, inp):
    import numpy as np
    import decision_tree
    
    out = interface.output(0)
    ensemble_size = len(state["forest"])
    fill_in_values = state["fill_in_values"]

    for row in inp:
        row = row.strip().split(state["delimiter"])
        if len(row) > 1:
            x_id = "" if state["id_index"] == -1 else row[state["id_index"]]
            x = [(fill_in_values[j] if row[j] in state["missing_vals"] else float(row[j]) if state["X_meta"][i] == "c" else row[j]) for i,j in enumerate(state["X_indices"])]
            
            pred_dist = [decision_tree.predict(tree, x, dist=True) for tree in state["forest"]]
            y_dist = {k:v/float(ensemble_size) for k, v in np.sum(pred_dist).iteritems()}
            prediction = max(y_dist, key=y_dist.get)
            out.add(x_id, (prediction, y_dist[prediction]))

def fit(input, trees_per_chunk=3, max_tree_nodes=None, min_samples_leaf=10, min_samples_split=5, class_majority=1, measure="info_gain", accuracy=1, separate_max=True, random_state=None, save_results=True, show=False):

    from disco.worker.pipeline.worker import Worker, Stage
    from disco.core import Job
    import discomll
    path = "/".join(discomll.__file__.split("/")[:-1] + ["ensemble", "core",""])

    job = Job(worker = Worker(save_results = save_results))

    job.pipeline = [
    ("split", Stage("map",input_chain = input.params["input_chain"], init = map_init, process = map_fit)),
    ('group_all', Stage("reduce", init = simple_init, process = reduce_fit, combine = True))]

    try:
        trees_per_chunk = int(trees_per_chunk)
        max_tree_nodes = int(max_tree_nodes) if max_tree_nodes != None else max_tree_nodes
        min_samples_leaf = int(min_samples_leaf)
        min_samples_split = int(min_samples_split)
        class_majority = float(class_majority)
        accuracy = int(accuracy)

        if trees_per_chunk <= 0 or min_samples_leaf <= 0 or class_majority <= 0 or min_samples_split <= 0 and accuracy < 0:
            raise Exception("Parameters should be greater than 0.")  
    except ValueError:
        raise Exception("Parameters should be numerical.")

    if measure not in ["info_gain", "mdl"]:
        raise Exception("measure should be set to info_gain or mdl.")

    job.params = input.params
    job.params["trees_per_chunk"] = trees_per_chunk
    job.params["max_tree_nodes"] = max_tree_nodes
    job.params["min_samples_leaf"] = min_samples_leaf
    job.params["min_samples_split"] = min_samples_split
    job.params["class_majority"] = class_majority
    job.params["measure"] = measure
    job.params["accuracy"] = accuracy
    job.params["separate_max"] = separate_max
    job.params['seed'] = random_state

    job.run(name = "distributed_random_forest_fit", input = input.params["data_tag"], required_files =[path+"decision_tree.py", path+"measures.py"])
    
    fitmodel_url =  job.wait(show = show)
    return {"drf_fitmodel": fitmodel_url} #return fitmodel url

def predict(input, fitmodel_url, voting=False, save_results=True, show=False):
    from disco.worker.pipeline.worker import Worker, Stage
    from disco.core import Job, result_iterator
    import discomll
    
    path = "/".join(discomll.__file__.split("/")[:-1] + ["ensemble", "core",""])

    if "drf_fitmodel" not in fitmodel_url:
        raise Exception("Incorrect fit model.")

    job = Job(worker = Worker(save_results = save_results))
    job.pipeline = [("split", Stage("map",input_chain = input.params["input_chain"],init = simple_init, process = map_predict_voting if voting else map_predict_dist))]

    job.params = input.params
    for k, v in result_iterator(fitmodel_url["drf_fitmodel"]):
        job.params[k] = v

    if len(job.params["forest"]) == 0:
        print "Warning: There is no decision trees in forest"
        return []

    job.run(name = "distributed_random_forest_predict", input = input.params["data_tag"], required_files = [path+"decision_tree.py"])
    
    return job.wait(show = show)


























