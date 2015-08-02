"""
Forest of Distributed Decision Trees

Fit phase
Decision trees algorithm builds one decision tree on a subset of data and it estimates all attributes in every tree node.

Predict phase
Each tree votes and algorithm selects prediction with most votes.

Reference
Similar algorithm is proposed in Gongqing Wu, Haiguang Li, Xuegang Hu, Yuanjun Bi, Jing Zhang, and Xindong Wu. MRec4.5: C4. 5 ensemble classification with mapreduce.
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
    
    attr_mapping, y_mapping = {}, {}
    x, y, fill_in_values = [], [], []
    out = interface.output(0)
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

    tree = decision_tree.fit(
        x = np.array(x, dtype=np.float32), 
        y = np.array(y, dtype=np.uint16), 
        t = state["X_meta"], 
        randomized = False, 
        max_tree_nodes=state["max_tree_nodes"], 
        min_samples_leaf=state["min_samples_leaf"], 
        min_samples_split=state["min_samples_split"],
        class_majority = state["class_majority"], 
        measure = measures.info_gain if state["measure"] == "info_gain" else measures.mdl,
        accuracy=state["accuracy"],
        separate_max=state["separate_max"])

    tree_mapped = {}
    for k,v in tree.iteritems():
        tree_mapped[k] = [None for i in range(2)]   
        for i, node in enumerate(v):
            dist_map = dict([(y_mapping[label],freq) for label, freq in node[3].iteritems()])
            split_map = set([attr_mapping[int(s)] for s in list(node[2])]) if node[5] == "d" else node[2]
            tree_mapped[k][i] = (node[0], node[1], split_map, dist_map, node[4],node[5])
    out.add("tree", tree_mapped)
    out.add("fill_in_values", fill_in_values)

def map_fit_bootstrap(interface, state, label, inp):
    import numpy as np
    import decision_tree, measures
    from collections import Counter
    
    out = interface.output(0)
    num_samples = sum([1 for row in inp if len(row.strip().split(state["delimiter"])) > 1])
    
    for counter in range(state["trees_per_chunk"]):
        bag_indices = Counter(np.random.randint(num_samples, size=(num_samples)))
        x, y, fill_in_values = [], [], []
        row_num = 0
        attr_mapping, y_mapping = {}, {}
        
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
                            new_row.append(float(row[j]))
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

        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.uint16)
        tree = decision_tree.fit(
            x = x, 
            y = y, 
            t = state["X_meta"], 
            randomized = False, 
            max_tree_nodes=state["max_tree_nodes"], 
            min_samples_leaf=state["min_samples_leaf"], 
            min_samples_split=state["min_samples_split"],
            class_majority = state["class_majority"], 
            measure = measures.info_gain if state["measure"] == "info_gain" else measures.mdl,
            accuracy=state["accuracy"],
            separate_max=state["separate_max"])
        
        if len(tree) < 2:
            continue
    
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
    import numpy as np
    import decision_tree
    
    out = interface.output(0)
    half_ensemble = round(len(state["forest"])/2.)
    fill_in_values = state["fill_in_values"]
    
    for row in inp:
        row = row.strip().split(state["delimiter"])
        if len(row) > 1:
            x_id = "" if state["id_index"] == -1 else row[state["id_index"]]
            x = [(fill_in_values[j] if row[j] in state["missing_vals"] else float(row[j]) if state["X_meta"][i] == "c" else row[j]) for i,j in enumerate(state["X_indices"])]
            
            predictions = {}
            for i, tree in enumerate(state["forest"]):
                pred = decision_tree.predict(tree, x)
                predictions[pred] = predictions.get(pred, 0) + 1 
                
                if i >= half_ensemble-1:
                    prediction = max(predictions, key=predictions.get)
                    value = predictions[prediction]
                    if value == half_ensemble:
                        break
            out.add(x_id, (prediction, i+1))

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


def fit(input, trees_per_chunk=1, bootstrap=True, max_tree_nodes=None, min_samples_leaf=10, min_samples_split=5, class_majority=1, separate_max=True, measure="info_gain", accuracy=1, random_state=None, save_results=True, show=False):
    
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
        accuracy = int(accuracy)
        separate_max = separate_max
        if trees_per_chunk > 1 and bootstrap == False:
            print "Warning: bootstrap was set to true. trees_per_chunk should be 1 to disable bootstrap."
            bootstrap = True
        if trees_per_chunk <= 0 or min_samples_leaf <= 0 or class_majority <= 0 or min_samples_split <= 0 and accuracy < 0 or type(bootstrap) != bool:
            raise Exception("Parameters should be greater than 0.")  
    except ValueError:
        raise Exception("Parameters should be numerical.")

    if measure not in ["info_gain", "mdl"]:
        raise Exception("measure should be set to info_gain or mdl.")

    job = Job(worker = Worker(save_results = save_results))
    job.pipeline = [
    ("split", Stage("map",input_chain = input.params["input_chain"], init = map_init, process = map_fit_bootstrap if bootstrap else map_fit)),
    ('group_all', Stage("reduce", init = simple_init, process = reduce_fit, combine = True))]

    job.params = input.params
    job.params["trees_per_chunk"] = trees_per_chunk
    job.params["max_tree_nodes"] = max_tree_nodes
    job.params["min_samples_leaf"] = min_samples_leaf
    job.params["min_samples_split"] = min_samples_split
    job.params["class_majority"] = class_majority
    job.params["measure"] = measure
    job.params["bootstrap"] = bootstrap
    job.params["accuracy"] = accuracy
    job.params["separate_max"] = separate_max
    job.params['seed'] = random_state

    job.run(name = "forest_distributed_decision_trees_fit", input = input.params["data_tag"], required_files =[path+"decision_tree.py", path+"measures.py"])
    
    fitmodel_url =  job.wait(show = show)
    return {"fddt_fitmodel": fitmodel_url} #return results url

def predict(input, fitmodel_url, voting=False, save_results = True, show = False):
    from disco.worker.pipeline.worker import Worker, Stage
    from disco.core import Job, result_iterator
    import discomll
    path = "/".join(discomll.__file__.split("/")[:-1] + ["ensemble", "core",""])

    if "fddt_fitmodel" not in fitmodel_url:
        raise Exception("Incorrect fit model.")

    job = Job(worker = Worker(save_results = save_results))
    job.pipeline = [("split", Stage("map",input_chain = input.params["input_chain"], init = simple_init, process = map_predict_voting if voting else map_predict_dist))]

    job.params = input.params
    for k, v in result_iterator(fitmodel_url["fddt_fitmodel"]):
        job.params[k] = v

    if len(job.params["forest"]) == 0:
        print "Warning: There is no decision trees in forest"
        return []

    job.run(name = "forest_distributed_decision_trees_predict", input = input.params["data_tag"], required_files = [path+"decision_tree.py"])
    
    return job.wait(show = show)


























