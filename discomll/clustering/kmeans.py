"""
kmeans with MapReduce

k-means is a partitional clustering technique that attempts to find a user-specified number of clusters k represented by their centroids.

Implementation is taken from Disco dataming examples and addapted to work with discomll
"""

# HACK: The following dictionary will be transformed into a class once
# class support in Params has been added to Disco.
mean_point_center = {
    'create':(lambda x,w: { '_x':x, 'w':w }),
    'update':(lambda p,q: { '_x':[ pxi+qxi for pxi,qxi in zip(p['_x'],q['_x']) ], 'w':p['w']+q['w'] }),
    'finalize':(lambda p: { 'x':[v/p['w'] for v in p['_x']],'_x':p['_x'], 'w':p['w'] }),
    'dist':(lambda p,x: sum((pxi-xi)**2 for pxi,xi in zip(p['x'],x)) )
    }

def simple_init(interface, params):
    return params

def map_init(interface, params):
    """Intialize random number generator with given seed `params.seed`."""
    import random
    random.seed(params['seed'])
    return params

def random_init_map(interface, state, label, inp):
    """Assign datapoint `e` randomly to one of the `k` clusters."""
    import random
    out = interface.output(0)
    centers = {}

    for row in inp:
        row = row.strip().split(state["delimiter"])
        if len(row) > 1:
            x = [(0 if row[i] in state["missing_vals"] else float(row[i])) for i in state["X_indices"]]
            cluster = random.randint(0, state['k']-1)
            vertex = state['create'](x, 1.0)
            centers[cluster] = vertex if cluster not in centers else state["update"](centers[cluster], vertex)
    for cluster, values in centers.iteritems():
        out.add(cluster, values) 

def estimate_map(interface, state, label, inp):
    """Find the cluster `i` that is closest to the datapoint `e`."""
    out = interface.output(0)
    centers = {}
    for row in inp:
        row = row.strip().split(state["delimiter"])
        if len(row) > 1:
            x = [(0 if row[i] in state["missing_vals"] else float(row[i])) for i in state["X_indices"]]
            cluster = min((state['dist'](c, x), i) for i,c in state['centers'])[1]
            vertex = state['create'](x, 1.0)
            centers[cluster] = vertex if cluster not in centers else state["update"](centers[cluster], vertex)

    for cluster, values in centers.iteritems():
            out.add(cluster, values) 


def estimate_reduce(interface, state, label, inp):
    """Estimate the cluster centers for each cluster."""
    centers = {}
    for i, c in inp:
        centers[i] = c if i not in centers else state['update'](centers[i], c)

    out = interface.output(0)
    for i, c in centers.items():
        out.add(i, state['finalize'](c))  

def predict_map(interface, state, label, inp):
    """Determine the closest cluster for the datapoint `e`."""
    out = interface.output(0)
    for row in inp:
        if len(row) > 1:
            row = row.strip().split(state["delimiter"])
            x_id = "" if state["id_index"] == -1 else row[state["id_index"]]
            x = [(0 if row[i] in state["missing_vals"] else float(row[i])) for i in state["X_indices"]]
            out.add(x_id,  min([(i, state["dist"](c, x)) for i,c in state["centers"]], key = lambda t: t[1]))
        
def fit(input, n_clusters = 5, max_iterations = 10, random_state = None, save_results = True, show = False):
    """
    Optimize k-clustering for `iterations` iterations with cluster
    center definitions as given in `center`.
    """
    from disco.job import Job
    from disco.worker.pipeline.worker import Worker,Stage
    from disco.core import result_iterator

    try:
        n_clusters = int(n_clusters)
        max_iterations = int(max_iterations)
        if n_clusters < 2:
            raise Exception("Parameter n_clusters should be greater than 1.")  
        if max_iterations < 1:
            raise Exception("Parameter max_iterations should be greater than 0.")  
    except ValueError:
        raise Exception("Parameters should be numerical.")

    
    job = Job(worker = Worker(save_results = save_results))
    job.pipeline = [("split",
                 Stage("kmeans_init_map", input_chain = input.params["input_chain"], init = map_init, process = random_init_map)),
                ('group_label', Stage("kmeans_init_reduce", process = estimate_reduce, init = simple_init, combine = True))]
    job.params = dict(input.params.items() + mean_point_center.items())
    job.params['seed'] = random_state
    job.params['k'] = n_clusters
    
    job.run(input = input.params["data_tag"], name = "kmeans_init")
    init = job.wait(show = show)
    centers = [(i,c) for i,c in result_iterator(init)]
    
    for j in range(max_iterations):
        job = Job(worker = Worker(save_results = save_results))
        job.params = dict(input.params.items() + mean_point_center.items())
        job.params['k'] = n_clusters
        job.params['centers'] = centers

        job.pipeline = [('split', Stage("kmeans_map_iter_%s" %(j+1,),
                input_chain = input.params["input_chain"],
                process=estimate_map, init = simple_init)),
            ('group_label', Stage("kmeans_reduce_iter_%s" %(j+1,),
                process=estimate_reduce, init = simple_init, combine = True))]

        job.run(input = input.params["data_tag"], name = 'kmeans_iter_%d' %(j+1,))
        fitmodel_url = job.wait(show = show)
        centers = [(i,c) for i,c in result_iterator(fitmodel_url)]
        
    return {"kmeans_fitmodel": fitmodel_url} #return results url
    
def predict(input, fitmodel_url, save_results = True, show = False):
    """
    Predict the closest clusters for the datapoints in input.
    """

    from disco.job import Job
    from disco.worker.pipeline.worker import Worker,Stage
    from disco.core import result_iterator

    if "kmeans_fitmodel" not in fitmodel_url:
        raise Exception("Incorrect fit model.")

    job = Job(worker = Worker(save_results = save_results))
    job.params = dict(input.params.items() + mean_point_center.items())
    job.params["centers"] = [(i,c) for i,c in result_iterator(fitmodel_url["kmeans_fitmodel"])]
    
    job.pipeline = [("split", Stage("kmeans_predict", input_chain = input.params["input_chain"], init = simple_init, process = predict_map))]
    
    job.run(input = input.params["data_tag"], name="kmeans_predict")

    return job.wait(show = show)


























