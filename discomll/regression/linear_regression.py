"""
Linear regression with MapReduce

The linear regression fits theta parameters to training data.

Reference:
MapReduce version of algorithm is proposed by Cheng-Tao Chu; Sang Kyun Kim, Yi-An Lin, YuanYuan Yu, Gary Bradski, Andrew Ng, and Kunle Olukotun. "Map-Reduce for Machine Learning on Multicore". NIPS 2006.   

"""

def simple_init(interface, params):
    return params

def map_fit(interface, state, label, inp):
    import numpy as np
    A,b = 0,0
    out = interface.output(0)

    for row in inp:
        row = row.strip().split(state["delimiter"])
        if len(row) > 1:
            x = np.array([1] + [(0 if v in state["missing_vals"] else float(v)) for i, v in enumerate(row) if i in state["X_indices"]])
            y = float(row[state["y_index"]])
            A += np.outer(x,x) 
            b += x * y
    out.add("b", b)
    for i, row in enumerate(A):
        out.add(i, row)

def reduce_fit(interface, state, label, inp):
    import numpy as np
    
    out = interface.output(0)
    sum_b = 0
    sum_A = [0 for i in range(len(state["X_indices"])+1)]
    for key,value in inp:
        if key == "b":
            sum_b += value 
        else:
            sum_A[key] += value

    out.add("thetas", np.linalg.lstsq(sum_A, sum_b)[0])


def map_predict(interface, state, label, inp):
    import numpy as np
    A,b = 0,0
    out = interface.output(0)

    for row in inp:
        row = row.strip().split(state["delimiter"])
        if len(row) > 1:
            x_id = "" if state["id_index"] == -1 else row[state["id_index"]]
            x = np.array([1] + [(0 if v in state["missing_vals"] else float(v)) for i, v in enumerate(row) if i in state["X_indices"]])
            out.add(x_id, (np.dot(x, state["thetas"]),))


def fit(input, save_results = True, show = False):
    from disco.worker.pipeline.worker import Worker, Stage
    from disco.core import Job
    job = Job(worker = Worker(save_results = save_results))
    
    job.pipeline = [
    ("split", Stage("map",input_chain = input.params["input_chain"], init = simple_init, process = map_fit)),
    ('group_all', Stage("reduce", init = simple_init, process = reduce_fit, combine = True))]

    job.params = input.params
    job.run(name = "linreg_fit", input = input.params["data_tag"])
    
    fitmodel_url =  job.wait(show = show)
    return {"linreg_fitmodel": fitmodel_url} #return results url

def predict(input, fitmodel_url, save_results = True, show = False):
    from disco.worker.pipeline.worker import Worker, Stage
    from disco.core import Job, result_iterator
    
    if "linreg_fitmodel" not in fitmodel_url:
        raise Exception("Incorrect fit model.")

    job = Job(worker = Worker(save_results = save_results))
    job.pipeline = [
    ("split", Stage("map",input_chain = input.params["input_chain"], init = simple_init, process = map_predict))]
    job.params = input.params
    job.params["thetas"] = [v for _,v in result_iterator(fitmodel_url["linreg_fitmodel"])][0]

    job.run(name = "linreg_predict", input = input.params["data_tag"])
    return job.wait(show = show)


















