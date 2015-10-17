"""
Linear SVM with MapReduce

Algorithm builds a model with continuous features and predicts binary target label (-1, 1). 

Reference
Algorithm is proposed by Glenn Fung, O. L. Mangasarian. Incremental Support Vector Machine Classification. Description of algorithm can be found at ftp://ftp.cs.wisc.edu/pub/dmi/tech-reports/01-08.pdf. 
"""


def simple_init(interface, params):
    return params


def map_fit(interface, state, label, inp):
    """
    Function calculates matrices ete and etde for every sample, aggregates and output them.
    """
    import numpy as np
    ete, etde = 0, 0
    out = interface.output(0)

    for row in inp:
        row = row.strip().split(state["delimiter"])  # split row
        if len(row) > 1:  # check if row is empty
            # intercept term is added to every sample
            x = np.array([(0 if v in state["missing_vals"] else float(v)) for i, v in enumerate(row) if
                          i in state["X_indices"]] + [-1])
            # map label value to 1 or -1. If label does not match set error
            y = 1 if state["y_map"][0] == row[state["y_index"]] else -1 if state["y_map"][1] == row[
                state["y_index"]] else "Error"
            ete += np.outer(x, x)
            etde += x * y
    out.add("etde", etde)
    for i, row in enumerate(ete):
        out.add(i, row)


def reduce_fit(interface, state, label, inp):
    """
    Function joins all partially calculated matrices ETE and ETDe, aggregates them and it calculates final parameters.
    """
    import numpy as np

    out = interface.output(0)
    sum_etde = 0
    sum_ete = [0 for _ in range(len(state["X_indices"]) + 1)]
    for key, value in inp:
        if key == "etde":
            sum_etde += value
        else:
            sum_ete[key] += value

    sum_ete += np.true_divide(np.eye(len(sum_ete)), state["nu"])
    out.add("params", np.linalg.lstsq(sum_ete, sum_etde)[0])


def map_predict(interface, state, label, inp):
    import numpy as np
    out = interface.output(0)

    for row in inp:
        row = row.strip().split(state["delimiter"])
        if len(row) > 1:
            # set id of current sample
            x_id = "" if state["id_index"] == -1 else row[state["id_index"]]
            # add intercept term
            x = [(0 if v in state["missing_vals"] else float(v)) for i, v in enumerate(row) if
                 i in state["X_indices"]] + [-1]

            # make a prediction with parameters
            value = np.dot(x, state["fit_params"])
            y = state["y_map"][0] if value >= 0 else state["y_map"][1]
            out.add(x_id, (y,))


def fit(dataset, nu=0.1, save_results=True, show=False):
    """
    Function starts a job for calculation of model parameters

    Parameters
    ----------
    input - dataset object with input urls and other parameters
    nu - parameter to adjust the classifier
    save_results - save results to ddfs
    show - show info about job execution

    Returns
    -------
    Urls of fit model results on ddfs
    """
    from disco.worker.pipeline.worker import Worker, Stage
    from disco.core import Job

    if dataset.params["y_map"] == []:
        raise Exception("Linear proximal SVM requires a target label mapping parameter.")
    try:
        nu = float(nu)
        if nu <= 0:
            raise Exception("Parameter nu should be greater than 0")
    except ValueError:
        raise Exception("Parameter should be numerical.")

    job = Job(worker=Worker(save_results=save_results))

    # job parallelizes mappers and joins them with one reducer
    job.pipeline = [
        ("split", Stage("map", input_chain=dataset.params["input_chain"], init=simple_init, process=map_fit)),
        ('group_all', Stage("reduce", init=simple_init, process=reduce_fit, combine=True))]

    job.params = dataset.params
    job.params["nu"] = nu
    job.run(name="linearsvm_fit", input=dataset.params["data_tag"])
    fitmodel_url = job.wait(show=show)
    return {"linsvm_fitmodel": fitmodel_url}  # return results url


def predict(dataset, fitmodel_url, save_results=True, show=False):
    """
    Function starts a job that makes predictions to input data with a given model.

    Parameters
    ----------
    input - dataset object with input urls and other parameters
    fitmodel_url - model created in fit phase
    save_results - save results to ddfs
    show - show info about job execution

    Returns
    -------
    Urls with predictions on ddfs
    """
    from disco.worker.pipeline.worker import Worker, Stage
    from disco.core import Job, result_iterator

    if "linsvm_fitmodel" not in fitmodel_url:
        raise Exception("Incorrect fit model.")

    job = Job(worker=Worker(save_results=save_results))
    # job parallelizes execution of mappers
    job.pipeline = [
        ("split", Stage("map", input_chain=dataset.params["input_chain"], init=simple_init, process=map_predict))]

    job.params = dataset.params
    job.params["fit_params"] = [v for _, v in result_iterator(fitmodel_url["linsvm_fitmodel"])][0]
    job.run(name="linsvm_predict", input=dataset.params["data_tag"])

    return job.wait(show=show)
