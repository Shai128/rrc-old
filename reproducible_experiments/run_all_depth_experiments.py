from itertools import product
import time
from depth_main import SMALL_GAMMA_SET
from run_experiment import run_experiment

def cartesian_product(inp):
    if len(inp) == 0:
        return []
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


datasets = ['KITTI']

processes_to_run_in_parallel = 1

seeds = list(range(0, 10))
suppress_plots = 0
alphas = [0.2]

single_risk_params = {
    'main_program_name': ['depth_main'],
    'seed': seeds,
    'dataset_name': datasets,
    'alpha': alphas,
    'ds_type': ['REAL'],
    'backbone': ['res101'],
    'uq_method': ["baseline", "residual_magnitude", 'previous_residual', 'previous_residual_with_flow'],
    'suppress_plots': [suppress_plots],
    'option': [-1],
}

multiple_risk_params = {
    'main_program_name': ['depth_main'],
    'seed': seeds,
    'dataset_name': datasets,
    'alpha': alphas,
    'ds_type': ['REAL'],
    'backbone': ['res101'],
    'uq_method': ['previous_residual_with_flow'],
    'suppress_plots': [suppress_plots],
    'option': list(range(len(SMALL_GAMMA_SET))),
}

params = []
params += list(cartesian_product(single_risk_params))
params += list(cartesian_product(multiple_risk_params))

processes_to_run_in_parallel = min(processes_to_run_in_parallel, len(params))
run_on_slurm = False
cpus = 2
gpus = 0
if __name__ == '__main__':

    print("jobs to do: ", len(params))
    # initializing processes_to_run_in_parallel workers
    workers = []
    jobs_finished_so_far = 0
    assert len(params) >= processes_to_run_in_parallel
    for _ in range(processes_to_run_in_parallel):
        curr_params = params.pop(0)
        main_program_name = curr_params['main_program_name']
        curr_params.pop('main_program_name')
        p = run_experiment(curr_params, main_program_name, run_on_slurm=run_on_slurm,
                           cpus=cpus, gpus=gpus)
        workers.append(p)

    # creating a new process when an old one dies
    while len(params) > 0:
        dead_workers_indexes = [i for i in range(len(workers)) if (workers[i].poll() is not None)]
        for i in dead_workers_indexes:
            worker = workers[i]
            worker.communicate()
            jobs_finished_so_far += 1
            if len(params) > 0:
                curr_params = params.pop(0)
                main_program_name = curr_params['main_program_name']
                curr_params.pop('main_program_name')
                p = run_experiment(curr_params, main_program_name, run_on_slurm=run_on_slurm,
                                   cpus=cpus, gpus=gpus)
                workers[i] = p
                if jobs_finished_so_far % processes_to_run_in_parallel == 0:
                    print(f"finished so far: {jobs_finished_so_far}, {len(params)} jobs left")
            time.sleep(10)

    # joining all last proccesses
    for worker in workers:
        worker.communicate()
        jobs_finished_so_far += 1

    print("finished all")
