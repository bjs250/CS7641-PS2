import constants

import numpy as np
from mlrose_hiive.generators import ContinuousPeaksGenerator
from mlrose_hiive.runners import RHCRunner, SARunner, GARunner, MIMICRunner
from mlrose_hiive.algorithms import ExpDecay

problem_name = "kcolors"
algorithm_names = ["RHC", "SA", "GA", "MIMIC"]
problem_sizes = [20]
T = [0.05, 0.10, 0.20]

for problem_size in problem_sizes:
    for t in T:
        for algorithm_name in algorithm_names:

            experiment_name = '{0}-{1}-{2}-{3}'.format(problem_name, algorithm_name, problem_size, t)
            problem = ContinuousPeaksGenerator.generate(
                seed=0,
                size=problem_size,
                t_pct=t
            )
            if algorithm_name == "RHC" and True:
                rhc = RHCRunner(problem=problem,
                                experiment_name=experiment_name,
                                output_directory=constants.OUTPUT_DIRECTORY,
                                seed=0,
                                iteration_list=constants.iteration_list,
                                max_attempts=20000,
                                restart_list=[5]
                                )
                df_run_stats, df_run_curves = rhc.run()
            if algorithm_name == "SA" and False:
                sa = SARunner(problem=problem,
                              experiment_name=experiment_name,
                              output_directory=constants.OUTPUT_DIRECTORY,
                              seed=0,
                              iteration_list=constants.iteration_list,
                              max_attempts=2000,
                              temperature_list=[500],
                              decay_list=[ExpDecay])
                sa_run_stats, sa_run_curves = sa.run()
            if algorithm_name == "GA" and False:
                ga = GARunner(problem=problem,
                              experiment_name=experiment_name,
                              output_directory=constants.OUTPUT_DIRECTORY,
                              seed=0,
                              iteration_list=2 ** np.arange(16),
                              max_attempts=2000,
                              population_sizes=[10],
                              mutation_rates=[0.05])
                ga_run_stats, ga_run_curves = ga.run()
            if algorithm_name == "MIMIC" and False:
                mimic = MIMICRunner(problem=problem,
                                    experiment_name=experiment_name,
                                    output_directory=constants.OUTPUT_DIRECTORY,
                                    seed=0,
                                    iteration_list=2 ** np.arange(16),
                                    population_sizes=[10],
                                    max_attempts=500,
                                    keep_percent_list=[0.05],
                                    use_fast_mimic=True)
                mimic_run_stats, mimic_run_curves = mimic.run()
