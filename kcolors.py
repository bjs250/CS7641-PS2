import constants

import numpy as np
from mlrose_hiive.generators import MaxKColorGenerator
from mlrose_hiive.runners import RHCRunner, SARunner, GARunner, MIMICRunner
from mlrose_hiive.algorithms import ExpDecay

problem_name = "kcolors"
algorithm_names = ["RHC", "SA", "GA", "MIMIC"]
problem_sizes = [10, 25, 50, 75, 100]

for problem_size in problem_sizes:
    for algorithm_name in algorithm_names:

        experiment_name = '{0}-{1}-{2}'.format(problem_name, algorithm_name, problem_size)
        problem = MaxKColorGenerator.generate(
            seed=0,
            number_of_nodes=problem_size,
            max_connections_per_node=int(np.floor(problem_size/2)),
            max_colors=int(np.floor(problem_size/2))
        )
        if algorithm_name == "RHC" and True:
            rhc = RHCRunner(problem=problem,
                            experiment_name=experiment_name,
                            output_directory=constants.OUTPUT_DIRECTORY,
                            seed=0,
                            iteration_list=range(0, 2000, 100),
                            max_attempts=100,
                            restart_list=[5]
                            )
            df_run_stats, df_run_curves = rhc.run()
        if algorithm_name == "SA" and True:
            sa = SARunner(problem=problem,
                          experiment_name=experiment_name,
                          output_directory=constants.OUTPUT_DIRECTORY,
                          seed=0,
                          iteration_list=range(0, 2000, 100),
                          max_attempts=100,
                          temperature_list=[10],
                          decay_list=[ExpDecay])
            sa_run_stats, sa_run_curves = sa.run()
        if algorithm_name == "GA" and True:
            ga = GARunner(problem=problem,
                          experiment_name=experiment_name,
                          output_directory=constants.OUTPUT_DIRECTORY,
                          seed=0,
                          iteration_list=constants.iteration_list,
                          max_attempts=100,
                          population_sizes=[100],
                          mutation_rates=[0.05])
            ga_run_stats, ga_run_curves = ga.run()
        if algorithm_name == "MIMIC" and True:
            mimic = MIMICRunner(problem=problem,
                                experiment_name=experiment_name,
                                output_directory=constants.OUTPUT_DIRECTORY,
                                seed=0,
                                iteration_list=[1,2,3,4,5,10,15,20,25,30],
                                population_sizes=[200],
                                max_attempts=500,
                                keep_percent_list=[0.3],
                                use_fast_mimic=True)
            mimic_run_stats, mimic_run_curves = mimic.run()
