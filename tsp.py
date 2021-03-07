import constants

from mlrose_hiive.generators import TSPGenerator
from mlrose_hiive.runners import RHCRunner, SARunner, GARunner, MIMICRunner
from mlrose_hiive.algorithms import ExpDecay

problem_name = "tsp"
algorithm_names = ["RHC", "SA", "GA", "MIMIC"]
problem_sizes = [10, 30, 50]

for problem_size in problem_sizes:
    for algorithm_name in algorithm_names:

        experiment_name = '{0}-{1}-{2}'.format(problem_name, algorithm_name, problem_size)
        problem = TSPGenerator.generate(
            seed=0,
            number_of_cities=problem_size,
            area_width=500,
            area_height=500
        )
        if algorithm_name == "RHC" and True:
            rhc = RHCRunner(problem=problem,
                            experiment_name=experiment_name,
                            output_directory=constants.OUTPUT_DIRECTORY,
                            seed=0,
                            iteration_list=constants.iteration_list,
                            max_attempts=100,
                            restart_list=[0]
                            )
            df_run_stats, df_run_curves = rhc.run()
        if algorithm_name == "SA" and True:
            sa = SARunner(problem=problem,
                          experiment_name=experiment_name,
                          output_directory=constants.OUTPUT_DIRECTORY,
                          seed=0,
                          iteration_list=constants.iteration_list,
                          max_attempts=100,
                          temperature_list=[50],
                          decay_list=[ExpDecay])
            sa_run_stats, sa_run_curves = sa.run()
        if algorithm_name == "GA":
            ga = GARunner(problem=problem,
                          experiment_name=experiment_name,
                          output_directory=constants.OUTPUT_DIRECTORY,
                          seed=0,
                          iteration_list=constants.iteration_list,
                          max_attempts=100,
                          population_sizes=[50],
                          mutation_rates=[0.20])
            ga_run_stats, ga_run_curves = ga.run()
        if algorithm_name == "MIMIC" and True:
            mimic = MIMICRunner(problem=problem,
                                experiment_name=experiment_name,
                                output_directory=constants.OUTPUT_DIRECTORY,
                                seed=0,
                                iteration_list=[1,5,10,20,30,50],
                                population_sizes=[200],
                                max_attempts=100,
                                keep_percent_list=[0.10],
                                use_fast_mimic=True)
            mimic_run_stats, mimic_run_curves = mimic.run()
