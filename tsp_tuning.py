import constants

import pandas as pd
from mlrose_hiive.generators import TSPGenerator
from mlrose_hiive.runners import RHCRunner, SARunner, GARunner, MIMICRunner
from mlrose_hiive.algorithms import ExpDecay
import matplotlib.pyplot as plt

problem_name = "TSP"
algorithm_name = "GA"
problem_size = 30

experiment_name = '{0}-{1}-{2}-experiment'.format(problem_name, algorithm_name, problem_size)

problem = TSPGenerator.generate(
    seed=0,
    number_of_cities=problem_size,
    area_width=500,
    area_height=500
)

if algorithm_name == "RHC" and False:
    restart_list = [5, 10, 15, 100]
    rhc = RHCRunner(problem=problem,
                    experiment_name=experiment_name,
                    output_directory=constants.OUTPUT_DIRECTORY,
                    seed=0,
                    iteration_list=constants.iteration_list,
                    max_attempts=100,
                    restart_list=restart_list
                    )
    rhc_run_stats, rhc_run_curves = rhc.run()

if algorithm_name == "SA" and False:
    temperature_list=[10, 50, 100, 200, 300, 500, 1000]
    sa = SARunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=constants.OUTPUT_DIRECTORY,
                  seed=0,
                  iteration_list=constants.iteration_list,
                  max_attempts=100,
                  temperature_list=temperature_list,
                  decay_list=[ExpDecay])
    sa_run_stats, sa_run_curves = sa.run()

if algorithm_name == "GA" and True:
    population_sizes = [200]
    mutation_rates = [0.05, 0.1, 0.2, 0.3, 0.4]
    ga = GARunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=constants.OUTPUT_DIRECTORY,
                  seed=0,
                  iteration_list=constants.iteration_list,
                  max_attempts=100,
                  population_sizes=population_sizes,
                  mutation_rates=mutation_rates)
    ga_run_stats, ga_run_curves = ga.run()

if algorithm_name == "MIMIC" and False:
    population_sizes = [10, 50, 100, 200]
    keep_percent_list = [0.1]
    mimic = MIMICRunner(problem=problem,
                        experiment_name=experiment_name,
                        output_directory=constants.OUTPUT_DIRECTORY,
                        seed=0,
                        iteration_list=[1,5,10,20,30,50],
                        population_sizes=population_sizes,
                        max_attempts=100,
                        keep_percent_list=keep_percent_list,
                        use_fast_mimic=True)
    mimic_run_stats, mimic_run_curves = mimic.run()

type = "curves"
dir = 'output/{0}-{1}-{2}-experiment'.format(problem_name, algorithm_name, problem_size)
filename = '{3}__{0}-{3}-{1}-experiment__{2}_df.csv'.format(problem_name, problem_size, type, algorithm_name)
path = '{0}/{1}'.format(dir, filename)
df = pd.read_csv(path)

max_row = df.iloc[df["Fitness"].idxmin()]
print(max_row)

type = "run_stats"
dir = 'output/{0}-{1}-{2}-experiment'.format(problem_name, algorithm_name, problem_size)
filename = '{3}__{0}-{3}-{1}-experiment__{2}_df.csv'.format(problem_name, problem_size, type, algorithm_name)
path = '{0}/{1}'.format(dir, filename)
df2 = pd.read_csv(path)

if algorithm_name == "RHC":
    optimal_restarts = max_row["Restarts"]
    df2 = df2[df2["Restarts"].isin([optimal_restarts])].groupby(["Iteration"]).mean().reset_index()

if algorithm_name == "SA":
    optimal_temperature = max_row["Temperature"]
    df2 = df2[df2["Temperature"].isin([optimal_temperature])]

if algorithm_name == "GA":
    optimal_population_size = max_row["Population Size"]
    optimal_keep_percent = max_row["Mutation Rate"]
    df2 = df2[df2["Population Size"].isin([optimal_population_size])]
    df2 = df2[df2["Mutation Rate"].isin([optimal_keep_percent])]

if algorithm_name == "MIMIC":
    optimal_population_size = max_row["Population Size"]
    optimal_keep_percent = max_row["Keep Percent"]
    df2 = df2[df2["Population Size"].isin([optimal_population_size])]
    df2 = df2[df2["Keep Percent"].isin([optimal_keep_percent])]

X = df2["Iteration"]
Y = df2["Fitness"]
plt.plot(X, Y)
plt.show()
