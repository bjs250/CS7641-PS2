import constants

import pandas as pd
from mlrose_hiive.generators import ContinuousPeaksGenerator
from mlrose_hiive.runners import RHCRunner, SARunner, GARunner, MIMICRunner
from mlrose_hiive.algorithms import ExpDecay
import matplotlib.pyplot as plt

problem_name = "continuouspeaks"
algorithm_name = "MIMIC"
problem_size = 50
t = 0.25

experiment_name = '{0}-{1}-{2}-{3}-experiment'.format(problem_name, algorithm_name, problem_size, t)
problem = ContinuousPeaksGenerator.generate(
    seed=0,
    size=problem_size,
    t_pct=t
)

if algorithm_name == "RHC" and True:
    restart_list = [5, 10, 15, 100, 1000]
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
    temperature_list=[5, 10, 25, 50, 100]
    sa = SARunner(problem=problem,
                  experiment_name=experiment_name,
                  output_directory=constants.OUTPUT_DIRECTORY,
                  seed=0,
                  iteration_list=constants.iteration_list,
                  max_attempts=100,
                  temperature_list=temperature_list,
                  decay_list=[ExpDecay])
    sa_run_stats, sa_run_curves = sa.run()

if algorithm_name == "GA" and False:
    population_sizes = [100]
    mutation_rates = [0.05, 0.2, 0.3, 0.4]
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
    population_sizes = [100]
    keep_percent_list = [0.05, 0.1, 0.2, 0.3, 0.4]
    mimic = MIMICRunner(problem=problem,
                        experiment_name=experiment_name,
                        output_directory=constants.OUTPUT_DIRECTORY,
                        seed=0,
                        iteration_list=[1,2,3,4,5,10,15,20,25,30],
                        population_sizes=population_sizes,
                        max_attempts=100,
                        keep_percent_list=keep_percent_list,
                        use_fast_mimic=True)
    mimic_run_stats, mimic_run_curves = mimic.run()

type = "curves"
dir = 'output/{0}-{1}-{2}-{3}-experiment'.format(problem_name, algorithm_name, problem_size, t)
filename = '{3}__{0}-{3}-{1}-{4}-experiment__{2}_df.csv'.format(problem_name, problem_size, type, algorithm_name, t)
path = '{0}/{1}'.format(dir, filename)
df = pd.read_csv(path)

max_row = df.iloc[df["Fitness"].idxmax()]
print(max_row)

type = "run_stats"
dir = 'output/{0}-{1}-{2}-{3}-experiment'.format(problem_name, algorithm_name, problem_size, t)
filename = '{3}__{0}-{3}-{1}-{4}-experiment__{2}_df.csv'.format(problem_name, problem_size, type, algorithm_name, t)
path = '{0}/{1}'.format(dir, filename)
df2 = pd.read_csv(path)

if algorithm_name == "RHC":
    optimal_restarts = max_row["Restarts"]
    df2 = df2[df2["Restarts"].isin([optimal_restarts])]

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
# Y = (df2["Fitness"] * -1) + df2["Fitness"].max()
plt.plot(X, Y)
plt.show()
