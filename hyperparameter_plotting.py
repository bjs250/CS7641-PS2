import pandas as pd
import matplotlib.pyplot as plt

problem_name = "tsp"
algorithm_name = "MIMIC"
type = "run_stats"

if problem_name == "kcolors":
    problem_size = 50
    dir = 'output/{0}-{1}-{2}-experiment'.format(problem_name, algorithm_name, problem_size)
    filename = '{3}__{0}-{3}-{1}-experiment__{2}_df.csv'.format(problem_name, problem_size, type, algorithm_name)
    path = '{0}/{1}'.format(dir, filename)
    df = pd.read_csv(path)

if problem_name == "continuouspeaks":
    problem_size = 50
    t = 0.25
    dir = 'output/{0}-{1}-{2}-{3}-experiment'.format(problem_name, algorithm_name, problem_size, t)
    filename = '{3}__{0}-{3}-{1}-{4}-experiment__{2}_df.csv'.format(problem_name, problem_size, type, algorithm_name, t)
    path = '{0}/{1}'.format(dir, filename)
    df = pd.read_csv(path)

if problem_name == "tsp":
    problem_size = 30
    dir = 'output/{0}-{1}-{2}-experiment'.format(problem_name, algorithm_name, problem_size)
    filename = '{3}__{0}-{3}-{1}-experiment__{2}_df.csv'.format(problem_name, problem_size, type, algorithm_name)
    path = '{0}/{1}'.format(dir, filename)
    df = pd.read_csv(path)

if algorithm_name == "SA" and True:
    d = {}
    temperature_list = [10, 50, 100, 200, 300, 500, 1000]
    colors = ["b-", "k-", "c-", "g-", "y", "m-", "r-"]
    for index, temperature in enumerate(temperature_list):

        df2 = df[df["Temperature"].isin([temperature])]

        d[temperature] = {}
        d[temperature]["X"] = df2["Iteration"]
        d[temperature]["Y"] = df2["Fitness"] * -1
        plt.plot(d[temperature]["X"], d[temperature]["Y"], colors[index], label="Temperature:{0}".format(temperature))

if algorithm_name == "GA" and True:
    d = {}
    pop_sizes = [10, 50, 100, 200]
    mutation_rates = [0.05, 0.2, 0.3, 0.4]
    colors = ["b-", "g-", "c-", "m-", "r-"]
    for index, param in enumerate(pop_sizes):
        df2 = df[df["Population Size"].isin([param])]
        df2 = df2[df2["Mutation Rate"].isin([0.2])]
        print(df2)

        d[param] = {}
        d[param]["X"] = df2["Iteration"]
        d[param]["Y"] = df2["Fitness"] * -1
        plt.plot(d[param]["X"][0:7], d[param]["Y"][0:7], colors[index], label="{0}:{1}".format("Pop Size", param))


if algorithm_name == "MIMIC" and True:
    d = {}
    population_sizes = [10, 50, 100, 200]
    keep_percent_list = [0.05, 0.1, 0.2]
    colors = ["b.", "go", "c-", "m-"]
    for index, param in enumerate(population_sizes):
        df2 = df[df["Population Size"].isin([param])]
        df2 = df2[df2["Keep Percent"].isin([0.1])]

        d[param] = {}
        d[param]["X"] = df2["Iteration"]
        d[param]["Y"] = df2["Fitness"] * -1
        plt.plot(d[param]["X"], d[param]["Y"], colors[index], label="{0}:{1}".format("Population Size", param))

plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.title('{0}:{1} Tuning'.format(problem_name, algorithm_name))
plt.grid()
plt.legend(loc="lower right")
plt.show()