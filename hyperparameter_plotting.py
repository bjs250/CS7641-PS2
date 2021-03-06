import pandas as pd
import matplotlib.pyplot as plt

problem_name = "kcolors"
algorithm_name = "GA"
problem_size = 50

type = "run_stats"
dir = 'output/{0}-{1}-{2}-experiment'.format(problem_name, algorithm_name, problem_size)
filename = '{3}__{0}-{3}-{1}-experiment__{2}_df.csv'.format(problem_name, problem_size, type, algorithm_name)
path = '{0}/{1}'.format(dir, filename)
df = pd.read_csv(path)

if algorithm_name == "GA" and True:
    d = {}
    mutation_rates = [0.05, 0.1, 0.2, 0.3, 0.4]
    colors = ["b-", "g-", "c-", "m-", "r-"]
    for index, param in enumerate(mutation_rates):
        df2 = df[df["Population Size"].isin([10])]
        df2 = df2[df2["Mutation Rate"].isin([param])]
        print(df2)

        d[param] = {}
        d[param]["X"] = df2["Iteration"]
        d[param]["Y"] = (df2["Fitness"] * -1) + df2["Fitness"].max()
        plt.plot(d[param]["X"][0:5], d[param]["Y"][0:5], colors[index], label="{0}:{1}".format("Mutation Rate", param))


if algorithm_name == "SA" and False:
    d = {}
    temperature_list = [200, 250, 500, 1000]
    colors = ["b-", "g-", "m-", "r-"]
    for index, temperature in enumerate(temperature_list):

        df2 = df[df["Temperature"].isin([temperature])]

        d[temperature] = {}
        d[temperature]["X"] = df2["Iteration"]
        d[temperature]["Y"] = (df2["Fitness"] * -1) + df2["Fitness"].max()
        plt.plot(d[temperature]["X"], d[temperature]["Y"], colors[index], label="Temperature:{0}".format(temperature))

plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.title('{0}:{1} Tuning'.format(problem_name, algorithm_name))
plt.grid()
plt.legend()
plt.show()