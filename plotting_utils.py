import constants

import pandas as pd
import matplotlib.pyplot as plt

problem_name = "kcolors"
type = "curves"

# problem_sizes = [10, 50]
problem_size = 10

plt.xlabel("Iterations")
plt.ylabel("Fitness")
plt.title('{0}:{1}'.format(problem_name, problem_size))
plt.grid()

d = {}
for algorithm_name in constants.algorithm_names:
    dir = 'output/{0}-{1}-{2}'.format(problem_name, algorithm_name, problem_size)
    filename = '{3}__{0}-{3}-{1}__{2}_df.csv'.format(problem_name, problem_size, type, algorithm_name)
    path = '{0}/{1}'.format(dir, filename)

    df = pd.read_csv(path)
    d[algorithm_name] = {}
    d[algorithm_name]["X"] = df["Iteration"]
    d[algorithm_name]["Y"] = (df["Fitness"] * -1) + df["Fitness"].max()
    plt.plot(d[algorithm_name]["X"], d[algorithm_name]["Y"], label=algorithm_name)

plt.legend()
plt.show()
