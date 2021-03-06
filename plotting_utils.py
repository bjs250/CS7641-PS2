import constants

import pandas as pd
import matplotlib.pyplot as plt

problem_name = "kcolors"
type = "curves"

problem_size = 50


if False:
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
        d[algorithm_name]["Y"] = df["Fitness"]
        plt.plot(d[algorithm_name]["X"], d[algorithm_name]["Y"], label=algorithm_name)

    plt.xlim(0, 300)
    plt.legend()
    plt.show()

problem_name = "kcolors"
type = "curves"
problem_sizes = [10, 25, 50, 75, 100]
colors = ["bo", "yx", "g.", "r."]

if True:
    plt.xlabel("Problem Size")
    plt.ylabel("Fitness")
    plt.title('{0} Complexity Curve'.format(problem_name))
    plt.grid()

    d = {}
    for index, algorithm_name in enumerate(constants.algorithm_names):
        d[algorithm_name] = []
        for problem_size in problem_sizes:
            dir = 'output/{0}-{1}-{2}'.format(problem_name, algorithm_name, problem_size)
            filename = '{3}__{0}-{3}-{1}__{2}_df.csv'.format(problem_name, problem_size, type, algorithm_name)
            path = '{0}/{1}'.format(dir, filename)

            df = pd.read_csv(path)
            d[algorithm_name].append(df["Fitness"].min())

    for index, algorithm_name in enumerate(constants.algorithm_names):
        plt.plot(problem_sizes, d[algorithm_name], colors[index], label=algorithm_name)

    plt.xlim(5, 105)
    plt.legend()
    plt.show()
