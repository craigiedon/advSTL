import math
import pandas as pd
from hypermapper import optimizer
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os

matplotlib.use("GTK3Agg")
print(plt.get_backend())



def branin_function_1d(X):
    # The function must receive a dictionary
    x1 = X['x1']

    # Branin function computation
    a = 1.0
    b = 5.1 / (4.0 * math.pi * math.pi)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    x2 = 2.275
    value = a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s

    # The functio must return the objective value (a number)
    return value


plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 18
point_size = matplotlib.rcParams['lines.markersize'] ** 2.8
point_size_optimum = matplotlib.rcParams['lines.markersize'] ** 2

optimum = math.pi
value_at_optimum = branin_function_1d({'x1': optimum})

# Sample 1000 (x,y) pairs from the function to plot its curve
branin_line_xs = np.linspace(-5, 10, 1000)
branin_line_ys = []
for x in branin_line_xs:
    y = branin_function_1d({'x1': x})
    branin_line_ys.append(y)

plt.plot(branin_line_xs, branin_line_ys, label='1D Branin Function')

# Mark the known optimum on the curve
plt.scatter(optimum, value_at_optimum, s=point_size_optimum, marker='o', color="black", label="Minimum")
plt.legend()
plt.xlabel("x1")
plt.ylabel("value")
plt.show()
print("The 1d Branin function has one global optimum at x1 = \u03C0", flush=True)
print("(x, y) at minimum is: (" + str(optimum) + "," + str(value_at_optimum) + ")", flush=True)

scenario = {}
scenario["application_name"] = "1d_branin"
scenario["optimization_objectives"] = ["value"]

number_of_RS = 3
scenario["design_of_experiment"] = {}
scenario["design_of_experiment"]["number_of_samples"] = number_of_RS

scenario["optimization_iterations"] = 10

scenario["models"] = {}
scenario["models"]["model"] = "gaussian_process"

scenario["input_parameters"] = {}
x1 = {}
x1["parameter_type"] = "real"
x1["values"] = [-5, 10]

scenario["input_parameters"]["x1"] = x1

with open("example_1d_branin_scenario.json", "w") as scenario_file:
    json.dump(scenario, scenario_file, indent=4)

f = open("example_1d_branin_scenario.json", "r")
text = f.read()
print(text, flush=True)
f.close()

# stdout = sys.stdout
#
# optimizer.optimize("example_1d_branin_scenario.json", branin_function_1d)
# sys.stdout = stdout

cmap = plt.get_cmap('winter')
plt.plot(branin_line_xs, branin_line_ys, label="1D Branin Function")

# Load the points evaluated by HyperMapper during optimization
optimum = math.pi
sampled_points = pd.read_csv("1d_branin_output_samples.csv", usecols=['x1', 'value'])
x_points = sampled_points['x1'].values
y_points = sampled_points['value'].values

# Split between DoE and BO
doe_x = x_points[:number_of_RS]
doe_y = y_points[:number_of_RS]
bo_x = x_points[number_of_RS:]
bo_y = y_points[number_of_RS:]
bo_iterations = list(range(len(bo_x)))

plt.scatter(doe_x, doe_y, s=point_size, marker='x', color="red", label="Initial Random Sampling")
plt.scatter(optimum, value_at_optimum, s=point_size_optimum, marker='o', color="black", label="Minimum")
plt.scatter(bo_x, bo_y, s=point_size, marker='x', c=bo_iterations, cmap=cmap, label="Bayesian Otimization")

plt.legend()
plt.xlabel("x1")
plt.ylabel("value")
plt.show()

scenario = {}
scenario["application_name"] = "1d_branin"
scenario["optimization_objectives"] = ["value"]

number_of_RS = 2
scenario["design_of_experiment"] = {}
scenario["design_of_experiment"]["number_of_samples"] = number_of_RS
scenario["optimization_iterations"] = 8

scenario["models"] = {}
scenario["models"]["model"] = "gaussian_process"

scenario["input_parameters"] = {}
x1 = {}
x1["parameter_type"] = "ordinal"
x1["values"] = [-5.0, -4.5, -4.0, -3.5, -3.0, -1.5, -1.0, -0.5, 0.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 7.0, 7.5,
                8.0, 8.5, 9.0, 10.0]

scenario["input_parameters"]["x1"] = x1

with open("example_ordinal_1d_branin_scenario.json", "w") as scenario_file:
    json.dump(scenario, scenario_file, indent=4)


f = open("example_ordinal_1d_branin_scenario.json", "r")
text = f.read()
print(text, flush=True)
f.close()

optimizer.optimize("example_ordinal_1d_branin_scenario.json", branin_function_1d)
sys.stdout = stdout



cmap = plt.get_cmap('winter')
point_size_ordinal = matplotlib.rcParams['lines.markersize']**1.2


branin_scatter_xs = x1["values"]
branin_scatter_ys = []
for x in branin_scatter_xs:
    y = branin_function_1d({'x1': x})
    branin_scatter_ys.append(y)
plt.scatter(branin_scatter_xs, branin_scatter_ys, s=point_size_ordinal, marker='o', color="black", label="1D Ordinal Branin Function")

# Load the points evaluated by HyperMapper during optimization
sampled_points = pd.read_csv("1d_branin_output_samples.csv", usecols=['x1', 'value'])
x_points = sampled_points['x1'].values
y_points = sampled_points['value'].values

# Split between DoE and BO
doe_x = x_points[:number_of_RS]
doe_y = y_points[:number_of_RS]
bo_x = x_points[number_of_RS:]
bo_y = y_points[number_of_RS:]
bo_iterations = list(range(len(bo_x)))
optimum = 3

plt.scatter(doe_x, doe_y, s=point_size, marker='x', color="red", label="Initial Random Sampling")
plt.scatter(optimum, value_at_optimum, s=point_size_optimum, marker='o', color="black", label="Minimum")
plt.scatter(bo_x, bo_y, s=point_size, marker='x', c=bo_iterations, cmap=cmap, label="Bayesian Otimization")

plt.legend()
plt.xlabel("x1")
plt.ylabel("value")
plt.show()

