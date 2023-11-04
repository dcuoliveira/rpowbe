import os
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model_name', type=str, help='model name to be used for saving the model')
args = parser.parse_args()
model_name = args.model_name

# seeds = range(1, 50000)

long_only_options = ["True", "False"]

estimation_methods = ["mle", "cbb", "nobb", "sb"]

alphas = [0.95, 0.75, 0.05]

with open(f"run_experiment_{model_name}.sh", "w") as file:
        for em in estimation_methods:
            for lo in long_only_options:
                # seed = np.random.choice(seeds)

                if (model_name == "rbmvog") or (model_name == "rbmvo"):
                    for alpha in alphas:
                        command = f"python run_{model_name}.py --model_name {model_name} --long_only {lo} --mean_cov_estimator {em} --alpha {alpha} &\n"

                        file.write(command)
                else:
                    command = f"python run_{model_name}.py --model_name {model_name} --long_only {lo} --mean_estimator {em} --cov_estimator {em} &\n"
                    file.write(command)