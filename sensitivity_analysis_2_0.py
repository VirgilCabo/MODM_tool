import numpy as np
import pandas as pd
import itertools
import random
import matplotlib.pyplot as plt

def get_user_uncertainties(initial_weights):
    uncertainties = {}
    for criterion, weight in initial_weights.items():
        print(f"Initial weight for {criterion}: {weight}")
        lower_bound = float(input(f"Enter the lower bound of your confidence interval for {criterion}: "))
        upper_bound = float(input(f"Enter the upper bound of your confidence interval for {criterion}: "))
        
        # Assuming a 95% confidence interval, we can derive the standard deviation.
        std_dev = (upper_bound - lower_bound) / (2 * 1.96)
        uncertainties[criterion] = std_dev
    return uncertainties


def generate_random_weights_for_criterion(initial_weight, std_dev, num_samples=100):
    weights = []
    for _ in range(num_samples):
        weight = -1  # Initialize with a negative value to enter the while loop
        while weight < 0:  # Keep sampling until we get a non-negative weight
            weight = np.random.normal(initial_weight, std_dev)
        weights.append(weight)
    return weights


def generate_all_random_weights(initial_weights, uncertainties, num_samples=100):
    all_random_weights = {}
    for criterion, weight in initial_weights.items():
        all_random_weights[criterion] = generate_random_weights_for_criterion(weight, uncertainties[criterion], num_samples)
    return all_random_weights


def sample_combinations(all_random_weights, num_samples=1000):
    # Get the Cartesian product of all lists of random weights
    all_combinations = list(itertools.product(*all_random_weights.values()))
    
    # Randomly sample from the list of all combinations
    sampled_combinations = random.sample(all_combinations, num_samples)
    
    # Convert the sampled combinations to a list of dictionaries
    sampled_dicts = []
    for combination in sampled_combinations:
        weight_dict = dict(zip(all_random_weights.keys(), combination))
        sampled_dicts.append(weight_dict)
    
    return sampled_dicts


def efficient_sample_combinations(all_random_weights, num_samples=1000):
    sampled_dicts = []
    
    # List of criteria
    criteria = list(all_random_weights.keys())
    
    for _ in range(num_samples):
        # Randomly select one weight for each criterion
        sampled_weights = [random.choice(all_random_weights[criterion]) for criterion in criteria]
        
        # Combine the selected weights into a dictionary
        weight_dict = dict(zip(criteria, sampled_weights))
        sampled_dicts.append(weight_dict)
    
    return sampled_dicts


def plot_histogram(weights, criterion_name):
    plt.hist(weights, bins=30, density=True, alpha=0.75)
    plt.title(f'Distribution of Weights for {criterion_name}')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


initial_weights = {
    'C1' : 1,
    'C2' : 2,
    'C3' : 3

}
uncertainties = get_user_uncertainties(initial_weights)
all_random_weights = generate_all_random_weights(initial_weights, uncertainties, num_samples=10000)
print(all_random_weights)
sampled_dicts = efficient_sample_combinations(all_random_weights, num_samples=10000)
print(sampled_dicts)
plot_histogram(all_random_weights['C1'], 'C1')






