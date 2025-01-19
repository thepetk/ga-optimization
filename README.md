# JSSP Optimization Examples

This repo provides a couple of solutions for the Job-Shop Scheduling Problem. More specifically, the two solutions available are:

- The genetic algorithm optimization, based on the `GAJobScheduler` class.
- The ants colony optimization, based on the `AntJobScheduler` class.

## Background

As wikipedia [describes](https://en.wikipedia.org/wiki/Job-shop_scheduling) _"the job-shop problem (JSP) or job-shop scheduling problem (JSSP) is an optimization problem in computer science and operations research. It is a variant of optimal job scheduling"_. In other words given a dataset of jobs and machines we need to find the optimal combination of task jobs inside machines that will help us complete all jobs, with the number of machines avaiable, in the least time possible.

The current approach tries to represent all different items involved into objects. Therefore, we have classes like `Job`, `Task` and `Machine` to help us capture the solution to the problem easier. To help us find the optimal solution we have selected to approaches:

- [Genetic Algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm): So we have translated the JSP into a chromosome evolution problem where the elements of a chromosome are the different tasks of all jobs and our goal is to find the best combination of parent chromosomes that will help us evolve to the best solution for the problem, or else the least execution time possible.
- [Ants Colony Optimization Algorithms](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms): So we have translated the JSP into an optimal ant's route to find food, and each time the next ant bases its route on the pheromone traces that all previous ants left.

## Dataset & Acknowledgements

The application uses a set of predefined datasets inside a given `.yaml` file. All the datasets used as examples are taken from a library which can be found [here](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt).

## Installation

The project is based on `python 3.10` and [uv](https://github.com/astral-sh/uv). In order to install all dependencies:

```bash
uv venv --python 3.10
```

## Run

In order to run the script simply run:

```bash
uv run main.py
```

## Configurations

The structure of the project allows multiple configurations with the usage of environment variables. For example one can:

- Change the dataset name by:

```bash
DATASET_NAME="abz6" uv run main.py
```

- Choose optimization algorithm:

```bash
SCHEDULER_TYPE="ants" uv run main.py
```

### General Configurations

| Name                         | Description                                                                      | Type     | Default     |
| ---------------------------- | -------------------------------------------------------------------------------- | -------- | ----------- |
| `DATA_PATH`                  | The path to the dataset yaml file.                                               | `string` | "data.yaml" |
| `DATASET_NAME`               | The name of the dataset we will use during the execution.                        | `string` | "thpe"      |
| `NUM_GENERATIONS`            | The number of iterations for the problem.                                        | `int`    | 100         |
| `SCHEDULER_TYPE`             | The optimization algorithm used to optimize the solution. One of ["ants", "ga"]. | `string` | "ga"        |
| `VERBOSE`                    | If `True` a more verbose output is provided.                                     | `bool`   | `False`     |
| `STATIONARY_STATE_THRESHOLD` | The maximum number of continuous iterations sharing the same best_fitness.       | `int`    | 50          |

### Genetic Algorithm Configurations

| Name               | Description                                                                                      | Type     | Default      |
| ------------------ | ------------------------------------------------------------------------------------------------ | -------- | ------------ |
| `CROSSOVER_METHOD` | The method used for crossover. One of ["one_point", "two_points"].                               | `string` | "one_point"  |
| `MUTATION_MODE`    | The mode of mutation. One of ["flip", "shift"].                                                  | `string` | "flip"       |
| `MUTATE_THRESHOLD` | The percentage of mutations applied in a sample.                                                 | `float`  | 0.5          |
| `POPULATION_LIMIT` | The total number of chromosomes in a population.                                                 | `int`    | 1000         |
| `SAMPLE_LEN`       | The length of each population used each time to select parents in the tournament selection mode. | `int`    | 100          |
| `SELECTION_MODE`   | The parent selection method. One of ["tournament", "stochastic"].                                | `string` | "tournament" |

### Ant Colony Algorithm Configurations

| Name                       | Description                                                       | Type    | Default |
| -------------------------- | ----------------------------------------------------------------- | ------- | ------- |
| `ANTS_NUMBER`              | The total number ants included in the algorithm.                  | `int`   | 10      |
| `ANTS_PHEROMONE_INFLUENCE` | The influence rate of the pheromone on the decisions of each ant. | `float` | 0.5     |
| `ANTS_EVAPORATION_RATE`    | The declining of pheromone over the number of iterations.         | `float` | 0.5     |
| `ANTS_PHEROMONE_DEPOSIT`   | The amount of pheromone left by every ant each time.              | `int`   | 1       |
