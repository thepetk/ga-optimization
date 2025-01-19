import copy
from dataclasses import dataclass
import os
import random

from matplotlib import pyplot as plt
import numpy as np
import yaml

# Optimization Parameters
CROSSOVER_METHOD = os.getenv("CROSSOVER_METHOD", "one_point")
DATA_PATH = os.getenv("DATA_YAML", "data.yaml")
DATASET_NAME = os.getenv("DATASET_NAME", "thpe")
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", 100))
MUTATE_THRESHOLD = float(os.getenv("SAMPLE_LEN", 0.5))
MUTATION_MODE = os.getenv("MUTATION_MODE", "flip")
POPULATION_LIMIT = int(os.getenv("POPOPULATION_LIMITPU", 1000))
RAW_TASK = list[tuple[int, int]]
SAMPLE_LEN = int(os.getenv("SAMPLE_LEN", 100))
SCHEDULER_TYPE = os.getenv("SCHEDULER_TYPE", "ga")
SELECTION_MODE = os.getenv("SELECTION_MODE", "tournament")
STATIONARY_STATE_THRESHOLD = int(os.getenv("STATIONARY_STATE_THRESHOLD", 50))
VERBOSE = bool(os.getenv("VERBOSE", 0))


# Error classes
class ParentSelectionNotFoundError(Exception):
    pass


class CrossoverMethodNotFoundError(Exception):
    pass


class MutationModeNotFoundError(Exception):
    pass


class SchedulerTypeNotFoundError(Exception):
    pass


# Choice Classes
class ParentSelection:
    TOURNAMENT = 1
    STOCHASTIC = 2


class CrossoverMethod:
    ONE_POINT = 1
    TWO_POINTS = 2


class MutationMode:
    FLIP = 1
    SHIFT = 2


class SchedulerType:
    GA = 1
    ANTS_COLONY = 2


# Choice Methods
def get_crossover_method() -> "int":
    if CROSSOVER_METHOD == "one_point":
        return CrossoverMethod.ONE_POINT
    elif CROSSOVER_METHOD == "two_points":
        return CrossoverMethod.TWO_POINTS
    else:
        raise CrossoverMethodNotFoundError(
            f"Crossover method {CROSSOVER_METHOD} is not supported"
        )


def get_selection_mode() -> "int":
    if SELECTION_MODE == "tournament":
        return ParentSelection.TOURNAMENT
    elif SELECTION_MODE == "stochastic":
        return ParentSelection.STOCHASTIC
    else:
        raise ParentSelectionNotFoundError(
            f"Parent selection mode {SELECTION_MODE} is not supported"
        )


def get_mutation_mode() -> "int":
    if MUTATION_MODE == "flip":
        return MutationMode.FLIP
    elif MUTATION_MODE == "shift":
        return MutationMode.SHIFT
    else:
        raise MutationModeNotFoundError(
            f"Mutation mode {MUTATION_MODE} is not supported"
        )


def get_scheduler_type() -> "int":
    if SCHEDULER_TYPE == "ga":
        return SchedulerType.GA
    elif SCHEDULER_TYPE == "ants":
        return SchedulerType.ANTS_COLONY
    else:
        raise SchedulerTypeNotFoundError(
            f"Scheduler type {SCHEDULER_TYPE} is not supported"
        )


# Refine choices
SET_PARENT_SELECTION_MODE = get_selection_mode()
SET_CROSSOVER_METHOD = get_crossover_method()
SET_MUTATION_MODE = get_mutation_mode()
SET_SCHEDULER_TYPE = get_scheduler_type()


# Second Optimization algorithm parameters
ANTS_NUMBER = int(os.getenv("ANTS_NUMBER", 10))
ANTS_PHEROMONE_INFLUENCE = float(os.getenv("ANTS_PHEROMONE_INFLUENCE", 0.5))
ANTS_HEURISTIC_INFLUENCE = float(os.getenv("ANTS_HEURISTIC_INFLUENCE", 0.8))
ANTS_EVAPORATION_RATE = float(os.getenv("ANTS_EVAPORATION_RATE", 0.5))
ANTS_PHEROMONE_DEPOSIT = int(os.getenv("ANTS_PHEROMONE_DEPOSIT", 1))


@dataclass
class Chromosome:
    data: "list[int]"


@dataclass
class Task:
    task_id: "int"
    runtime: "int"
    machine_id: "int"


class Job:
    """
    The job class is the representation of a job in the job shop problem
    """

    def __init__(self, job_id: "int", raw_task_list: "list[RAW_TASK]") -> "None":
        self.job_id = job_id
        self.tasks = self._init_tasks(raw_task_list)
        self.tasks_iterator = iter(self.tasks)
        self.task_end_time: "int" = 0

    def _init_tasks(self, raw_task_list: "list[RAW_TASK]") -> "list[Task]":
        return [
            Task(task_id=task_id, runtime=runtime, machine_id=machine_id)
            for task_id, (machine_id, runtime) in enumerate(raw_task_list)
        ]

    def reset_tasks(self) -> "None":
        """ "
        resets all tasks
        """
        self.tasks_iterator = iter(self.tasks)
        self.task_end_time = 0

    def get_next_task(self) -> "Task":
        """
        get next task from the tasks_iterator
        """
        return next(self.tasks_iterator)


@dataclass
class Machine:
    machine_id: "int"
    end_time: "int" = 0

    def reset(self) -> "None":
        """
        reset the end time of the machine
        """
        self.end_time = 0


class BaseJobScheduler:
    def _generate_jobs(self, data: "list[list[tuple[int, int]]]") -> "list[Job]":
        """
        create a list of jobs for a given raw data.
        """
        return [Job(job_id=idx, raw_task_list=row) for idx, row in enumerate(data)]

    def _generate_machines(self, num_machines: "int") -> "list[Machine]":
        """
        create a list of Machine objects for a given number of machines
        """
        return [Machine(machine_id=idx) for idx in range(num_machines)]


class AntJobScheduler(BaseJobScheduler):
    def __init__(self, data: "list[RAW_TASK]", num_machines: "int") -> "None":
        self.jobs = self._generate_jobs(data)
        self.machines = self._generate_machines(num_machines)
        self.n_ants = ANTS_NUMBER
        self.n_iterations = NUM_GENERATIONS
        self.pheromone_influence = 0.8  # ANTS_PHEROMONE_INFLUENCE
        self.heuristic_influence = 2.5  # ANTS_HEURISTIC_INFLUENCE
        self.evaporation_rate = 0.8  # ANTS_EVAPORATION_RATE
        self.pheromone_deposit = ANTS_PHEROMONE_DEPOSIT
        self.pheromone = np.ones(
            (len(self.jobs), len(self.jobs[0].tasks), len(self.machines))
        )
        self.pheromone = np.random.uniform(0.5, 1.5, size=self.pheromone.shape)

    def run(self) -> "None":
        best_fitness = None
        best_fitnesses: "list[int]" = []

        for iteration in range(self.n_iterations):
            routes = []
            fitnesses = []

            for ant in range(self.n_ants):
                route, fitness = self._calculate_route()
                routes.append(route)
                fitnesses.append(fitness)

                if best_fitness is None or fitness < best_fitness:
                    best_fitness = fitness
                    best_ant = ant

                if VERBOSE:
                    print(
                        "[{}/{}] Local fitness {} by ant {}".format(
                            iteration, self.n_iterations, fitness, ant
                        )
                    )
            best_fitnesses.append(best_fitness)
            self._update_pheromone(routes, fitnesses)
            print(
                "[{}/{}] Iteration best fitness {} by ant {}".format(
                    iteration, self.n_iterations, best_fitness, best_ant
                )
            )

        return best_fitnesses

    def _get_probability(
        self,
        machine_schedule: "list[int]",
        jobs_schedule: "list[int]",
        m_id: "int",
        j_id: "int",
        runtime: "int",
    ) -> "float":
        pheromone = (
            self.pheromone[j_id][jobs_schedule[j_id]][m_id] ** self.pheromone_influence
        )
        heuristic = (1 / (machine_schedule[m_id] + runtime)) ** self.heuristic_influence
        return pheromone * heuristic

    def _get_random_job(self, idx_probs: "list[tuple[int, float]]") -> "int":
        corrected_probs = [
            (float(p[1]) / (sum([p[1] for p in idx_probs]))) for p in idx_probs
        ]
        indices = [p[0] for p in idx_probs]
        return int(np.random.choice(indices, p=corrected_probs))

    def _calculate_route(self) -> "tuple[int, int, int, int]":
        route = []
        jobs_schedule = [0] * len(self.jobs)
        machine_schedule = [0] * len(self.machines)

        for _ in range(sum(len(job.tasks) for job in self.jobs)):
            idx_probs = []
            for j_id, job in enumerate(self.jobs):
                if jobs_schedule[j_id] < len(job.tasks):
                    task = job.tasks[jobs_schedule[j_id]]
                    probability = self._get_probability(
                        machine_schedule,
                        jobs_schedule,
                        task.machine_id,
                        j_id,
                        task.runtime,
                    )
                    idx_probs.append((j_id, probability))

            if not idx_probs:
                continue

            selected_job = self._get_random_job(idx_probs)
            selected_task = self.jobs[selected_job].tasks[jobs_schedule[selected_job]]
            start_time = max(
                jobs_schedule[selected_job], machine_schedule[selected_task.machine_id]
            )
            route.append(
                (
                    selected_job,
                    selected_task.machine_id,
                    start_time,
                    start_time + selected_task.runtime,
                )
            )
            jobs_schedule[selected_job] = start_time + selected_task.runtime
            machine_schedule[selected_task.machine_id] = (
                start_time + selected_task.runtime
            )
            import pdb

            pdb.set_trace()

        print("route: ", route)

        fitness = max(machine_schedule)
        return route, fitness

    def _update_pheromone(
        self, routes: "list[list[tuple[int, int, int, int]]]", fitnesses: "list[int]"
    ) -> "None":
        self.pheromone *= 1 - self.evaporation_rate

        for route, fitness in zip(routes, fitnesses):
            contribution = self.pheromone_deposit / fitness
            for job_idx, machine, start, end in route:
                task_idx = [
                    idx
                    for idx, task in enumerate(self.jobs[job_idx].tasks)
                    if task.machine_id == machine and task.runtime == (end - start)
                ][0]
                self.pheromone[job_idx][task_idx][machine] += contribution


class GAJobScheduler(BaseJobScheduler):
    def __init__(self, data: "list[RAW_TASK]", num_machines: "int") -> "None":
        self.jobs = self._generate_jobs(data)
        self.machines = self._generate_machines(num_machines)
        self.base_chromosome = self._generate_base_chromosome()

    def _generate_base_chromosome(self) -> "Chromosome":
        """
        generate an initial Chromosome object
        """
        _initial_chromosome = [
            [idx] * len(job.tasks) for idx, job in enumerate(self.jobs)
        ]
        return Chromosome(data=[l for ll in _initial_chromosome for l in ll])

    def random_chromosome(self) -> "Chromosome":
        """
        randomize the initial chromosome synthesis
        """
        c = copy.deepcopy(self.base_chromosome)
        random.shuffle(c.data)
        return c

    def create_population(self) -> "list[Chromosome]":
        """
        creates a population (list of Chromosome objects)
        for a given population limit (POPULATION_LIMIT env var)
        """
        population: "list[Chromosome]" = []
        for _ in range(POPULATION_LIMIT):
            c = self.random_chromosome()
            while not self.validate(c):
                c = self.random_chromosome()
            population.append(c)
        return population

    def show_population(self, population: "list[Chromosome]") -> "None":
        """
        ouptuts the current population
        """
        print(f"Population length: {len(population)}")
        if VERBOSE:
            for chromosome in population:
                print(chromosome)

    def reset(self) -> "None":
        """
        resets all jobs and machines of scheduler
        """
        for job in self.jobs:
            job.reset_tasks()

        for machine in self.machines:
            machine.reset()

    def validate(self, chromosome: "Chromosome") -> "bool":
        """
        validates a given chromosome
        """
        for idx in range(len(self.jobs)):
            job_times = chromosome.data.count(idx)
            # the occurencies of a job inside the chromosome
            # should be equal to the number of job's tasks
            if job_times != len(self.jobs[idx].tasks):
                print(f"Invalid generated chromosome: {chromosome.data}")
                return False
        return True

    def fitness(self, chromosome: "Chromosome") -> "int":
        """
        get the fitness of a given chromosome
        """
        self.reset()

        if not self.validate(chromosome):
            return -1

        for job_idx in chromosome.data:
            current_job = self.jobs[job_idx]
            current_task = current_job.get_next_task()

            machine_id = current_task.machine_id
            machine = self.machines[machine_id]

            task_start_time = max(machine.end_time, current_job.task_end_time)
            task_end_time = task_start_time + current_task.runtime

            machine.end_time = task_end_time
            current_job.task_end_time = task_end_time

        end_time_max = -1
        for machine in self.machines:
            if machine.end_time > end_time_max:
                end_time_max = machine.end_time
        if VERBOSE:
            print("Fitness {} for chromosome {}".format(end_time_max, chromosome))
        return end_time_max

    def crossover(self, parent1: "Chromosome", parent2: "Chromosome") -> "Chromosome":
        if SET_CROSSOVER_METHOD == CrossoverMethod.ONE_POINT:
            return self.one_point_crossover(parent1, parent2)
        else:  # SET_CROSSOVER_METHOD == CrossoverMethod.TWO_POINTS
            return self.two_point_crossover(parent1, parent2)

    def one_point_crossover(
        self, parent1: "Chromosome", parent2: "Chromosome"
    ) -> "Chromosome":
        crossover_point = random.randint(1, len(parent1.data) - 1)

        # split the child on on point
        child = Chromosome(data=[-1] * len(parent1.data))
        child.data[:crossover_point] = parent1.data[:crossover_point]

        # fill the rest of child
        fill_i = crossover_point
        for job_idx in parent2.data:
            job_times = child.data.count(job_idx)
            if job_times != len(self.jobs[job_idx].tasks):
                child.data[fill_i] = job_idx
                fill_i += 1

        if not self.validate(child):
            return -1

        return child

    def two_point_crossover(
        self, parent1: "Chromosome", parent2: "Chromosome"
    ) -> "Chromosome":
        crossover_start = random.randint(1, len(parent1.data) - 1)
        crossover_end = random.randint(crossover_start, len(parent1.data))

        # split the child on two points
        child = Chromosome(data=[-1] * len(parent1.data))
        child.data[:crossover_start] = parent1.data[:crossover_start]
        child.data[crossover_end:] = parent1.data[crossover_end:]

        # fill the child after first point
        fill_i = crossover_start
        for job_idx in parent2.data:
            job_times = child.data.count(job_idx)
            if job_times != len(self.jobs[job_idx].tasks):
                child.data[fill_i] = job_idx
                fill_i += 1

        if not self.validate(child):
            return -1

        return child

    def mutate(self, child: "Chromosome") -> "Chromosome":
        if random.random() < MUTATE_THRESHOLD:
            return child
        if SET_MUTATION_MODE == MutationMode.FLIP:
            child = self.flip_mutation(child)
        else:  # SET_MUTATION_MODE == MutationMode.SHIFT
            child = self.shift_mutation(child)

        if not self.validate(child):
            return -1

        return child

    def flip_mutation(self, child: "Chromosome") -> "Chromosome":
        """
        flips two items in the data of a given chromosome
        """
        idx1, idx2 = random.sample(range(len(child.data)), 2)
        child.data[idx1], child.data[idx2] = child.data[idx2], child.data[idx1]
        return child

    def shift_mutation(self, child: "Chromosome") -> "Chromosome":
        """
        shifts the data of a given chromosome for 1 position
        """
        child.data = child.data[-1:] + child.data[:-1]
        return child

    def tournament_select(
        self,
        generation: "list[Chromosome]",
        generation_fitness: "list[int]",
    ):
        """
        applies tournament selection of parent for a given generation
        """
        idxs = random.sample(range(len(generation)), SAMPLE_LEN)
        samples = [generation[idx] for idx in idxs]
        samples_fitness = [generation_fitness[idx] for idx in idxs]
        parent = samples[samples_fitness.index(min(samples_fitness))]
        return parent

    def _stochastic_select(
        self,
        dr: "float",
        ranks: "list[int]",
    ) -> "np.NDArray":
        """
        get the indices on the cumulative fitness array
        """
        r = random.randint(0, dr)
        pattern_positions = [(r + dr * i) for i in range(len(ranks))]
        cum_fitness = np.cumsum(ranks)
        return cum_fitness.searchsorted(pattern_positions)

    def precompute_stochastic_select(
        self,
        generation: "list[Chromosome]",
        generation_fitness: "list[int]",
    ) -> "list[tuple[Chromosome]]":
        """
        applies stochastic selection of parent for a given generation
        based on ranks
        """
        stochastic_parents: "list[tuple[Chromosome]]" = []

        ranks = [
            sorted(generation_fitness, reverse=True).index(x)
            for x in generation_fitness
        ]

        dr = int(sum(ranks) / len(ranks))
        parent1_idx = self._stochastic_select(dr, ranks)
        random.shuffle(parent1_idx)
        parent2_idx = self._stochastic_select(dr, ranks)
        random.shuffle(parent2_idx)
        for idx in range(len(generation)):
            parent1 = generation[parent1_idx[idx]]
            parent2 = generation[parent2_idx[idx]]
            stochastic_parents.append((parent1, parent2))
        return stochastic_parents

    def next_generation(self, generation: "list[Chromosome]") -> "list[Chromosome]":
        """
        calculates the next generation for a given current generation of chromosomes
        """
        new_generation: "list[Chromosome]" = []
        generation_fitness = [self.fitness(chromosome) for chromosome in generation]
        stochastic_parents: "list[tuple[Chromosome]]" = []

        # if stochastic precompute the parents
        if SET_PARENT_SELECTION_MODE == ParentSelection.STOCHASTIC:
            stochastic_parents = self.precompute_stochastic_select(
                generation, generation_fitness
            )

        for idx in range(len(generation)):
            # if tournament get the parents on every iteration
            if SET_PARENT_SELECTION_MODE == ParentSelection.TOURNAMENT:
                parent1 = self.tournament_select(generation, generation_fitness)
                parent2 = self.tournament_select(generation, generation_fitness)
                while parent1 == parent2:
                    # ensure parent 1 not equal to parent 2
                    parent2 = self.tournament_select(generation, generation_fitness)
            else:
                parent1, parent2 = stochastic_parents[idx]

            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_generation.append(child)
        return new_generation

    def calc_best_fitness(
        self, generation: "list[Chromosome]"
    ) -> "tuple[Chromosome, int]":
        """
        get the best chromosome of the generation and the best fitness
        """
        generation_fitness = [self.fitness(chromosome) for chromosome in generation]
        best_fitness = min(generation_fitness)
        best_chromosome = generation[generation_fitness.index(best_fitness)]
        return best_chromosome, best_fitness

    def evolve(self) -> "tuple[list[int], list[int]]":
        """
        runs the optimization for a number of generations (NUM_GENERATIONS)
        """
        generation = self.create_population()
        best_fitnesses = []
        streak = 0
        old_fitness = 0
        best_chromosome, best_fitness = self.calc_best_fitness(generation)
        print(
            "Initial population best fitness: {} from chromosome: {}".format(
                best_fitness, best_chromosome
            )
            if VERBOSE
            else "Initial population best fitness: {}".format(best_fitness)
        )
        for i in range(NUM_GENERATIONS):
            generation = self.next_generation(generation)
            best_chromosome, best_fitness = self.calc_best_fitness(generation)
            print(
                "[{}/{}] Generation best fitness: {} from chromosome: {}".format(
                    i, NUM_GENERATIONS, best_fitness, best_chromosome
                )
                if VERBOSE
                else "[{}/{}] Generation best fitness: {}".format(
                    i, NUM_GENERATIONS, best_fitness
                )
            )
            best_fitnesses.append(best_fitness)
            if streak == STATIONARY_STATE_THRESHOLD:
                print(
                    f"Stationary state reached after {streak} times of equal best fitness {best_fitness}. Breaking"
                )
                break

            if old_fitness == best_fitness:
                streak += 1
            else:
                old_fitness = best_fitness
                streak = 0

        return best_fitnesses

    def run(self) -> "None":
        """
        mimics the execution flow of the ant scheduler.
        """
        r = self.random_chromosome()
        self.validate(r)
        population = self.create_population()
        self.show_population(population)
        return self.evolve()


def load_data() -> "tuple[int, int, list[RAW_TASK]]":
    """
    reads the selected dataset from data.yaml
    """
    f = open(DATA_PATH)
    data_file = yaml.safe_load(f)
    dataset = None
    for _d in data_file["instances"]:
        if _d["name"] == DATASET_NAME:
            dataset = _d["data"]

    lines = [line for line in dataset.splitlines() if line != []][1:]
    lines = [list(map(int, line.split())) for line in lines]

    num_jobs, num_machines = lines[0][0], lines[0][1]
    jobs_data = []
    for job in range(num_jobs):
        job_line = lines[job + 1]
        zipped = list(zip(job_line[0::2], job_line[1::2]))
        jobs_data.append(zipped)

    return num_machines, num_jobs, jobs_data


def plot(best_fitnesses: "list[int]") -> "None":
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(len(best_fitnesses)),
        best_fitnesses,
        marker="o",
        linestyle="-",
        color="b",
        label="Time",
    )
    plt.title("Evolution of fitness", fontsize=16)
    plt.xlabel("Generations", fontsize=14)
    plt.ylabel("Best Fitness", fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    num_machines, num_jobs, jobs_data = load_data()
    if SET_SCHEDULER_TYPE == SchedulerType.ANTS_COLONY:
        scheduler = AntJobScheduler(jobs_data, num_machines)
    else:  # SET_SCHEDULER_TYPE == SchedulerType.GA
        scheduler = GAJobScheduler(jobs_data, num_machines)

    best_fitnesses = scheduler.run()
    plot(best_fitnesses)
