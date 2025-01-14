import copy
from dataclasses import dataclass
import random

JOBS_DATA = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3)],
]
NUM_MACHINES = 3
POPULATION_LIMIT = 1000

MUTATE_THRESHOLD = 0.5

RAW_TASK = list[tuple[int, int]]


@dataclass
class Chromosome:
    data: "list[int]"


@dataclass
class Task:
    task_id: "int"
    runtime: "int"
    machine_id: "int"


class Job:
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
        self.tasks_iterator = iter(self.tasks)
        self.task_end_time = 0

    def get_next_task(self) -> "Task":
        return next(self.tasks_iterator)


@dataclass
class Machine:
    machine_id: "int"
    end_time: "int" = 0

    def reset(self) -> "None":
        self.end_time = 0


class JobScheduler:

    def __init__(self, data: "list[RAW_TASK]", num_machines: "int") -> "None":
        self.jobs = self._generate_jobs(data)
        self.machines = self._generate_machines(num_machines)
        self.base_chromosome = self._generate_base_chromosome()

    def _generate_jobs(self, data: "list[list[tuple[int, int]]]") -> "list[Job]":
        return [Job(job_id=idx, raw_task_list=row) for idx, row in enumerate(data)]

    def _generate_machines(self, num_machines: "int") -> "list[Machine]":
        return [Machine(machine_id=idx) for idx in range(num_machines)]

    def _generate_base_chromosome(self) -> "Chromosome":
        _initial_chromosome = [
            [idx] * len(job.tasks) for idx, job in enumerate(self.jobs)
        ]
        return Chromosome(data=[l for ll in _initial_chromosome for l in ll])

    def random_chromosome(self) -> "Chromosome":
        c = copy.deepcopy(self.base_chromosome)
        random.shuffle(c.data)
        return c

    def create_population(self) -> "list[Chromosome]":
        population: "list[Chromosome]" = []
        for _ in range(POPULATION_LIMIT):
            c = self.random_chromosome()
            while not self.validate(c):
                c = self.random_chromosome()
            population.append(c)
        return population

    def show_population(self, population: "list[Chromosome]") -> "None":
        print(f"Population length: {len(population)}")
        for chromosome in population:
            print(chromosome)

    def reset(self) -> "None":
        for job in self.jobs:
            job.reset_tasks()

        for machine in self.machines:
            machine.reset()

    def validate(self, chromosome: "Chromosome") -> "bool":
        for idx in range(len(self.jobs)):
            job_times = chromosome.data.count(idx)
            if job_times != len(self.jobs[idx].tasks):
                print(f"Invalid generated chromosome: {chromosome.data}")
                return False
        return True

    def fitness(self, chromosome: "Chromosome") -> "int":
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

        print("Fitness {} for chromosome {}".format(end_time_max, chromosome))
        return end_time_max

    def crossover(self, parent1: "Chromosome", parent2: "Chromosome") -> "Chromosome":
        crossover_start = random.randint(1, len(parent1) - 1)
        crossover_end = random.randint(crossover_start, len(parent1))

        child = [-1] * len(parent1.data)
        child[:crossover_start] = parent1.data[:crossover_start]
        child[crossover_end:] = parent1.data[crossover_end:]

        fill_i = crossover_start
        for job_idx in parent2.data:
            job_times = child.count(job_idx)
            if job_times != len(self.jobs[job_idx]):
                child[fill_i] = job_idx
                fill_i += 1

        if not self.validate(child):
            return -1

        return child

    def mutate(self, child: "Chromosome") -> "Chromosome":
        if random.random() < MUTATE_THRESHOLD:
            return child

        idx1, idx2 = random.sample(range(len(child.data)), 2)
        child.data[idx1], child.data[idx2] = child.data[idx2], child.data[idx1]

        if not self.validate(child):
            return -1

        return child


if __name__ == "__main__":
    scheduler = JobScheduler(JOBS_DATA, NUM_MACHINES)
    r = scheduler.random_chromosome()
    scheduler.validate(r)
    population = scheduler.create_population()
    scheduler.show_population(population)
    scheduler.fitness(r)
