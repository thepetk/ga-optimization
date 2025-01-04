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


@dataclass
class Chromosome:
    data: "list[int]"


@dataclass
class Job:
    data: "list[tuple[int, int]]"


@dataclass
class Machine:
    name: "str"


class JobScheduler:

    def __init__(
        self, data: "list[list[tuple[int, int]]]", num_machines: "int"
    ) -> "None":
        self.jobs = self._generate_jobs(data)
        self.machines = self._generate_machines(num_machines)
        self.base_chromosome = self._generate_base_chromosome()

    def _generate_jobs(self, data: "list[list[tuple[int, int]]]") -> "list[Job]":
        return [Job(data=row) for row in data]

    def _generate_machines(self, num_machines: "int") -> "list[Machine]":
        return [Machine(name=f"m{idx}") for idx in range(1, num_machines)]

    def _generate_base_chromosome(self) -> "Chromosome":
        _initial_chromosome = [
            [idx] * len(job.data) for idx, job in enumerate(self.jobs)
        ]
        return Chromosome(data=[l for ll in _initial_chromosome for l in ll])

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

    def random_chromosome(self) -> "Chromosome":
        c = copy.deepcopy(self.base_chromosome)
        random.shuffle(c.data)
        return c

    def validate(self, chromosome: "Chromosome") -> "bool":
        for idx in range(len(self.jobs)):
            job_times = chromosome.data.count(idx)
            if job_times != len(self.jobs[idx].data):
                print(f"Invalid generated chromosome: {chromosome.data}")
                return False
        return True

    def fitness(self, chromosome: "Chromosome"):
        pass


if __name__ == "__main__":
    scheduler = JobScheduler(JOBS_DATA, NUM_MACHINES)
    r = scheduler.random_chromosome()
    scheduler.validate(r)
    population = scheduler.create_population()
    scheduler.show_population(population)
