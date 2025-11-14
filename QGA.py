import numpy as np
import math
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


class QGA:
    def __init__(
        self,
        X_train,
        y_train,
        population_size=30,
        genome_length=16,
        generations=100,
        p_alpha=0.5,
        pop_mutation_rate=0.05,
        mutation_rate=0.01,
        kernel="rbf",
        verbose=True,
        output_file="output.dat",
    ):
        self.X_train = X_train
        self.y_train = y_train

        self.N = population_size
        self.Genome = genome_length
        self.generation_max = generations
        self.P_ALPHA = p_alpha
        self.POP_MUTATION_RATE = pop_mutation_rate
        self.MUTATION_RATE = mutation_rate
        self.kernel = kernel
        self.verbose = verbose
        self.output_file = output_file

        self.pop_size = self.N + 1
        self.genome_length = self.Genome + 1
        self.top_bottom = 3

        self.QuBitZero = np.array([1.0, 0.0])
        self.AlphaBeta = np.empty([self.top_bottom])
        self.fitness = np.empty([self.pop_size])
        self.probability = np.empty([self.pop_size])

        self.qpv = np.empty([self.pop_size, self.genome_length, self.top_bottom])
        self.nqpv = np.empty([self.pop_size, self.genome_length, self.top_bottom])

        self.chromosome = np.empty([self.pop_size, self.genome_length], dtype=int)
        self.best_chrom = np.empty([self.generation_max], dtype=int)

        self.generation = 0

        with open(self.output_file, "w"):
            pass

    def init_population(self):
        r2 = math.sqrt(2.0)
        h = np.array([[1 / r2, 1 / r2], [1 / r2, -1 / r2]])
        rot = np.empty([2, 2])

        for i in range(1, self.pop_size):
            for j in range(1, self.genome_length):
                theta = math.radians(np.random.uniform(0, 1) * 90)

                rot[0, 0] = math.cos(theta)
                rot[0, 1] = -math.sin(theta)
                rot[1, 0] = math.sin(theta)
                rot[1, 1] = math.cos(theta)

                self.AlphaBeta[0] = rot[0, 0] * (h[0][0] * self.QuBitZero[0]) + rot[0, 1] * (h[0][1] * self.QuBitZero[1])
                self.AlphaBeta[1] = rot[1, 0] * (h[1][0] * self.QuBitZero[0]) + rot[1, 1] * (h[1][1] * self.QuBitZero[1])

                self.qpv[i, j, 0] = np.around(2 * float(self.AlphaBeta[0]) ** 2, 2)
                self.qpv[i, j, 1] = np.around(2 * float(self.AlphaBeta[1]) ** 2, 2)

    def measure(self):
        for i in range(1, self.pop_size):
            for j in range(1, self.genome_length):
                if self.P_ALPHA <= self.qpv[i, j, 0]:
                    self.chromosome[i, j] = 0
                else:
                    self.chromosome[i, j] = 1

    def decode_param(self, bits, low, high):
        bit_str = ''.join(str(b) for b in bits)
        int_val = int(bit_str, 2)
        normalized = int_val / (2 ** len(bits) - 1)
        return 10 ** (low + normalized * (high - low))

    def evaluate_fitness(self):
        fitness_total = 0

        for i in range(1, self.pop_size):
            half = self.genome_length // 2
            C_bits = self.chromosome[i, 1:half+1]
            gamma_bits = self.chromosome[i, half+1:self.genome_length + 1]

            C = self.decode_param(C_bits, -3, 3)
            gamma = self.decode_param(gamma_bits, -4, 1)

            model = SVC(kernel=self.kernel, C=C, gamma=gamma)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=5)

            self.fitness[i] = np.mean(scores) * 100
            fitness_total += self.fitness[i]

            if self.verbose:
                print(
                    f"[Gen {self.generation}] Chromosome {i}:"
                    f" C={C:.5f}, gamma={gamma:.5f}, fitness={self.fitness[i]:.2f}"
                )

        avg_fitness = fitness_total / self.N
        best_idx = np.argmax(self.fitness[1:self.N + 1]) + 1
        self.best_chrom[self.generation] = best_idx

        if self.verbose:
            print(f"→ Generation {self.generation}: mean={avg_fitness:.3f}, best={self.fitness[best_idx]:.3f}\n")

        with open(self.output_file, "a") as f:
            f.write(f"{self.generation} {avg_fitness}\n")

    def rotate(self):
        rot = np.empty([2, 2])

        best = int(self.best_chrom[self.generation])

        for i in range(1, self.pop_size):
            for j in range(1, self.genome_length):
                if self.fitness[i] < self.fitness[best]:

                    if self.chromosome[i, j] == 0 and self.chromosome[best, j] == 1:
                        delta_theta = 0.0785398163
                    elif self.chromosome[i, j] == 1 and self.chromosome[best, j] == 0:
                        delta_theta = -0.0785398163
                    else:
                        continue

                    rot[0, 0] = math.cos(delta_theta)
                    rot[0, 1] = -math.sin(delta_theta)
                    rot[1, 0] = math.sin(delta_theta)
                    rot[1, 1] = math.cos(delta_theta)

                    self.nqpv[i, j, 0] = rot[0, 0] * self.qpv[i, j, 0] + rot[0, 1] * self.qpv[i, j, 1]
                    self.nqpv[i, j, 1] = rot[1, 0] * self.qpv[i, j, 0] + rot[1, 1] * self.qpv[i, j, 1]

                    self.qpv[i, j, 0] = round(self.nqpv[i, j, 0], 2)
                    self.qpv[i, j, 1] = round(1 - self.qpv[i, j, 0], 2)

    def mutate(self):
        for i in range(1, self.pop_size):
            up = np.random.randint(0, 101) / 100
            if up <= self.POP_MUTATION_RATE:
                for j in range(1, self.genome_length):
                    um = np.random.randint(0, 101) / 100
                    if um <= self.MUTATION_RATE:
                        self.nqpv[i, j, 0] = self.qpv[i, j, 1]
                        self.nqpv[i, j, 1] = self.qpv[i, j, 0]
                    else:
                        self.nqpv[i, j, :] = self.qpv[i, j, :]
            else:
                for j in range(1, self.genome_length):
                    self.nqpv[i, j, :] = self.qpv[i, j, :]

        self.qpv[:] = self.nqpv[:]

    def plot_output(self):
        data = np.loadtxt(self.output_file)
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x, y)
        plt.xlabel("Generation")
        plt.ylabel("Fitness average")
        plt.xlim(0, self.generation_max)
        plt.show()

    def run(self):
        print("QUANTUM GENETIC ALGORITHM\n")

        self.generation = 0
        self.init_population()
        self.measure()
        self.evaluate_fitness()

        while self.generation < self.generation_max - 1:
            if self.verbose:
                print(f"=== GENERATION {self.generation+1} ===")

            self.rotate()
            self.mutate()

            self.generation += 1
            self.measure()
            self.evaluate_fitness()
