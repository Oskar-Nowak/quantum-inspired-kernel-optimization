import numpy as np
import math
import random
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
        mutation_rate=0.01,  
        kernel="rbf",
        verbose_logging=True,
        output_file="output.dat",
        min_C=-3,
        max_C=3,
        min_gamma=-4,
        max_gamma=1,
    ):
        self.X_train = X_train
        self.y_train = y_train

        self.N = population_size
        self.Genome = genome_length
        self.generation_max = generations
        self.MUTATION_RATE = mutation_rate
        self.kernel = kernel
        self.verbose_logging = verbose_logging
        self.output_file = output_file

        # Parametry dynamicznego kąta rotacji (zgodnie z teorią Improved QGA)
        self.theta_min = 0.01 * np.pi
        self.theta_max = 0.05 * np.pi

        self.pop_size = self.N
        # Przechowujemy amplitudy [alpha, beta] dla każdego genu
        # Wymiary: [populacja, długość_genomu, 2]
        self.qpv = np.empty([self.pop_size, self.Genome, 2])
        
        # Zmierzone chromosomy (binarne)
        self.chromosome = np.empty([self.pop_size, self.Genome], dtype=int)
        
        self.fitness = np.empty([self.pop_size])
        self.best_chrom_history = np.empty([self.generation_max])
        
        self.generation = 0
        self.min_C = min_C
        self.max_C = max_C
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        
        self.best_global_fitness = -1
        self.best_global_chromosome = None

        self.history = []

        with open(self.output_file, "w"):
            pass

    def init_population(self):
        # Inicjalizacja w stanie superpozycji: alpha = beta = 1/sqrt(2)
        # Oznacza to 50% szans na 0 i 50% szans na 1
        r2 = 1 / math.sqrt(2.0)
        self.qpv[:, :, 0] = r2  # Alpha
        self.qpv[:, :, 1] = r2  # Beta

    def measure(self):
        """
        Dokonuje kolapsu funkcji falowej.
        Dla każdego qubitu losujemy liczbę [0,1]. 
        Jeśli losowa < |beta|^2 -> stan 1, w przeciwnym razie stan 0.
        """
        for i in range(self.pop_size):
            for j in range(self.Genome):
                # Prawdopodobieństwo stanu |1> to |beta|^2
                prob_one = self.qpv[i, j, 1] ** 2
                if np.random.rand() < prob_one:
                    self.chromosome[i, j] = 1
                else:
                    self.chromosome[i, j] = 0

    def decode_param(self, bits, low, high):
        """Dekoduje ciąg bitów na wartość rzeczywistą w skali logarytmicznej"""
        bit_str = ''.join(str(b) for b in bits)
        if not bit_str: return 10**low 
        int_val = int(bit_str, 2)
        max_val = (2 ** len(bits)) - 1
        normalized = int_val / max_val if max_val > 0 else 0
        # Skala logarytmiczna (np. 10^-3 do 10^3)
        return 10 ** (low + normalized * (high - low))

    def evaluate_fitness(self):
        fitness_total = 0
        current_best_fitness = -1
        current_best_idx = -1

        for i in range(self.pop_size):
            half = self.Genome // 2
            C_bits = self.chromosome[i, 0:half]
            gamma_bits = self.chromosome[i, half:]

            C = self.decode_param(C_bits, self.min_C, self.max_C)
            gamma = self.decode_param(gamma_bits, self.min_gamma, self.max_gamma)

            model = SVC(kernel=self.kernel, C=C, gamma=gamma)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=3, n_jobs=-1)
            
            score = np.mean(scores) * 100
            self.fitness[i] = score
            fitness_total += score

            if score > current_best_fitness:
                current_best_fitness = score
                current_best_idx = i

            if self.verbose_logging:
                print(f"[Gen {self.generation}] Indiv {i}: C={C:.4f}, g={gamma:.4f}, acc={score:.2f}%")

        # Aktualizacja globalnego najlepszego rozwiązania
        if current_best_fitness > self.best_global_fitness:
            self.best_global_fitness = current_best_fitness
            self.best_global_chromosome = self.chromosome[current_best_idx].copy()

        avg_fitness = fitness_total / self.pop_size
        self.best_chrom_history[self.generation] = current_best_fitness

        if self.verbose_logging:
            print(f"--> Gen {self.generation} Stats: Avg={avg_fitness:.2f}%, Best={current_best_fitness:.2f}%")

        self.history.append({
            'generation': self.generation,
            'mean_fitness': avg_fitness,
            'best_fitness': current_best_fitness
        })
        
        with open(self.output_file, "a") as f:
            f.write(f"{self.generation} {avg_fitness} {current_best_fitness}\n")

    def _rotation_angle(self):
        """
        Implementacja Dynamicznego Kąta Rotacji (Improved QGA).
        Kąt maleje liniowo wraz z liczbą generacji, co pozwala na
        eksplorację na początku i eksploatację na końcu.
        """
        ratio = self.generation / self.generation_max
        # Wzór liniowy: theta zmienia się od theta_max do theta_min
        current_theta = self.theta_max - (self.theta_max - self.theta_min) * ratio
        return current_theta

    def rotate(self):
        """Aktualizacja bramek kwantowych (obrót w stronę najlepszego osobnika)"""
        if self.best_global_chromosome is None:
            return

        delta_theta_mag = self._rotation_angle()
        
        for i in range(self.pop_size):
            for j in range(self.Genome):
                # Strategia Lookup Table:
                # Jeśli bit osobnika różni się od bitu najlepszego globalnie -> obróć w jego stronę
                # Jeśli bity są takie same -> nie obracaj (lub mały obrót)
                
                best_bit = self.best_global_chromosome[j]
                curr_bit = self.chromosome[i, j]
                
                # Znak obrotu
                sign = 0
                if curr_bit == 0 and best_bit == 1:
                    sign = 1 # Zwiększamy prawdopodobieństwo 1
                elif curr_bit == 1 and best_bit == 0:
                    sign = -1 # Zwiększamy prawdopodobieństwo 0 (czyli zmniejszamy alpha)
                
                if sign != 0:
                    theta = sign * delta_theta_mag
                    
                    alpha = self.qpv[i, j, 0]
                    beta = self.qpv[i, j, 1]
                    
                    # Macierz rotacji (Wzór 2.16 z pracy)
                    new_alpha = alpha * math.cos(theta) - beta * math.sin(theta)
                    new_beta  = alpha * math.sin(theta) + beta * math.cos(theta)
                    
                    self.qpv[i, j, 0] = new_alpha
                    self.qpv[i, j, 1] = new_beta

    def convergence_gate(self):
        """
        Implementacja Bramki Konwergencji (Quantum Convergence Gate).
        Zapobiega przedwczesnej zbieżności do czystych stanów |0> lub |1>.
        """
        epsilon = 0.02 # Próg tolerancji
        mutation_prob = self.MUTATION_RATE
        
        for i in range(self.pop_size):
            if np.random.rand() < mutation_prob:
                # POPRAWKA: Mutacja (Pauli-X) na konkretnym genie
                # Losujemy index genu (qubitu) do mutacji
                idx = np.random.randint(0, self.Genome)
                
                # Zamieniamy alpha z beta W TYM SAMYM qubicie
                temp = self.qpv[i, idx, 0]
                self.qpv[i, idx, 0] = self.qpv[i, idx, 1]
                self.qpv[i, idx, 1] = temp
                continue

            # Reszta logiki zbieżności (bez zmian)
            for j in range(self.Genome):
                alpha_sq = self.qpv[i, j, 0] ** 2
                
                if alpha_sq < epsilon or alpha_sq > (1 - epsilon):
                    if self.qpv[i, j, 0] > 0:
                        self.qpv[i, j, 0] = math.sqrt(1 - epsilon)
                        self.qpv[i, j, 1] = math.sqrt(epsilon)
                    else:
                        self.qpv[i, j, 0] = math.sqrt(epsilon)
                        self.qpv[i, j, 1] = math.sqrt(1 - epsilon)

    def plot_output(self):
        gens = [x['generation'] for x in self.history]
        means = [x['mean_fitness'] for x in self.history]
        bests = [x['best_fitness'] for x in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(gens, means, label="Średnia dokładność", linestyle='--')
        plt.plot(gens, bests, label="Najlepsza dokładność", linewidth=2)
        plt.xlabel("Generacja")
        plt.ylabel("Dokładność (Accuracy %)")
        plt.title(f"Zbieżność QGA-SVM (Max: {max(bests):.2f}%)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        print("--- STARTING IMPROVED QUANTUM GENETIC ALGORITHM (IQGA-SVM) ---\n")

        self.generation = 0
        self.init_population()
        self.measure()
        self.evaluate_fitness()

        # POPRAWKA: Zmieniono warunek pętli z '<' na '< ... - 1'
        # Ponieważ generacja 0 została zrobiona przed pętlą, musimy zrobić 
        # jeszcze (max - 1) kroków, aby dojść do ostatniego indeksu.
        while self.generation < self.generation_max - 1:
            self.generation += 1
            
            # 1. Rotacja w stronę najlepszego
            self.rotate()
            
            # 2. Bramka Konwergencji (Mutation / Convergence Gate)
            self.convergence_gate()
            
            # 3. Pomiar i Ewaluacja
            self.measure()
            self.evaluate_fitness()
            
        print(f"\nOptimization Finished.")
        print(f"Best Accuracy: {self.best_global_fitness:.2f}%")
        
        # Decode best params
        if self.best_global_chromosome is not None:
            half = self.Genome // 2
            best_C_bits = self.best_global_chromosome[0:half]
            best_gamma_bits = self.best_global_chromosome[half:]
            best_C = self.decode_param(best_C_bits, self.min_C, self.max_C)
            best_gamma = self.decode_param(best_gamma_bits, self.min_gamma, self.max_gamma)
            
            print(f"Best Parameters: C={best_C:.5f}, Gamma={best_gamma:.5f}")
            return best_C, best_gamma
        else:
            print("No improvement found.")
            return None, None