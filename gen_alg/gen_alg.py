#!/usr/bin/env python3
import random
import re
from xgcd_new import xgcd_bitwise 

# --- Load Worst-Case Pairs from File ---
def load_worst_cases(filename):
    """
    Reads worst-case pairs from a file.
    Each line should contain a pair in the format: {number, number},
    for example: {289667, 208885},
    Returns a list of tuples: [(289667, 208885), ...]
    """
    worst_cases = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Use regex to extract two numbers between curly braces.
            match = re.search(r'\{(\d+),\s*(\d+)\}', line)
            if match:
                a = int(match.group(1))
                b = int(match.group(2))
                worst_cases.append((a, b))
    return worst_cases

# --- Compute Consensus Pattern from Worst-Case Pairs ---
def compute_consensus(worst_cases, n_bits, msb_count):
    """
    Computes consensus for the top msb_count bits for both numbers.
    For each bit position (of the msb_count most significant bits), if at least half of
    the worst-case pairs have a '1' in that position, that bit is set to '1' in the consensus.
    Returns a tuple (consensus_a, consensus_b). The consensus is shifted to align with
    the n-bit representation.
    """
    count_a = [0] * msb_count
    count_b = [0] * msb_count
    total = len(worst_cases)
    for (a, b) in worst_cases:
        bin_a = f"{a:0{n_bits}b}"
        bin_b = f"{b:0{n_bits}b}"
        for i in range(msb_count):
            if bin_a[i] == '1':
                count_a[i] += 1
            if bin_b[i] == '1':
                count_b[i] += 1
    consensus_a_bits = ""
    consensus_b_bits = ""
    for i in range(msb_count):
        consensus_a_bits += '1' if count_a[i] >= total/2 else '0'
        consensus_b_bits += '1' if count_b[i] >= total/2 else '0'
    # Shift the consensus bits into the most significant positions.
    consensus_a = int(consensus_a_bits, 2) << (n_bits - msb_count)
    consensus_b = int(consensus_b_bits, 2) << (n_bits - msb_count)
    return (consensus_a, consensus_b)

# --- Structural Similarity ---
def structural_similarity(candidate, consensus, msb_count, n_bits):
    """
    Computes a similarity score between the candidate's top msb_count bits and the consensus pattern.
    The candidate is represented as two n-bit numbers.
    Returns a score between 0 and 2*msb_count (higher means more similar).
    """
    a, b = candidate
    cons_a, cons_b = consensus
    bin_a = f"{a:0{n_bits}b}"
    bin_b = f"{b:0{n_bits}b}"
    bin_cons_a = f"{cons_a:0{n_bits}b}"
    bin_cons_b = f"{cons_b:0{n_bits}b}"
    sim_a = sum(1 for i in range(msb_count) if bin_a[i] == bin_cons_a[i])
    sim_b = sum(1 for i in range(msb_count) if bin_b[i] == bin_cons_b[i])
    return sim_a + sim_b

# --- Base Fitness Function (Iteration Count) ---
def base_fitness(candidate, n_bits=256):
    """
    Returns the iteration count produced by xgcd_bitwise for candidate (a, b).
    Higher iteration counts mean the candidate is "worse" (and hence fitter for our search).
    """
    a, b = candidate
    if b == 0:
        return 0  # Avoid trivial or degenerate cases.
    _, iterations, _ = xgcd_bitwise(a, b, total_bits=n_bits, approx_bits=4)
    return iterations

# --- Augmented Fitness Function (Original Additive Form) ---
def augmented_fitness(candidate, consensus, msb_count, n_bits, weight=0.5):
    """
    Combines the base fitness (iteration count) with a bonus for matching structural features.
    The bonus is simply weight times the structural similarity score.
    """
    return base_fitness(candidate, n_bits) + weight * structural_similarity(candidate, consensus, msb_count, n_bits)

# --- Create a Random Candidate ---
def random_candidate(n=256):
    """
    Generates a candidate as a tuple of two n-bit numbers.
    """
    a = random.randint(1, (1 << n) - 1)
    b = random.randint(1, (1 << n) - 1)
    return (a, b)

# --- Crossover Function ---
def crossover(parent1, parent2, n=256):
    """
    Perform single-point crossover on two parent candidates.
    Each candidate is represented as two n-bit numbers; we convert them to a 2*n bit string,
    perform crossover, and then split them back.
    """
    a1, b1 = parent1
    a2, b2 = parent2
    # Convert each number to a fixed-width binary string.
    bin_a1 = f"{a1:0{n}b}"
    bin_b1 = f"{b1:0{n}b}"
    bin_a2 = f"{a2:0{n}b}"
    bin_b2 = f"{b2:0{n}b}"
    
    # Concatenate to form chromosomes of length 2*n.
    chromo1 = bin_a1 + bin_b1
    chromo2 = bin_a2 + bin_b2
    
    # Choose a random crossover point (avoid extreme ends).
    point = random.randint(1, 2 * n - 1)
    child1_chromo = chromo1[:point] + chromo2[point:]
    child2_chromo = chromo2[:point] + chromo1[point:]
    
    # Split the chromosomes back into two parts for a and b.
    child1 = (int(child1_chromo[:n], 2), int(child1_chromo[n:], 2))
    child2 = (int(child2_chromo[:n], 2), int(child2_chromo[n:], 2))
    return child1, child2

# --- Mutation Function ---
def mutate(candidate, mutation_rate=0.01, n=256):
    """
    Mutate a candidate by flipping bits with a probability equal to mutation_rate.
    """
    a, b = candidate
    chromo = f"{a:0{n}b}" + f"{b:0{n}b}"
    mutated = ""
    for bit in chromo:
        if random.random() < mutation_rate:
            mutated += '1' if bit == '0' else '0'
        else:
            mutated += bit
    new_a = int(mutated[:n], 2)
    new_b = int(mutated[n:], 2)
    return (new_a, new_b)

# --- Main GA Routine ---
def main():
    # GA Parameters (reverting to original settings)
    population_size = 1000
    generations = 100
    mutation_rate = 0.05
    elite_fraction = 0.05
    n_bits = 256
    msb_count = 64
    weight = 1.0
    
    # Load worst-case pairs from file.
    worst_cases = load_worst_cases("results19bits.txt")
    
    # Compute the consensus pattern from worst-case pairs.
    consensus = compute_consensus(worst_cases, n_bits, msb_count)
    
    # Seed the population: use up to 20% worst-case pairs, then fill the rest with random candidates.
    seed_fraction = 0.2
    seed_count = min(len(worst_cases), int(population_size * seed_fraction))
    seeded_population = worst_cases[:seed_count]
    random_population = [random_candidate(n_bits) for _ in range(population_size - seed_count)]
    population = seeded_population + random_population

    # Evolution loop.
    for generation in range(generations):
        # Evaluate fitness for each candidate using the augmented fitness function.
        fitness_values = [
            augmented_fitness(ind, consensus, msb_count, n_bits, weight)
            for ind in population
        ]
        # Sort population by fitness (descending: higher is better).
        sorted_population = [
            ind for _, ind in sorted(zip(fitness_values, population), key=lambda x: x[0], reverse=True)
        ]
        best_fitness = max(fitness_values)
        print(f"Generation {generation} best fitness: {best_fitness}")
        
        # Elitism: Keep a top fraction of the population.
        elite_count = int(elite_fraction * population_size)
        new_population = sorted_population[:elite_count]
        
        # Generate offspring until the population is replenished.
        while len(new_population) < population_size:
            # Selection: choose parents from the top candidates (e.g., top 20).
            parent1 = random.choice(sorted_population[:20])
            parent2 = random.choice(sorted_population[:20])
            
            # Crossover.
            child1, child2 = crossover(parent1, parent2, n_bits)
            # Mutation.
            child1 = mutate(child1, mutation_rate, n_bits)
            child2 = mutate(child2, mutation_rate, n_bits)
            
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        
        population = new_population

    # Report the best candidate found after evolution.
    fitness_values = [
        augmented_fitness(ind, consensus, msb_count, n_bits, weight)
        for ind in population
    ]
    best_index = fitness_values.index(max(fitness_values))
    best_candidate = population[best_index]
    print("Best candidate found:", best_candidate,
          "with augmented fitness:", augmented_fitness(best_candidate, consensus, msb_count, n_bits, weight))
    # Optionally, run xgcd_bitwise individually to see its iteration count.
    gcd_val, iterations, avg_bit_clears = xgcd_bitwise(best_candidate[0], best_candidate[1], total_bits=n_bits, approx_bits=4)
    print(f"(Individual run) GCD: {gcd_val}, Iterations: {iterations}, Avg. bit clears: {avg_bit_clears}")

if __name__ == "__main__":
    main()
