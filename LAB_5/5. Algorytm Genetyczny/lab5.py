import random

def select_roulette_rule(fitnesses):
    sum_fit = sum(fitnesses)
    selected = []
    for _ in range(len(fitnesses)):
        r = random.random() * sum_fit
        sf = 0
        for j, f in enumerate(fitnesses):
            sf += f
            if sf > r:
                selected.append(j)
                break
    return selected


def crossover(a, b):
    cross_point = random.randint(0, len(a) - 1)
    new_a = [a[i] if i < cross_point else b[i] for i in range(len(a))]
    new_b = [b[i] if i < cross_point else a[i] for i in range(len(a))]
    return new_a, new_b


def mutate(individual, mutation_prob):
    for i in range(len(individual)):
        if random.random() < mutation_prob:
            individual[i] = 1 - individual[i]
    return individual


def genetic_algorithm(fitness, random_solution, iterations, pop_size, crossover_prob=0.5, mutation_prob=0.1):
    population = [random_solution() for _ in range(pop_size)]

    for gen in range(iterations):
        fitnesses = [fitness(individual) for individual in population]
        print("f:", fitnesses)

        selected = select_roulette_rule(fitnesses)
        print("  ", selected)

        new_population = []
        for i in range(int(len(population) / 2)):
            parent1 = population[selected[i * 2]]
            parent2 = population[selected[i * 2 + 1]]

            if random.random() < crossover_prob:
                offspring1, offspring2 = crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1[:], parent2[:]

            offspring1 = mutate(offspring1, mutation_prob)
            offspring2 = mutate(offspring2, mutation_prob)

            new_population.extend([offspring1, offspring2])

        population = new_population

    return population


def random_packing(n):
    return [random.randint(0, 1) for _ in range(n)]


def value_knapsack(knapsack, packing):
    value = 0
    weight = 0
    for i, item in enumerate(knapsack['items']):
        value += item['value'] if packing[i] == 1 else 0
        weight += item['weight'] if packing[i] == 1 else 0
        if weight > knapsack['capacity']:
            return 0
    return value


def main():
    knapsack = {
        "capacity": 10,
        "items": [{"weight": 1, "value": 2}, {"weight": 2, "value": 3},
                  {"weight": 5, "value": 6}, {"weight": 4, "value": 10}]
    }
    genetic_algorithm(
        fitness=lambda x: value_knapsack(knapsack, x),
        random_solution=lambda: random_packing(len(knapsack['items'])),
        iterations=10,
        pop_size=10,
        crossover_prob=0.7,
        mutation_prob=0.02
    )


if __name__ == "__main__":
    main()