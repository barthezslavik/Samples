import random

# Set the initial population size
pop_size = 100

# Set the mutation rate
mutation_rate = 0.01

# Set the number of generations
num_generations = 1000

# Create the initial population
population = [random.random() for _ in range(pop_size)]

# Run the simulation loop
for generation in range(num_generations):
    # Select a subset of the population for reproduction
    reproduction_pool = random.sample(population, k=pop_size)

    # Create the next generation of the population
    next_generation = []
    for i in range(pop_size):
        # Select two parents at random
        parent1 = reproduction_pool[random.randint(0, pop_size - 1)]
        parent2 = reproduction_pool[random.randint(0, pop_size - 1)]

        # Combine the traits of the parents to create the offspring
        offspring = (parent1 + parent2) / 2

        # Mutate the offspring with a certain probability
        if random.random() < mutation_rate:
            offspring += random.gauss(0, 0.1)

        # Add the offspring to the next generation
        next_generation.append(offspring)

    # Replace the current population with the next generation
    population = next_generation
