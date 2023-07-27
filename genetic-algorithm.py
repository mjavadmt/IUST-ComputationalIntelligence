# Q4_graded
# Do not change the above line.
# write your code here
import random


def main_function(x):
    return 168 * x ** 3 - 7.22 * x ** 2 + 15.5 * x - 13.2


class Sample(object):
    def __init__(self, chromosome):
        self.chromosome = chromosome

    def fitness(self):
        chromosome_string = str(self.chromosome)

        left = int(chromosome_string[2:9], 2)
        right = int(chromosome_string[9:], 2)

        x = left + (right / (10 ** len(str(right))))

        if chromosome_string[1] == "1":
            x *= -1

        return abs(main_function(x))


    def cross_over(self, parent2):

        dad = list(str(self.chromosome))
        mom = list(str(parent2.chromosome))

        r1 = random.randint(1, 63)
        numbers = list(range(1, 64))
        numbers.remove(r1)
        r2 = random.choice(numbers)

        mom[min(r1, r2):max(r1, r2)] = dad[min(r1, r2):max(r1, r2)]
        child = dad

        if random.random() < 0.8:
            child = mutation(child)
        child = int("".join(child))
        return Sample(child)


def initilize_chromosome():
    chromosome = '1'

    for i in range(63):
        chromosome += str(random.randint(0, 1))

    chromosome = int(chromosome)
    return chromosome


def mutation(chromosome):
    r = random.randint(1, 63)

    if chromosome[r] == "0":
        chromosome[r] = "1"
    else:
        chromosome[r] = "0"

    return chromosome


def main():
    generation = 0
    samples = []
    number_of_population = 600
    fitness = []

    for i in range(number_of_population):
        samples.append(Sample(initilize_chromosome()))

    for i in range(500):
        generation += 1
        samples = sorted(samples, key=lambda a: a.fitness())
        fitness.append(samples[0].fitness())
        if samples[0].fitness() < 0.001:
            break
        else:
            children = []

            p = int(number_of_population * 0.3)
            children.extend(samples[:p])

            p = int(number_of_population * 0.7)
            for _ in range(p):
                parent1 = random.choice(samples[:50])
                parent2 = random.choice(samples[:50])
                child = parent1.cross_over(parent2)
                children.append(child)

            samples = children

    chromosome_string = str(samples[0].chromosome)
    left = int(chromosome_string[2:9], 2)
    right = int(chromosome_string[9:], 2)
    x = left + (right / (10 ** len(str(right))))
    if chromosome_string[1] == "1":
        x *= -1
    print("approximate Answer is : ", x)
    print("best chromosome: ", samples[0].chromosome)
    print("generation: ", generation)
    print("result of chromosome: ", main_function(x))



if __name__ == '__main__':
    main()

