from ray_optics_criteria_ITMO import calc_loss
import time
from rayoptics.environment import *
from matplotlib.colors import LogNorm, PowerNorm, Normalize
from rayoptics.util.misc_math import normalize
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import re
import io
import pandas as pd
from rayoptics.environment import *
import warnings

from deap import base, algorithms
from deap import creator
from deap import tools



import random
import matplotlib.pyplot as plt
import numpy as np


warnings.filterwarnings('ignore')

isdark = False

path2model = 'test.roa'
efl_for_loss = 5  # Focal dist mm | =
fD_for_loss = 2.1  # F/# mm | <=
total_length_for_loss = 7.0  # Total lenght mm | <=
radius_enclosed_energy_for_loss = 50  # micron
perc_max_enclosed_energy_for_loss = 80  # % | >=
perc_min_enclosed_energy_for_loss = 50  # %
min_thickness_for_loss = 0.1  # Lens thickness mm | >=
min_thickness_air_for_loss = 0.0  # Air thickness mm | >=
wavelength = [470.0, 650.0]  # nm
fields = [0., 5., 10., 15., 20.]  # deg
number_of_field = len(fields)
number_of_wavelength = len(wavelength)

opm = open_model('random_gen.roa', info=True)

sm = opm['seq_model']
osp = opm['optical_spec']
pm = opm['parax_model']
em = opm['ele_model']
pt = opm['part_tree']
ar = opm['analysis_results']

# thickness = [sm.get_surface_and_gap(i)[1].thi for i in range(2, len(sm.ifcs) - 1)
#              if type(sm.get_surface_and_gap(i)[1].medium) == ModelGlass]

thickness = [sm.get_surface_and_gap(i)[1].thi for i in range(2, len(sm.ifcs) - 1)]

curvs = [sm.ifcs[i].profile.cv for i in range(2, len(sm.ifcs) - 2)]

n = [sm.get_surface_and_gap(i)[1].medium.n for i in range(len(sm.ifcs) - 1)
     if type(sm.get_surface_and_gap(i)[1].medium) == ModelGlass]

abbe = [sm.get_surface_and_gap(i)[1].medium.v for i in range(len(sm.ifcs) - 1)
        if type(sm.get_surface_and_gap(i)[1].medium) == ModelGlass]

coefs = np.array([sm.ifcs[i].profile.coefs for i in range(len(sm.ifcs))
         if type(sm.ifcs[i].profile) == EvenPolynomial])

ind = thickness + curvs + n  + coefs.flatten().tolist()
len(ind)

LOW = -1

UP = 50

ETA = 20
LENGTH_CHROM = len(ind)  # длина хромосомы, подлежащей оптимизации

# константы генетического алгоритма
POPULATION_SIZE = 10  # количество индивидуумов в популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.2  # вероятность мутации индивидуума
MAX_GENERATIONS = 2  # максимальное количество поколений
HALL_OF_FAME_SIZE = 5

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


# ЗАДАВАТЬ НЕ (АББЕ И Н) А К
def randomPoint(X, Y):

    return np.concatenate([np.random.uniform(0, 1.2, size=len(thickness)),
                           np.random.uniform(-.2, .2, size=len(curvs)),
                           np.random.uniform(0, 1, size=len(n)),
                           np.random.uniform(-0.05, 0.05, size=len(coefs))])


toolbox = base.Toolbox()
toolbox.register("randomPoint", randomPoint, LOW, UP)
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)


def himmelblau(individual):
    
    opm = open_model('random_gen.roa', info=True)
    sm = opm['seq_model']
    osp = opm['optical_spec']
    pm = opm['parax_model']

    osp['fov'] = FieldSpec(osp, key=['object', 'angle'], is_relative=False, flds=fields)
    osp['wvls'] = WvlSpec([(wavelength[0], 1.0), (wavelength[1], 1.0)], ref_wl=0)

    x = individual
    # print(individual)
    
    k = 0

    for i in range(2, len(sm.ifcs) - 1):

        sm.get_surface_and_gap(i)[1].thi = x[k]

        k += 1
    
    for i in range(2, len(sm.ifcs) - 2):

        sm.ifcs[i].profile.cv = x[k]

        k += 1

    for i in range(len(sm.ifcs) - 1):

        if type(sm.get_surface_and_gap(i)[1].medium) == ModelGlass:

            sm.get_surface_and_gap(i)[1].medium.n = 1.54 * x[k] + 1.67 * (1 - x[k])

            sm.get_surface_and_gap(i)[1].medium.v = 75 * x[k] + 39 * (1 - x[k])

            k += 1
    
    for i in range(len(sm.ifcs)):
         
         if type(sm.ifcs[i].profile) == EvenPolynomial:
             
             sm.ifcs[i].profile.coefs = x[k: k + 10]

             k += 11
    try:
        opm.update_model()
        opm.save_model('genetic_opt.roa')
        f = calc_loss('genetic_opt.roa')
        if str(f)=='nan':
            f = 100000
        return f,
    except (ValueError, IndexError) as e:
        return 1000000,

from scoop import futures

toolbox.register("map", futures.map)
toolbox.register("evaluate", himmelblau)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)


from scoop import futures

toolbox.register("map", futures.map)
toolbox.register("evaluate", himmelblau)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxSimulatedBinary, eta=ETA)
toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0 / LENGTH_CHROM)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)


population, logbook = algorithms.eaSimple(population, 
                                          toolbox,
                                          cxpb=P_CROSSOVER,
                                          mutpb=P_MUTATION,
                                          ngen=MAX_GENERATIONS,
                                          halloffame=hof,
                                          stats=stats,
                                          verbose=True)

maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")

best = hof.items[0]
print(best)

plt.ioff()
plt.show()

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()