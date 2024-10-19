"""Генетический алгоритм, принимает параметры для оптимизации, далее оптимизирует."""

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

file_path = 'random_gen_focus.roa'  # Путь файла который мы хотим оптимизировать.
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

opm = open_model(file_path, info=True)

sm = opm['seq_model']
osp = opm['optical_spec']
pm = opm['parax_model']
em = opm['ele_model']
pt = opm['part_tree']
ar = opm['analysis_results']

ppl = [2.45]

# Толщины
thickness = [sm.get_surface_and_gap(i)[1].thi for i in range(2, len(sm.ifcs) - 1)]

# Кривизны
curvs = [sm.ifcs[i].profile.cv for i in range(2, len(sm.ifcs) - 2)
         if type(sm.ifcs[i].profile) == EvenPolynomial]

# Коэф преломления
n = [sm.get_surface_and_gap(i)[1].medium.n for i in range(len(sm.ifcs) - 1)
     if type(sm.get_surface_and_gap(i)[1].medium) == ModelGlass]

# Аббе
abbe = [sm.get_surface_and_gap(i)[1].medium.v for i in range(len(sm.ifcs) - 1)
        if type(sm.get_surface_and_gap(i)[1].medium) == ModelGlass]

# Коэффициенты полиномов.
coefs = np.array([sm.ifcs[i].profile.coefs for i in range(len(sm.ifcs))
                  if type(sm.ifcs[i].profile) == EvenPolynomial])

CFS_LEN = len(coefs[0])
# В данном случае мы оптимизируем по толщинам, n, abbe, и коэффициентам полиномов.
ind = thickness + n + coefs.flatten().tolist()

# Если хотите оптимизировать по другим параметрам пишите например так:
# ind = curvs
# в таком варианте оптимизация будет проходить только по кривизнам.

# ВАЖНО!!!
# 1. Количество параметров для оптимизации не сильно влияет на скорость работы алгоритма.
# 2. Для многопоточности запускайте скрипт через командную строку из директории: (нужен модуль scoop)
# pip install scoop
# Пример запуска:
# cd path (переходим в директорию)
# py -m scoop multiproc_genetic.py


LOW = -1

UP = 50

ETA = 15  # Похожесть потомков на родителей, обычно от 10 до 20

LENGTH_CHROM = len(ind)  # длина хромосомы, подлежащей оптимизации

# константы генетического алгоритма
POPULATION_SIZE = 1000  # количество индивидуумов в популяции (для хорошего лосса следует ставить от 300влияет на время работы))
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.2  # вероятность мутации индивидуума
MAX_GENERATIONS = 20  # максимальное количество поколений (для хорошего лосса следует ставить от 15(влияет на время работы))
HALL_OF_FAME_SIZE = 5  # количество лучших особей в каждом поколении

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def randomPoint(X, Y):
    """
    Данная функция создает случайную особь.
    !!!ВАЖНО!!!
    Коэфы задаются в соответствии с тем, что вы собираетесь оптимизировать.
    Например ваш индивид ind = thickness + n + coefs.flatten().tolist(), ТОГДА КОММЕНТИРУЙТЕ ВСЕ КРОМЕ ЭТИХ ПАРАМЕТРОВ
    !!!ВАЖНО!!!
    Т.к n и abbe задаются согласно ТЗ относительно коэфа k, массив np.random.uniform(0, 1, size=len(n))
    отвечает за эти коэффициенты k.
    :param X: не устанавливать (нужно для работы алгоритма)
    :param Y: не устанавливать (нужно для работы алгоритма)
    :return:
    """
    # доавляйте или удаляйте то, что хотите или не хотите оптимизировать, исключительно в таком же порядке,
    # как и при инициализации.
    # Н-р: Если вы оптимизируете только thickness и curvs, тогда return будет выглядеть так:

    # return np.concatenate([np.random.uniform(0.1, 1.2, size=len(thickness)),
    #      np.random.uniform(-0.2, 0.2, size=len(curvs)),
    #      ])

    return np.concatenate([np.random.uniform(0.1, 1.2, size=len(thickness)),  # Комментировать если не нужно менять толщины
         np.random.uniform(0, 1, size=len(n)),  # Комментировать если не хотим менять n и abbe
         np.random.uniform(-0.2, 0.2, size=len(coefs.flatten().tolist()))  # Комментировать если не хотим менять коэфы
         ])


toolbox = base.Toolbox()

# Случайная особь создается так:
toolbox.register("randomPoint", randomPoint, LOW, UP)

# Создание особи:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)

# Создание популяции:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
population = toolbox.populationCreator(n=POPULATION_SIZE)


# !!!ВАЖНО!!!
# ФУНКЦИЯ КОТОРАЯ СЧИТАЕТ ЛОСС, МЫ ЕЕ МИНИМИЗИРУЕМ.
def himmelblau(individual):
    """
    Данная функция записывает коэфы из нашего индивидума(особи) в параметры оптической схемы, далее происходит подсчет лосса
    и соответственно оптимизация.
    :param individual: особь
    :return: loss
    """
    x = individual  # Наша особь

    opm = open_model(file_path, info=True)
    sm = opm['seq_model']
    osp = opm['optical_spec']
    pm = opm['parax_model']

    osp['fov'] = FieldSpec(osp, key=['object', 'angle'], is_relative=False, flds=fields)
    osp['wvls'] = WvlSpec([(wavelength[0], 1.0), (wavelength[1], 1.0)], ref_wl=0)

    # Т.к наша особь - список параметров, расположенных в строгом порядке, то чтобы не перепутать порядок,
    # задается счетчик к, который будет увеличиваться вместе с записыванием параметров в нашу оптическую схему.

    k = 0
    # Записываем толщины
    for i in range(2, len(sm.ifcs) - 1):
        sm.get_surface_and_gap(i)[1].thi = x[k]

        k += 1

    # Записываем кривизны(ЕСЛИ МЫ УКАЗАЛИ ЧТО ХОТИМ ОПТИМИЗИРОВАТЬ ПО НИМ, ТО РАСКОММЕНТИРОВАТЬ!!!)
    # for i in range(2, len(sm.ifcs) - 2):
    #     if type(sm.ifcs[i].profile) == EvenPolynomial:
    #         sm.ifcs[i].profile.cv = x[k]

    #         k += 1

    # Записываем n и аббе, согласно формулы из ТЗ.
    for i in range(len(sm.ifcs) - 1):

        if type(sm.get_surface_and_gap(i)[1].medium) == ModelGlass:
            sm.get_surface_and_gap(i)[1].medium.n = 1.54 * x[k] + 1.67 * (1 - x[k])

            sm.get_surface_and_gap(i)[1].medium.v = 75 * x[k] + 39 * (1 - x[k])

            k += 1

    # Записываем коэффициенты
    for i in range(len(sm.ifcs)):

        if type(sm.ifcs[i].profile) == EvenPolynomial:
            sm.ifcs[i].profile.coefs = x[k: k + CFS_LEN]
            k += CFS_LEN
    try:
        # Блок try, except служит для ловли ошибок в моменте update_model()
        # когда мы получаем ошибку - возвращаем большой лосс.
        opm.update_model()
        opm.save_model('genetic_opt.roa')
        f = calc_loss('genetic_opt.roa')
        if str(f) == 'nan':
            f = 10000
        return f,
    except (ValueError, IndexError) as e:
        return 10000,


# Модуль scoop отвечает за многопоточность, использование данного модуля в разы увеличивает скорость работы алгоритма
# Процесс запуска алгоритма с многопоточностью описан в Readme
from scoop import futures

toolbox.register("map", futures.map)

# Функция для оптимизации(определена выше)
toolbox.register("evaluate", himmelblau)

# Турнирный отбор - отбор лучших особей
toolbox.register("select", tools.selTournament, tournsize=3)

# Метод скрещевания особей
toolbox.register("mate", tools.cxSimulatedBinary, eta=ETA)

# Метод мутации особей
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

print(f'Лучшая особь: {best}')


# def save_gen_model(num):
#     """
#     Данная функция сохраняет 5 лучших особей (оптических схем) в папку bests, называет
#     bests/best{номер_инд, рандомное число}, лучше открывайте папку low_loss, т.к там в названии пишется лосс
#     Папка bests скорее нужна для вывода 5 лучших моделей, чтобы не потерять ни одну
#
#     Важно!
#     Если запускаете в многопоточности, то создастся 5*(количество потоков) файлов.
#
#     :param num: индекс (нужен далее в цикле)
#     :return:
#     """
#     opm = open_model(file_path)
#     sm = opm['seq_model']
#     osp = opm['optical_spec']
#     pm = opm['parax_model']
#     em = opm['ele_model']
#     pt = opm['part_tree']
#     ar = opm['analysis_results']
#
#     x = hof.items[num]
#     k = 0
#
#     for i in range(2, len(sm.ifcs) - 1):
#         sm.get_surface_and_gap(i)[1].thi = x[k]
#
#         k += 1
#
#     for i in range(len(sm.ifcs) - 1):
#
#         if type(sm.get_surface_and_gap(i)[1].medium) == ModelGlass:
#             sm.get_surface_and_gap(i)[1].medium.n = 1.54 * x[k] + 1.67 * (1 - x[k])
#
#             sm.get_surface_and_gap(i)[1].medium.v = 75 * x[k] + 39 * (1 - x[k])
#
#             k += 1
#
#     for i in range(len(sm.ifcs)):
#
#         if type(sm.ifcs[i].profile) == EvenPolynomial:
#             sm.ifcs[i].profile.coefs = x[k: k + CFS_LEN]
#
#             k += CFS_LEN
#
#     opm.update_model()
#     opm.save_model(f'bests/best{num, np.random.randint(0, 10000)}.roa')
#
#
# for i in range(len(hof.items)):
#     # Сохраняем n лучших оптических схем
#     save_gen_model(i)
