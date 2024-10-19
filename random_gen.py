"""
Генерирует хорошие по поведению лучей света, но плохие по попаданию в фокус оптические схемы.
"""


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

warnings.filterwarnings('ignore')

isdark = False

# base_param
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

n = np.array([1.54, 2.34, 1.67, 1.32, 1.85])
abbe = np.array([75., 39., 21., 34., 53.], dtype='float')


def random_opt_scheme(fields: np.array = np.array([0., 5., 10., 15., 20.], dtype='float'),
                      f_number: float = 2.1,
                      wave_lenghts: np.array = np.array([470.0, 650.0], dtype='float')):
    """Создает рандомные оптические схемы на параметрах из ТЗ."""

    opm = OpticalModel()
    sm = opm['seq_model']
    osp = opm['optical_spec']
    pm = opm['parax_model']
    em = opm['ele_model']
    pt = opm['part_tree']

    rand_fd = np.random.uniform(0.5, 2.1)

    osp['pupil'] = PupilSpec(osp, key=['image', 'f/#'], value=2.09)
    osp['fov'] = FieldSpec(osp, key=['object', 'angle'], is_relative=False, flds=fields)
    osp['wvls'] = WvlSpec([(wave_lenghts[0], 1.0), (wave_lenghts[1], 1.0)], ref_wl=0)

    lenght: int = np.random.randint(2, 5)
    # Можно поставить случайною или фиксированное количество линз.
    # lenght: int = 3

    radiuses: np.array = np.random.uniform(-30, 30, size=lenght)

    # (lenght x 8) coefs polinomial equation where first two = 0
    lens_coefs: np.array = np.append(np.append(np.zeros((3, 2)), np.random.uniform(-0.02, 0.02, size=(3, 5)), axis=1),
                                     np.zeros((3, 3)), axis=1)

    sm.gaps[0].thi = 1e10

    # Коэфы K задаются от 0 до 1, далее по ТЗ определяются n и abbe, согласно формулам ниже.

    k = np.random.uniform(0., 1.)

    n = 1.54 * k + 1.67 * (1 - k)

    abbe = 75*k + 39*(1-k)

    curv_r = np.random.uniform(-5, 5)

    thick = np.random.uniform(0.1, 0.6)

    air_thick = np.random.uniform(0.001, 0.9)

    conic_c = np.random.uniform(-500, 500)

    conv_lens_r = np.random.uniform(20., 50)

    # Параметры первой линзы были взяты из примера.
    sm.add_surface([0.274, .5, n, abbe])
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(c=0.274,
                                                     coefs=[0.0, 0.009109298409282469, -0.03374649200850791,
                                                            0.01797256809388843, -0.0050513483804677005, 0.0, 0.0, 0.0])
    sm.set_stop()

    conv_lens_flag: bool = False

    for i in range(lenght - 1):

        # Коэфы K задаются от 0 до 1, далее по ТЗ определяются n и abbe, согласно формулам ниже.

        k = np.random.uniform(0., 1.)

        n = 1.54 * k + 1.67 * (1 - k)

        abbe = 75 * k + 39 * (1 - k)

        curv_r = np.random.uniform(-5, 5)

        thick = np.random.uniform(0.3, 0.8)

        air_thick = np.random.uniform(0.001, 0.8)

        conic_c = np.random.uniform(-.2, .2)

        even_polynomial_coefs = np.append(np.append(np.zeros(1), np.random.uniform(-0.02, 0.02, size=4)), np.zeros(3))

        if conv_lens_flag:
            # Создается воздух (правая сторона линзы), записываются параметры кривизны и коэфов полинома
            sm.add_surface([conv_lens_r, air_thick])

            sm.ifcs[sm.cur_surface].profile = EvenPolynomial(ec=-conic_c, coefs=even_polynomial_coefs.tolist())

            conv_lens_flag = False

        else:
            # Создается воздух (правая сторона линзы), записываются параметры кривизны и коэфов полинома
            sm.add_surface([curv_r, air_thick])

            sm.ifcs[sm.cur_surface].profile = EvenPolynomial(ec=conic_c, coefs=even_polynomial_coefs.tolist())

            conv_lens_flag = True

        # Создается линза с фикс кривизной, сделано это было для того чтоб было легче генерить и не было ошибок

        even_polynomial_coefs = np.append(np.append(np.zeros(1), np.random.uniform(-0.02, 0.02, size=4)), np.zeros(3))
        sm.add_surface([0.2, thick, n, abbe])
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(c=0.2, coefs=even_polynomial_coefs.tolist())

    # в конце обязательно создается воздух, с фикс коэфами из примера, тоже для того чтобы не было ошибок
    # в генетическом алгоритмы все коэфы меняются и оптимизируются.

    sm.add_surface([0., .64])
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(ec=0.3,
                                                     coefs=[0.0, -0.0231369463217776, 0.011956554928461116, -0.017782670650182023, 0.004077846642272649, 0.0, 0.0,0.0])

    try:

        # Блок try except для того чтобы ловить ошибки на этапе update_model().

        opm.update_model()

        efl = pm.opt_model['analysis_results']['parax_data'].fod.efl
        fD = pm.opt_model['analysis_results']['parax_data'].fod.fno

        if abs(efl - 5.0) < 0.25 or True:
            # Можно поставить or False, тогда генерация будет производиться до тех пор,
            # пока фокус не будет равен фокусу ТЗ.

            print(efl, fD)
            sm.do_apertures = False

            elmn = [node.id for node in pt.nodes_with_tag(tag='#element')]

            for el in elmn:
                el.do_flat1 = 'always'

                el.do_flat2 = 'always'

            opm.save_model('random_gen.roa')

            return opm

        else:

            opm = random_opt_scheme(fields=fields,
                                    f_number=f_number,
                                    wave_lenghts=wave_lenghts)

            return opm

    except (IndexError, ValueError, TypeError) as e:

        # блоки except служат для игнорирования поломок,
        # в процессе рандомной генерации происходят поломки библиотеки, поэтому если у нас ошибка в генерации,
        # то мы просто перегенерируем всю схему.

        opm = random_opt_scheme(fields=fields,
                                f_number=f_number,
                                wave_lenghts=wave_lenghts)

        return opm


def opt_scheme_plot(opt):
    try:

        layout_plt0 = plt.figure(FigureClass=InteractiveLayout, opt_model=opt,
                                 do_draw_rays=True, do_paraxial_layout=False,
                                 is_dark=isdark).plot()
        return layout_plt0

    except (ValueError, UnboundLocalError, TypeError) as e:

        # Данный блок except аналогичен, т.к иногда алгоритм может создать случайную схему, но не нарисовать.

        opt_scheme_plot(random_opt_scheme())


opm = random_opt_scheme(fields=np.array(fields),
                        f_number=2,
                        wave_lenghts=np.array(wavelength))

opt_scheme_plot(opm)
plt.show()

path = 'random_gen.roa'
print(f'loss = {calc_loss(path)}')

sm = opm['seq_model']
osp = opm['optical_spec']
pm = opm['parax_model']
efl = pm.opt_model['analysis_results']['parax_data'].fod.efl
fD = pm.opt_model['analysis_results']['parax_data'].fod.fno

print(sm.list_model())
print(sm.ifcs)
