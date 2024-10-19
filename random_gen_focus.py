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
efl_for_loss=5                      # Focal dist mm | =
fD_for_loss=2.1                       # F/# mm | <=
total_length_for_loss=7.0             #Total lenght mm | <=
radius_enclosed_energy_for_loss=50    #micron
perc_max_enclosed_energy_for_loss=80    #% | >=
perc_min_enclosed_energy_for_loss=50    #%
min_thickness_for_loss=0.1              #Lens thickness mm | >=
min_thickness_air_for_loss=0.0            # Air thickness mm | >=
wavelength=[470.0, 650.0]               #nm
fields=[0., 5., 10., 15., 20.]               #deg
number_of_field=len(fields)
number_of_wavelength=len(wavelength)


def random_opt_scheme(fields: np.array = np.array([0., 5., 10., 15., 20.], dtype='float'),
                      f_number: float = 2.1,
                      wave_lenghts: np.array = np.array([470.0, 650.0], dtype='float')):
    """Generate random optic scheme based on params."""

    opm = OpticalModel()
    sm = opm['seq_model']
    osp = opm['optical_spec']
    pm = opm['parax_model']
    em = opm['ele_model']
    pt = opm['part_tree']

    # osp['pupil'] = PupilSpec(osp, key=['object', 'pupil'], value=np.random.uniform(2, 3))
    osp['pupil'] = PupilSpec(osp, key=['image', 'f/#'], value=f_number)
    osp['fov'] = FieldSpec(osp, key=['object', 'angle'], is_relative=False, flds=fields)
    osp['wvls'] = WvlSpec([(wave_lenghts[0], 1.0), (wave_lenghts[1], 1.0)], ref_wl=0)

    # Может быть выбрано случайное количество линз или фиксированное.

    # lenght: int = np.random.randint(2, 6)

    lenght: int = 3

    # (lenght x 8) coefs polinomial equation where first two = 0
    lens_coefs: np.array = np.append(np.append(np.zeros((3, 2)), np.random.uniform(-0.02, 0.02, size=(3, 5)), axis=1),
                                     np.zeros((3, 3)), axis=1)

    opm.radius_mode = True

    sm.gaps[0].thi = 1e10

    k = np.random.uniform(0., 1.)

    n = 1.54 * k + 1.67 * (1 - k)

    abbe = 75 * k + 39 * (1 - k)

    curv_r = np.random.uniform(-5, 5)

    thick = np.random.uniform(0.1, 1.2)

    air_thick = np.random.uniform(0.001, 0.9)

    conic_c = np.random.uniform(-500, 500)

    conv_lens_r = np.random.uniform(20., 50)

    sm.add_surface([0., 0.])
    sm.set_stop()

    sm.add_surface([4, 0.5, n, abbe])
    sm.ifcs[sm.cur_surface].profile = Spherical(r=3)

    conv_lens_flag: bool = False

    for i in range(lenght - 1):

        k = np.random.uniform(0., 1.)

        n = 1.54 * k + 1.67 * (1 - k)

        abbe = 75 * k + 39 * (1 - k)

        curv_r = np.random.uniform(10, 20) if np.random.uniform(10, 20) != 0 else np.random.uniform(10, 20)

        thick = np.random.uniform(0.1, 0.6)

        air_thick = np.random.uniform(0.3, 0.8)

        conic_c = np.random.uniform(2, 500.)

        even_polynomial_coefs = np.append(np.zeros(2), np.random.uniform(-0.02, 0.02, size=8))

        if conv_lens_flag:

            sm.add_surface([curv_r, air_thick])
            sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=curv_r,
                                                             ec=conic_c,
                                                             coefs=even_polynomial_coefs / 10)

            curv_r = np.random.uniform(1, 10) if np.random.uniform(1, 10) != 0 else np.random.uniform(1, 10)

            conic_c = np.random.uniform(2, 40)

            even_polynomial_coefs = np.append(np.zeros(2), np.random.uniform(-0.02, 0.02, size=8))

            conv_lens_flag = False

        else:
            conv_lens_flag = True
            sm.add_surface([curv_r, air_thick])
            sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=curv_r,
                                                             ec=conic_c,
                                                             coefs=even_polynomial_coefs)
            curv_r = np.random.uniform(1, 10) if np.random.uniform(1, 10) != 0 else np.random.uniform(1, 10)

            conic_c = np.random.uniform(2, 40)

            even_polynomial_coefs = np.append(np.zeros(2), np.random.uniform(-0.02, 0.02, size=8))

        sm.add_surface([curv_r, thick, n, abbe])
        sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=curv_r,
                                                         ec=conic_c,
                                                         coefs=even_polynomial_coefs)

    even_polynomial_coefs = np.append(np.zeros(2), np.random.uniform(-0.02, 0.02, size=8))
    sm.add_surface([0., np.random.uniform(0.1, 1.)])
    sm.ifcs[sm.cur_surface].profile = EvenPolynomial(r=np.random.uniform(1., 10.),
                                                     ec=np.random.uniform(10., 50.),
                                                     coefs=even_polynomial_coefs)
    # Прямая линза мб под конец добавить пока хз

    # sm.add_surface([0., .40, n, abbe])
    # dist = np.sum(get_thichness(sm)[0][1:])
    # sm.add_surface([0., (5.0 - dist)])

    try:

        opm.update_model()

        efl = pm.opt_model['analysis_results']['parax_data'].fod.efl
        fD = pm.opt_model['analysis_results']['parax_data'].fod.fno

        if abs(efl - 5.0) < .25 or False:

            print(efl, fD)
            sm.do_apertures = False

            opm.save_model('random_gen_focus.roa')

            return opm

        else:

            random_opt_scheme(fields=fields,
                              f_number=f_number,
                              wave_lenghts=wave_lenghts)

    except (IndexError, ValueError, TypeError) as e:

        opm = random_opt_scheme(fields=fields,
                                f_number=f_number,
                                wave_lenghts=wave_lenghts)
        return opm


def opt_scheme_plot(opt,
                    f_number: float = 2.0):
    try:

        layout_plt0 = plt.figure(FigureClass=InteractiveLayout, opt_model=opt,
                                 do_draw_rays=True, do_paraxial_layout=False,
                                 is_dark=isdark)
        return layout_plt0.plot()

    except (ValueError, UnboundLocalError, TypeError) as e:


        opt_scheme_plot(random_opt_scheme())


opm = random_opt_scheme(fields=np.array(fields),
                        f_number=2,
                        wave_lenghts=np.array(wavelength))
opt_scheme_plot(opm)
plt.show()

path = 'random_gen_focus.roa'
print(f'loss = {calc_loss(path)}')

sm = opm['seq_model']
osp = opm['optical_spec']
pm = opm['parax_model']
efl = pm.opt_model['analysis_results']['parax_data'].fod.efl
fD = pm.opt_model['analysis_results']['parax_data'].fod.fno

print(sm.list_model())
print(sm.ifcs)
