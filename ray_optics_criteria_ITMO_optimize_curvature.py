isdark = False
from rayoptics.environment import *
from rayoptics.util.misc_math import normalize
import re
import io
from contextlib import redirect_stdout
import numpy as np
from scipy.optimize import minimize

# base_param
path2model = 'test.roa'
efl_for_loss = 5  # Focal dist mm
fD_for_loss = 2.1  # F/# mm
total_length_for_loss = 7.0  # Total lenght mm
radius_enclosed_energy_for_loss = 50  # micron
perc_max_enclosed_energy_for_loss = 80  # %
perc_min_enclosed_energy_for_loss = 50  # %
min_thickness_for_loss = 0.1  # Less thickness mm
min_thickness_air_for_loss = 0.0  # Air thickness mm
wavelength = [470.0, 650.0]  # nm
fields = [0., 5., 10., 15., 20.]  # deg
number_of_field = len(fields)
number_of_wavelength = len(wavelength)
opm = open_model(path2model)

sm = opm['seq_model']
osp = opm['optical_spec']
pm = opm['parax_model']
efl = pm.opt_model['analysis_results']['parax_data'].fod.efl
fD = pm.opt_model['analysis_results']['parax_data'].fod.fno
sm.list_model(), efl, fD, sm.ifcs

thickness = [sm.get_surface_and_gap(i)[1].thi for i in range(2, len(sm.ifcs) - 1)]

# Кривизны
curvs = [sm.ifcs[i].profile.cv for i in range(len(sm.ifcs) - 1)
         if type(sm.ifcs[i].profile) == EvenPolynomial]

# Коэф преломления
n = [sm.get_surface_and_gap(i)[1].medium.n for i in range(len(sm.ifcs) - 1)
     if type(sm.get_surface_and_gap(i)[1].medium) == ModelGlass]

# Аббе
abbe = [sm.get_surface_and_gap(i)[1].medium.v for i in range(len(sm.ifcs) - 1)
        if type(sm.get_surface_and_gap(i)[1].medium) == ModelGlass]

# Коэффициенты полиномов.
coefs = np.array([sm.ifcs[i].profile.coefs for i in range(len(sm.ifcs) - 1)
                  if type(sm.ifcs[i].profile) == EvenPolynomial])

CFS_LEN = len(coefs[0])
# В данном случае мы оптимизируем по толщинам, n, abbe, и коэффициентам полиномов.

thickness = [sm.get_surface_and_gap(i)[1].thi for i in range(1, len(sm.ifcs) - 1)]  # 4

coeff = curvs + thickness + coefs.flatten().tolist()


def funct_loss_enclosed_energy(enclosed_energy, perc_max_enclosed_energy_for_loss, perc_min_enclosed_energy_for_loss):
    if enclosed_energy < perc_max_enclosed_energy_for_loss:
        if enclosed_energy < perc_min_enclosed_energy_for_loss:
            loss_enclosed_energy = 1e3
        else:
            loss_enclosed_energy = (perc_max_enclosed_energy_for_loss - enclosed_energy)
    else:
        loss_enclosed_energy = 0
    return loss_enclosed_energy


def get_thichness(sm):
    f = io.StringIO()
    with redirect_stdout(f):
        sm.list_model()
    s = f.getvalue()
    rows = re.split(r"\n", s)
    thickness_list = []
    thickness_material_list = []
    thickness_air_list = []
    for row in rows[1:-1]:
        row = re.sub(r'\s+', r'!', row)
        values = re.split(r"!", row)
        if values[4] != 'air' and values[4] != '1':
            thickness_material_list.append(float(values[3]))
        if values[4] == 'air' or values[4] == '1':
            thickness_air_list.append(float(values[3]))
        thickness_list.append(float(values[3]))  # 3 - thickness, 2 - curvature, 4 - type of material
    number_of_surfaces = len(rows) - 2
    return thickness_list, thickness_material_list, thickness_air_list, number_of_surfaces


if path2model[-4:] == '.zmx':
    opm, info = open_model(f'{path2model}', info=True)
    opm.save_model(f'{path2model[:-4]}.roa')
    path2model = path2model[:-4] + '.roa'
    print('File format conversion. From zmx to roa.')
elif path2model[-4:] != '.zmx' and path2model[-4:] != '.roa':
    raise Exception('Invalid file format.')


def calc_loss_c1(c1: np.array):
    opm = open_model(f'{path2model}', info=True)
    sm = opm['seq_model']
    osp = opm['optical_spec']
    pm = opm['parax_model']

    osp['fov'] = FieldSpec(osp, key=['object', 'angle'], is_relative=False, flds=fields)
    osp['wvls'] = WvlSpec([(wavelength[0], 1.0), (wavelength[1], 1.0)], ref_wl=0)

    # По аналогии с генетическим алгоритмом, здесь будем проходиться по параметрам модели также.
    k = 0
    for i in range(len(sm.ifcs) - 1):

        if type(sm.ifcs[i].profile) == EvenPolynomial:

            sm.ifcs[i].profile.cv = c1[k]
            k += 1

    for i in range(1, len(sm.ifcs) - 1):

        sm.get_surface_and_gap(i)[1].thi = c1[k]

        k += 1

    for i in range(len(sm.ifcs) - 1):

        if type(sm.ifcs[i].profile) == EvenPolynomial:

            sm.ifcs[i].profile.coefs = c1[k: k + CFS_LEN]

            k += CFS_LEN

    opm.update_model()
    efl = pm.opt_model['analysis_results']['parax_data'].fod.efl  # Focal dist
    fD = pm.opt_model['analysis_results']['parax_data'].fod.fno  # F/#

    print(c1)
    field = 0  # 0-20deg, 4-0deg
    psf = SpotDiagramFigure(opm)
    test_psf = psf.axis_data_array[field][0][0][0]
    test_psf[:, 1] = test_psf[:, 1] - np.mean(test_psf[:, 1])
    # вывод диаграммы функции рассеяния точки
    # plt.plot(test_psf[:,0],test_psf[:,1],'o')
    # plt.rcParams['figure.figsize'] = (8, 8)
    # plt.show()

    efl = pm.opt_model['analysis_results']['parax_data'].fod.efl

    if abs(efl - efl_for_loss) > 0.25:
        loss_focus = 1e2 * (efl - efl_for_loss) ** 2
    else:
        loss_focus = 0

    if abs(fD) >= fD_for_loss:
        loss_FD = 5 * 1e4 * (fD - fD_for_loss) ** 2
    else:
        loss_FD = 0

    thickness_list, thickness_material_list, thickness_air_list, number_of_surfaces = get_thichness(sm)
    total_length = np.sum(thickness_list[1:])
    min_thickness = np.min(thickness_material_list)
    min_thickness_air = np.min(thickness_air_list)
    if (total_length - total_length_for_loss) > 0:
        loss_total_length = 1e4 * (total_length - total_length_for_loss) ** 2
    else:
        loss_total_length = 0

    if min_thickness < min_thickness_for_loss:
        loss_min_thickness = 1e6 * (min_thickness - min_thickness_for_loss) ** 2
    else:
        loss_min_thickness = 0

    if min_thickness_air < min_thickness_air_for_loss:
        loss_min_thickness_air = 8e4 * (min_thickness_air - min_thickness_air_for_loss) ** 2
    else:
        loss_min_thickness_air = 0

    loss_enclosed_energy_all = 0
    loss_rms_all = 0
    temp = 0
    for idx_field in range(number_of_field):
        for idx_wavelength in range(number_of_wavelength):
            test_psf = psf.axis_data_array[idx_field][0][0][idx_wavelength]
            test_psf[:, 1] = test_psf[:, 1] - np.mean(test_psf[:, 1])
            r_psf = np.sort(np.sqrt(test_psf[:, 0] ** 2 + test_psf[:, 1] ** 2))
            enclosed_energy = 100 * np.sum(r_psf <= radius_enclosed_energy_for_loss / 1e3) / len(test_psf[:, 0])
            loss_enclosed_energy = funct_loss_enclosed_energy(enclosed_energy, perc_max_enclosed_energy_for_loss,
                                                              perc_min_enclosed_energy_for_loss)
            loss_enclosed_energy_all = loss_enclosed_energy_all + loss_enclosed_energy

            dl = int(np.floor(len(test_psf[:, 0]) * perc_max_enclosed_energy_for_loss / 100))
            loss_rms = np.sqrt(np.sum((1e3 * r_psf[:dl]) ** 2) / dl)
            loss_rms_all = loss_rms_all + loss_rms

            temp = temp + 1
    loss_enclosed_energy_all = loss_enclosed_energy_all / temp
    loss_rms_all = loss_rms_all / temp
    loss = loss_focus + loss_FD + loss_total_length + loss_min_thickness + loss_min_thickness_air + loss_enclosed_energy_all + loss_rms_all
    # print(f'{loss_focus=}, {loss_FD=},  {loss_total_length=},  {loss_min_thickness=},  {loss_min_thickness_air=},  {loss_enclosed_energy_all=},  {loss_rms_all=}')
    # вывод изображения оптической схемы
    # if loss<100:
    #     layout_plt0 = plt.figure(FigureClass=InteractiveLayout, opt_model=opm,
    #                             do_draw_rays=True, do_paraxial_layout=False,
    #                             is_dark=isdark).plot()
    print(f'final loss:{loss}')
    return (loss)


# начальная кривизна
c1 = coeff
print(f'Initial value of curvature:{c1}')
# оптимизация
result = minimize(calc_loss_c1, np.array(c1))
print(f'Best value of parameter:{result.x}')
print(result.message)
