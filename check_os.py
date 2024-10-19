from ray_optics_criteria_ITMO import calc_loss
import time

t0=time.time()
path2model='' # insert the path to ray-optics model, for example 'C:/test.roa'
loss=calc_loss(path2model)
elapsed_time=time.time()-t0
print(f'{loss=}, {elapsed_time=} sec')