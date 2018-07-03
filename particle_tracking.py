'''functions for point cloud propagation'''
from multiprocessing import Pool
from functools import partial
import warnings as warn

import numpy as np
from numba import jit, njit

from radon_veto.noise_generation import *
from radon_veto.config import *
from radon_veto.convenience_functions import *
#import pdb

#To-do: make more functions run in no-python mode.

interp_velocity_array = np.load(array_filename)

@jit
def f(y, t, seed, dt, velocity_array_with_noise):
    #pylint: disable=unused-argument
    '''Derivative function'''
    if any(np.isnan(y).flatten()):
        return y
    coord_indices = interp_index_from_coord(y)
    coord_indices2 = []
    if (coord_indices < np.array([300, 300, 300])).all():
        v = velocity_array_with_noise[coord_indices[0],
                                      coord_indices[1],
                                      coord_indices[2], :]
    else:
        print('Warning: exceeded bounding box at {}'.format(y))
        for i, coord in enumerate(coord_indices):
            coord_indices2.append(min(coord,
                                      velocity_array_with_noise.shape[i]-1))
        v = velocity_array_with_noise[coord_indices2[0],
                                      coord_indices2[1],
                                      coord_indices2[2], :]

    return v

@jit
def RK4_step(y, t, dt, seed, velocity_array_with_noise):
    '''function representing RK4 step'''
    k1 = dt*f(y, t, seed, dt, velocity_array_with_noise)
    k2 = dt*f(y+k1/2, t+dt/2, seed, dt, velocity_array_with_noise)
    k3 = dt*f(y+k2/2, t+dt/2, seed, dt, velocity_array_with_noise)
    k4 = dt*f(y+k3, t+dt, seed, dt, velocity_array_with_noise)
    return y+1/6*(k1+2*k2+2*k3+k4)

def generate_path(dt, y0_and_seed_and_tlims):
    '''RK4 integration with a twist: Diffusion!'''
    y0 = y0_and_seed_and_tlims[:3]
    seed = int(np.floor(y0_and_seed_and_tlims[3]))
    t_lims = y0_and_seed_and_tlims[4:6].tolist()
    out_list = [y0]
    t_list = np.arange(*(t_lims+[dt]))
    y = y0
    D_sigma = np.sqrt(2*dt*diffusion_constant)
    #pylint: disable = unused-variable
    #coordinate_points_new, output_array = create_noise_field(seed)
    if seed >= 2**(noise_arrays_n+4):
        warn.warn('There are more points than can be supported by the'
                  'number of noise arrays and their transformations,'
                  'resulted in some points seeing the same velocity fields.',
                  RuntimeWarning)
    if use_static_arrays:
        velocity_array_with_noise = ((1-noise_only)*interp_velocity_array
                                     + noise_amplitude*load_noise_array(seed))
    else:
        coordinate_points_new, output_array = create_noise_field(seed)
        velocity_array_with_noise = ((1-noise_only)*interp_velocity_array
                                     + noise_amplitude*output_array)
    #print('Done with generating noise')
    for i, t in enumerate(t_list[1:]):
        np.random.seed(seed=((seed+round(i*1e5)) % 4294967295))
        if ((y[0]**2+y[1]**2) < (radius)**2
                and -1*height < y[2] < -1*liquid_level):
            y_old = y
            y = RK4_step(y, t, dt, seed, velocity_array_with_noise)\
                +np.random.normal(scale=np.array([D_sigma, D_sigma, D_sigma]))
            if any(np.isnan(y).flatten()):
                y = y_old
        out_list.append(y)
    return t_list, out_list

def point_cloud(initial_points, time, dt):
    '''Generate a point cloud given initial points,
    start and end time tuple, and timestep.
    Output format is a list of numpy arrays.
    '''
    map_list = []
    for i, point in enumerate(initial_points):
        map_list.append(np.array(point+[i]+time))
    map_f = partial(generate_path, dt)

    total = len(initial_points)
    print_progress(0, total)
    with Pool(threads) as p:
        output = []
        for i, thing in enumerate(p.imap(map_f, map_list)):
            print_progress(i+1, total)
            output.append(thing)
    return output

def point_cloud_tlist(initial_points, times, dt):
    '''Generate a point cloud given initial points,
    start and end time tuple, and timestep.
    Output format is a list of numpy arrays.
    This function differs from point_cloud in that it allows different
    start and end times for each point in the point_cloud.
    '''
    map_list = []
    for i, point in enumerate(initial_points):
        map_list.append(np.array(point+[i]+times[i]))
    map_f = partial(generate_path, dt)

    total = len(initial_points)
    print_progress(0, total)
    with Pool(threads) as p:
        output = []
        for i, thing in enumerate(p.imap(map_f, map_list)):
            output.append(thing)
    return output

def generate_np_array_list_from_points(point_list, dt):
    '''Converts a list of points into a large numpy array.'''
    number_of_points = len(point_list)
    max_t = 0
    for point in point_list:
        if point[0][-1]-point[0][0] > max_t:
            max_t = point[0][-1]-point[0][0]
#    pdb.set_trace()
    t_list = np.arange(0, int(max_t), dt)
    t_i_list = np.arange(0, int(max_t), int(dt))//int(dt)
    points_np_list = []
    for t_i in t_i_list:
        points_np_temp = np.empty((len(point_list[0][1][0]), number_of_points))
        for i in range(number_of_points):
            if len(point_list[i][1]) > t_i:
                points_np_temp[:, i] = point_list[i][1][t_i]
            else:
                points_np_temp[:, i] = point_list[i][1][-1]
        points_np_list.append(points_np_temp)
    return (t_list, np.array(points_np_list))

def generate_ellipsoid_matrices(points_at_t, sigma):
    '''A convenience wrapper over singular-value decomposition.'''
    mean = np.average(points_at_t, axis=1)
    n = points_at_t.shape[1]
    #print(n)
    #pylint: disable=unused-variable
    u, s, vh = np.linalg.svd(points_at_t.T-mean)
    s_mat = np.diag((n-1)/(s**2*sigma**2))
    #pdb.set_trace()
    return mean, s, vh, np.matmul(np.matmul(vh.T, s_mat), vh)

@njit
def volume(ellipsoid_mat):
    '''Gives volume of ellipsoid given matrix defining said ellipsoid.'''
    return 4/3*np.pi/np.sqrt(np.linalg.det(ellipsoid_mat))

def in_ellipsoid(mean, ellipsoid_matrix, point):
    '''Returns square of Mahalanobis distance.'''
    r = np.matmul(np.matmul((point-mean), ellipsoid_matrix), (point-mean).T)
    return r

def remove_wall_points(points_in):
    '''Remove points that are outside the TPC volume.'''
    points = points_in.copy()
    for i, point in enumerate(points):
        if ((point[1][-1][0]**2 + point[1][-1][1]**2 > (radius)**2)
                or (point[1][-1][2] < -1*height)
                or (point[1][-1][2] > -1*liquid_level)):
            points.pop(i)
    return points
