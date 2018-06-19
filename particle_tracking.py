#pylint: disable = E1129, W0614

import numpy as np
import scipy
from radon_veto.noise_generation import *
from radon_veto.config import *
from numba import jit,njit
from multiprocessing import Pool
from functools import partial
import scipy.interpolate, scipy.signal
import pdb

#unfortunately, numba speedup is minimal; it will likely stay this way until I can be bothered to write my own interpolation code.
#This is because numba is unable to enter no-python mode.
#That probably means it will never be fixed until someone else complains about the speed :)

interp_velocity_array = np.load('interp_velocity_array.npy')


@jit
def f(y, t, timestep, velocity_array_with_noise):
    if any(np.isnan(y).flatten()):
        return y
    coord_indices = interp_index_from_coord(y)
    coord_indices2 = []
    if (coord_indices < np.array(velocity_array_with_noise.shape)).all():
        v = velocity_array_with_noise[coord_indices[0],coord_indices[1],coord_indices[2],:]
    else:
        print('Warning: exceeded bounding box at {}'.format(y))
        for i,coord in enumerate(coord_indices):
            coord_indices2 = min(coord, velocity_array_with_noise.shape[i]-1)
        v = velocity_array_with_noise[coord_indices2[0],coord_indices2[1],coord_indices2[2],:]
    
    return v

@jit
def RK4_step(y, t, dt, velocity_array_with_noise):
    k1 = dt*f(y,t, dt, velocity_array_with_noise)
    k2 = dt*f(y+k1/2,t+dt/2, dt, velocity_array_with_noise)
    k3 = dt*f(y+k2/2,t+dt/2, dt, velocity_array_with_noise)
    k4 = dt*f(y+k3,t+dt, dt, velocity_array_with_noise)
    return(y+1/6*(k1+2*k2+2*k3+k4))

def generate_path(dt, y0_and_seed_and_tlims):
    '''RK4 integration with a twist: Diffusion!'''
    diffusion_constant = 1.7e-18 #units should be cm^2/ns.
    y0 = y0_and_seed_and_tlims[:3]
    seed = int(np.floor(y0_and_seed_and_tlims[3]))
    t_lims = y0_and_seed_and_tlims[4:6].tolist()
    y_list = [y0]
    t_list = np.arange(*(t_lims+[dt]))
    y = y0
    D_sigma = np.sqrt(2*dt*diffusion_constant)

    coordinate_points_new,output_array = create_noise_field(seed) #pylint: disable=W0612
    velocity_array_with_noise = interp_velocity_array + noise_amplitude*output_array
    #print('Done with generating noise')
    for i,t in enumerate(t_list[1:]):
        np.random.seed(seed= ((seed+i*1e4) % 4294967295))
        if (y[0]**2+y[1]**2)<2401 and -99<y[2]<1:
            y_old = y
            y = RK4_step(y,t,dt, velocity_array_with_noise)+np.random.normal(scale=np.array([D_sigma,D_sigma,D_sigma]))
            if any(np.isnan(y).flatten()):
                y=y_old
        y_list.append(y)
    return t_list,y_list

def point_cloud(initial_points, time, dt):
    map_list = []
    for i,point in enumerate(initial_points):
        map_list.append(np.array(point+[i]+time))
    map_f = partial(generate_path,dt)
    with Pool(threads) as p:
        output = []
        for thing in p.imap(map_f,map_list):
            output.append(thing)
    return output
    
def point_cloud_tlist(initial_points, times, dt):
    map_list = []
    for i,point in enumerate(initial_points):
        map_list.append(np.array(point+[i]+times[i]))
    map_f = partial(generate_path,dt)
    with Pool(threads) as p:
        output = []
        for thing in p.imap(map_f,map_list):
            output.append(thing)
    return output

def generate_np_array_list_from_points(point_list,dt):
    number_of_points = len(point_list)
    max_t = 0
    for point in point_list:
        if point[0][-1]-point[0][0]>max_t:
            max_t = point[0][-1]-point[0][0]
#    pdb.set_trace()
    t_list = np.arange(0,int(max_t),dt)
    t_i_list = np.arange(0,int(max_t),int(dt))//int(dt)
    points_np_list = []
    for t_i in t_i_list:
        points_np_temp = np.empty((len(point_list[0][1][0]),number_of_points))
        for i in range(number_of_points):
            if len(point_list[i][1])>t_i:
                points_np_temp[:,i] = point_list[i][1][t_i]
            else:
                points_np_temp[:,i] = point_list[i][1][-1]
        points_np_list.append(points_np_temp)
    return (t_list, np.array(points_np_list))

def generate_ellipsoid_matrices(points_at_t,sigma):
    mean = np.average(points_at_t,axis=1)
    n = points_at_t.shape[1]
    #print(n)
    u,s,vh = np.linalg.svd(points_at_t.T-mean) #pylint: disable=W0612
    s_mat = np.diag((n-1)/(s**2*sigma**2))
    #pdb.set_trace()
    return mean, s, vh, np.matmul(np.matmul(vh.T,s_mat),vh)

@njit
def volume(ellipsoid_mat):
    return 4/3*np.pi/np.sqrt(np.linalg.det(ellipsoid_mat))

def in_ellipsoid(mean,ellipsoid_matrix,point):
    r = np.matmul(np.matmul((point-mean),ellipsoid_matrix),(point-mean).T)
    return r

def remove_wall_points(points):
    for i,point in enumerate(points):
        if (point[1][-1][0]**2 + point[1][-1][1]**2 > radius**2) or (point[1][-1][2] < -height) or (point[1][-1][2] > 0.5):
            points.pop(i)
    return points
