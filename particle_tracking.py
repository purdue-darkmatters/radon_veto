'''functions for point cloud propagation'''
from multiprocessing import Pool
from functools import partial
import warnings as warn

import numpy as np
from numba import jit, njit
import pandas as pd
from scipy import spatial
from sklearn.neighbors.kde import KernelDensity
from sklearn.cluster import DBSCAN

from radon_veto.noise_generation import *
from radon_veto.config import *
from radon_veto.convenience_functions import *
#import pdb

#To-do: make more functions run in no-python mode.

interp_velocity_array = np.load(array_filename)

if invert_velocities:
    interp_velocity_array = interp_velocity_array*(-1)

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
    '''RK4 integration with a twist: Diffusion!

    Note that as arange is used to determine the list of timesteps the final
    time is not included.'''
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
                  'resulting in some points seeing the same velocity fields.',
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
                warn.warn('Nan encountered: ' +
                          'y = {}'.format(y_old), RuntimeWarning)
                y = y_old
        out_list.append(y)
    return t_list, out_list

def point_cloud(initial_points, time, dt, progressbar=True):
    '''Generate a point cloud given initial points,
    start and end time tuple, and timestep.
    Output format is a list of numpy arrays.

    Note that as arange is used to determine the list of timesteps the final
    time is not included.
    '''
    map_list = []
    for i, point in enumerate(initial_points):
        map_list.append(np.array(point+[i]+time))
    map_f = partial(generate_path, dt)

    total = len(initial_points)
    if progressbar:
        print_progress(0, total)
    with Pool(threads) as p:
        output = []
        for i, thing in enumerate(p.imap(map_f, map_list)):
            if progressbar:
                print_progress(i+1, total)
            output.append(thing)
    return output

def point_cloud_tlist(initial_points, times, dt, progressbar=True):
    '''Generate a point cloud given initial points,
    start and end time tuple, and timestep.
    Output format is a list of numpy arrays.
    This function differs from point_cloud in that it allows different
    start and end times for each point in the point_cloud.
    See point_cloud for more.
    '''
    map_list = []
    for i, point in enumerate(initial_points):
        map_list.append(np.array(point+[i]+times[i]))
    map_f = partial(generate_path, dt)

    total = len(initial_points)
    if progressbar:
        print_progress(0, total)
    with Pool(threads) as p:
        output = []
        for i, thing in enumerate(p.imap(map_f, map_list)):
            if progressbar:
                print_progress(i+1, total)
            output.append(thing)
    return output

def generate_np_array_list_from_points(point_list):
    '''Converts a list of points into a large numpy array.'''
    number_of_points = len(point_list)
    max_t = 0
    max_i = 0
    for i, point in enumerate(point_list):
        if point[0][-1]-point[0][0] > max_t:
            max_t = point[0][-1]-point[0][0]
            max_i = i
#    pdb.set_trace()
    t_list = point_list[max_i][0]
    points_np_list = []
    for t_i, _ in enumerate(t_list):
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

def remove_wall_points(points_in, j=-1):
    '''Remove points that are outside the TPC volume.'''
    points = points_in.copy()
    pop_list = []
    for i, point in enumerate(points_in):
        if ((point[1][j][0]**2 + point[1][j][1]**2 > (radius)**2)
                or (point[1][j][2] < -1*height)
                or (point[1][j][2] > -1*liquid_level)):
            pop_list.append(i)
    for i in range(len(pop_list)):
        points.pop(pop_list[-1*(i+1)])
    return points

def remove_wall_points_pointcloud(pointcloud):
    '''same as remove_wall_points, but acts on a single 3D pointcloud
    instead of a 3D+time series.'''
    del_i = []
    for i, point in enumerate(pointcloud.T):
        if ((point[0]**2 + point[1]**2 > (radius)**2)
                or (point[2] < -1*height)
                or (point[2] > -1*liquid_level)):
            del_i.append(i)
    return np.delete(pointcloud, del_i, axis=1)

def min_vol_hull(points, fraction):
    '''Finds the approximate minimum volume convex hull that
    contains a greater fraction of points than specified in the input.
    point_cloud should be N by d, where d is the number of dimensions (3).'''

    N = points.shape[1]
    N_in = np.ceil(fraction*N).astype(int)
    mean, _, _, mat = generate_ellipsoid_matrices(points, 3)
    points = points.T
    while points.shape[0] > N_in:
        max_r = 0
        max_i = 0
        for i, point in enumerate(points):
            r = in_ellipsoid(mean, mat, point)
            if r > max_r:
                max_r = r
                max_i = i
        points = np.delete(points, max_i, axis=0)
    return spatial.ConvexHull(points)

def point_in_hull(hull, point):
    '''creates a new hull with point added,
    and checks to see if it has the same volume as the previous one.
    This is a dumb method but it works.'''

    new_point_cloud = np.concatenate((hull.points, [point]), 0)
    new_hull = spatial.ConvexHull(new_point_cloud)
    return bool(abs(hull.volume-new_hull.volume) < tol)

def fraction_of_points(t, ld):
    '''function for fraction of points to keep'''
    return np.exp(-3*t*ld)*0.8

def check_if_event_in_hull(points_np, start_time, prefix, halflife, row_le):
    '''check if dataframe row (series) of low energy events is in hull'''
    if invert_time:
        t_index = int((start_time
                       - row_le[1]['event_time'] + timestep/2)//timestep)
        fraction = fraction_of_points(-row_le[1]['event_time'] +
                                      start_time, np.log(2)/halflife)
    else:
        t_index = int((row_le[1]['event_time']
                       - start_time + timestep/2)//timestep)
        fraction = fraction_of_points(row_le[1]['event_time'] -
                                      start_time, np.log(2)/halflife)
    pointcloud = remove_wall_points_pointcloud(points_np[1][t_index])

    if pointcloud.shape[1]*fraction > 3:
        try:
            hull = min_vol_hull(pointcloud, fraction)
        except spatial.qhull.QhullError:
            warn.warn('QHull Error encountered', RuntimeWarning)
            return (row_le[1]['event_number'],
                    row_le[1]['run_number'],
                    False)
        except ValueError:
            warn.warn('Value Error encountered', RuntimeWarning)
            print(pointcloud)
            return (row_le[1]['event_number'],
                    row_le[1]['run_number'],
                    False)
        return (row_le[1]['event_number'],
                row_le[1]['run_number'],
                point_in_hull(hull,
                              [row_le[1][prefix + 'x_3d_nn'],
                               row_le[1][prefix + 'y_3d_nn'],
                               row_le[1][prefix + 'z_3d_nn']]))
    else:
        warn.warn('Not enough points for convex'
                  'hull after removal of wall points.', RuntimeWarning)
        return (row_le[1]['event_number'],
                row_le[1]['run_number'],
                False)

def check_if_events_in_hull(points, events_dataframe, start_time,
                            halflife, prefix=''):
    '''check if dataframe of low energy events is in hull'''
    output = {'event_number': [], 'run_number': [], 'in_veto_volume': [], }
    input_array = []
    points_np = generate_np_array_list_from_points(points)
    map_f = partial(check_if_event_in_hull, points_np, start_time,
                    prefix, halflife)
    for row_le in events_dataframe.iterrows():
        input_array.append(row_le)
    with Pool(threads) as p:
        for thing in enumerate(p.imap(map_f, input_array)):
            output['event_number'].append(thing[1][0])
            output['run_number'].append(thing[1][1])
            output['in_veto_volume'].append(thing[1][2])

    return output

def pb214_decay(t):
    '''exponential decay of pb214'''
    return (1/2)**(t/(1e9*26.8*60))

def kde_likelihood(data_arr_nowall):
    '''Add likelihood based on KDE to each data point'''
    kde_fit = KernelDensity(kernel='tophat',
                            bandwidth=kernel_radius).fit(data_arr_nowall)
    data_arr_scores = np.zeros([data_arr_nowall.shape[0]],
                               dtype=[('x', np.double),
                                      ('y', np.double),
                                      ('z', np.double),
                                      ('t', np.double),
                                      ('score', np.double)])
    data_arr_scores['score'] = kde_score(kde_fit, data_arr_nowall)
    data_arr_scores['x'] = data_arr_nowall[:, 0]
    data_arr_scores['y'] = data_arr_nowall[:, 1]
    data_arr_scores['z'] = data_arr_nowall[:, 2]
    data_arr_scores['t'] = data_arr_nowall[:, 3]
    data_arr_scores['score'] += np.log(pb214_decay(data_arr_scores['t']*2*timestep))
    data_arr_scores.sort(axis=0, order='score')
    return data_arr_scores

def data_arr_from_points(points):
    '''Convert list of points into array structure for clustering'''
    number_of_data_points = len(points)*len(points[0][0])
    data_arr = np.zeros((number_of_data_points, 4), dtype=np.double)
    for i, point in enumerate(points):
        rownum = (i*len(points[0][0]), i*len(points[0][0])+len(points[0][0]))
        data_arr[rownum[0]:rownum[1], :3] = np.array(point[1])
        data_arr[rownum[0]:rownum[1], 3] = point[0]/(2*timestep)
    return data_arr

def kde_score(kde_fit, data_arr_nowall):
    '''Get likelihood of every point using multiple cores/processors'''
    chunks = threads*3
    map_list = []
    chunksize = data_arr_nowall.shape[0]//(chunks-1)
    for i in range(chunks):
        map_list.append(data_arr_nowall[i*chunksize: (i+1)*chunksize])
    with Pool(threads) as p:
        output = []
        for i, thing in enumerate(p.map(kde_fit.score_samples, map_list)):
            output.append(thing)
    return np.concatenate(output)

@njit
def remove_wall_points_np(data_arr):
    '''remove wall points of array in clustering format'''
    data_arr_out = np.zeros_like(data_arr)
    i = 0
    for j in range(data_arr.shape[0]):
        row = data_arr[j, :]
        if (row[0]**2+row[1]**2 < radius**2 and
                row[2] > -height and
                row[2] < -liquid_level):
            data_arr_out[i] = row
            i += 1
    return data_arr_out[:i]

def check_if_events_in_cluster(points, events, event_time):
    '''check if a list of events are in the 4D cluster.'''
    data_arr = data_arr_from_points(points)
    data_arr_scores = kde_likelihood(remove_wall_points_np(data_arr))
    data_arr_selected = data_arr_scores[-len(data_arr_scores)//n_selection:]
    db = DBSCAN(eps=DBSCAN_radius,
                min_samples=DBSCAN_samples, )\
                .fit(pd.DataFrame(data_arr_selected).values[:, :4])
    data_arr_cluster = np.zeros(data_arr_selected.shape,
                                dtype=[('x', np.double),
                                       ('y', np.double),
                                       ('z', np.double),
                                       ('t', np.double),
                                       ('score', np.double),
                                       ('label', int)])
    data_arr_cluster['x'] = data_arr_selected['x']
    data_arr_cluster['y'] = data_arr_selected['y']
    data_arr_cluster['z'] = data_arr_selected['z']
    data_arr_cluster['t'] = data_arr_selected['t']
    data_arr_cluster['score'] = data_arr_selected['score']
    data_arr_cluster['label'] = db.labels_
    data_arr_df = pd.DataFrame(data_arr_cluster)
    data_wo_outliers = data_arr_df.query('label != -1').values[:, :4]
    selected_fit = KernelDensity(kernel='tophat',
                                 bandwidth=kernel_radius).fit(data_wo_outliers)
    output = {'event_number': [], 'run_number': [], 'in_veto_volume': [], }
    for row in events.iterrows():
        if not invert_time:
            t = (row[1].event_time - event_time)/(2*timestep)
        else:
            t = -(row[1].event_time - event_time)/(2*timestep)
        score = selected_fit.score([[row[1].x_3d_nn,
                                     row[1].y_3d_nn,
                                     row[1].z_3d_nn,
                                     t]])
        output['event_number'].append(row[1].event_number)
        output['run_number'].append(row[1].run_number)
        output['in_veto_volume'].append(not score == -np.inf)
    return output
