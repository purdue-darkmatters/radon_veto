#pylint: disable = E1129, W0614

import numpy as np
import scipy
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
def f(y, t, seed, timestep, velocity_array_with_noise):
    '''Random seed is needed to make sure paths generated consistent between calls;
    that is, if the integrator calls the same value of f(t,y) twice it will get the same result.
    This also means we can simply use the coords as a random seed, plus some salt based on the particle's position in point cloud.'''
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
def RK4_step(y, t, dt, seed, velocity_array_with_noise):
    k1 = dt*f(y,t,seed,dt, velocity_array_with_noise)
    k2 = dt*f(y+k1/2,t+dt/2,seed,dt, velocity_array_with_noise)
    k3 = dt*f(y+k2/2,t+dt/2,seed,dt, velocity_array_with_noise)
    k4 = dt*f(y+k3,t+dt,seed,dt, velocity_array_with_noise)
    return(y+1/6*(k1+2*k2+2*k3+k4))

@njit
def ramp(r):
    '''ramp function for dealing with edges when generating smooth noise function'''
    if r<1.0 and r>-1:
        return 3/8*r**5-10/8*r**4+15/8*r**3
    elif r<=-1:
        return float(0.0)
    elif r>=1.0:
        return 1.0

def filtered_vec(x,y,z, vec):
    '''Filtering out edges in noise function'''
    r = np.sqrt(x**2+y**2)
    if z<-3 and z>3-height:
        a = ramp((radius-r)/(3))
        normal_vector = -1*(np.array([x,y,0])/r)
        #print(x,y,normal_vector,a)
        return (1-a)*normal_vector*np.dot(vec,normal_vector) + a*vec
        #return a*normal_vector
    if z>=-3:
        if r < radius-3:
            a = ramp((z)/(-3))
            normal_vector = np.array([0,0,-1.0])
            return (1-a)*normal_vector*np.dot(vec,normal_vector) + a*vec
            #return a*normal_vector
        if r >= radius-3:
            opp = 1-z/(-3)
            adj = 1-(radius-r)/3
            a = ramp(1-np.sqrt(opp**2+adj**2))
            normal_vector =  (-1*np.array([x,y,0])/r*adj/np.sqrt(opp**2+adj**2)\
                    + np.array([0,0,-1.0])*opp/np.sqrt(opp**2+adj**2))
            return (1-a)*normal_vector*np.dot(vec,normal_vector) + a*vec
            #return a*normal_vector
    if z<=3-height:
        if r < radius-3:
            a = ramp((z+height)/(3))
            normal_vector = np.array([0,0,1.0])
            return (1-a)*normal_vector*np.dot(vec,normal_vector) + a*vec
            #return a*normal_vector
        if r >= radius-3:
            opp = 1-(z+height)/(3)
            adj = 1-(radius-r)/3
            a = ramp(1-np.sqrt(opp**2+adj**2))
            normal_vector = (-1*np.array([x,y,0])/r*adj/np.sqrt(opp**2+adj**2)\
                    + np.array([0,0,1.0])*opp/np.sqrt(opp**2+adj**2))
            return (1-a)*normal_vector*np.dot(vec,normal_vector) + a*vec
            #return a*normal_vector

def create_noise_field(seed):
    '''Here, we create the a new velocity field which combines divergence-free noise and the existing velocity field. It's basically copied from the divergence_free_noise notebook.
    Look at the notebook for detailed explanations and some maths.'''
    np.random.seed(seed=seed)

    x_list_interp = np.linspace(limit_box[0][0],limit_box[0][1],(-limit_box[0][0]+limit_box[0][1])/(gridsize[0]/(subd))+1)
    y_list_interp = np.linspace(limit_box[1][0],limit_box[1][1],(-limit_box[1][0]+limit_box[1][1])/(gridsize[0]/(subd))+1)
    z_list_interp = np.linspace(limit_box[2][0],limit_box[2][1],(-limit_box[2][0]+limit_box[2][1])/(gridsize[0]/(subd))+1)
    coordinate_points_interp = np.meshgrid(x_list_interp,y_list_interp,z_list_interp,indexing='ij')
    extended_noise_array = np.random.normal(0,1,[coordinate_points_interp[0].shape[0], coordinate_points_interp[0].shape[1], coordinate_points_interp[0].shape[2], 3])
    interp_noise_array = np.zeros(extended_noise_array.shape)
    kernel_1d = scipy.signal.gaussian((subd)*5, (subd)*0.8)
    kernel = np.zeros((kernel_1d.shape[0],kernel_1d.shape[0],kernel_1d.shape[0]))
    for i in range(0,kernel_1d.shape[0]):
        for j in range(0,kernel_1d.shape[0]):
            for k in range(0,kernel_1d.shape[0]):
                kernel[i,j,k] = kernel_1d[i]*kernel_1d[j]*kernel_1d[k]
    interp_noise_array[:,:,:,0] = scipy.signal.fftconvolve(extended_noise_array[:,:,:,0], kernel, mode='same')
    interp_noise_array[:,:,:,1] = scipy.signal.fftconvolve(extended_noise_array[:,:,:,1], kernel, mode='same')
    interp_noise_array[:,:,:,2] = scipy.signal.fftconvolve(extended_noise_array[:,:,:,2], kernel, mode='same')
    for i in range(0,len(x_list_interp)):
        for j in range(0,len(y_list_interp)):
            for k in range(0,len(z_list_interp)):
                r = np.sqrt(coordinate_points_interp[0][i,j,k]**2+coordinate_points_interp[1][i,j,k]**2)
                z = coordinate_points_interp[2][i,j,k]
                if radius+3 > r > radius-3 or z >-3 or z<3-height:
                    interp_noise_array[i,j,k] = filtered_vec(coordinate_points_interp[0][i,j,k], coordinate_points_interp[1][i,j,k], coordinate_points_interp[2][i,j,k], interp_noise_array[i,j,k])
    dx = [np.diff(interp_noise_array[:,:,:,0],axis=0),np.diff(interp_noise_array[:,:,:,0],axis=1),np.diff(interp_noise_array[:,:,:,0],axis=2)]
    dy = [np.diff(interp_noise_array[:,:,:,1],axis=0),np.diff(interp_noise_array[:,:,:,1],axis=1),np.diff(interp_noise_array[:,:,:,1],axis=2)]
    dz = [np.diff(interp_noise_array[:,:,:,2],axis=0),np.diff(interp_noise_array[:,:,:,2],axis=1),np.diff(interp_noise_array[:,:,:,2],axis=2)]
    curl_x = ((dz[1][:-1,:,:-1]+dz[1][1:,:,1:])-(dy[2][:-1,:-1,:]+dy[2][1:,1:,:]))/2
    curl_y = ((dx[2][:-1,:-1,:]+dx[2][1:,1:,:])-(dz[0][:,:-1,:-1]+dz[0][:,1:,1:]))/2
    curl_z = ((dy[0][:,:-1,:-1]+dy[0][:,1:,1:])-(dx[1][:-1,:,:-1]+dx[1][1:,:,1:]))/2
    
    coordinate_points_new = np.meshgrid((x_list_interp[:-1]+x_list_interp[1:])/2,(y_list_interp[:-1]+y_list_interp[1:])/2,(z_list_interp[:-1]+z_list_interp[1:])/2, indexing='ij')
    
    output_array = np.zeros(list(curl_x.shape)+[3])
    curl_arr = np.array([curl_x,curl_y,curl_z])
    for i in range(len(output_array[0,0,0,:])):
        output_array[:,:,:,i] = curl_arr[i,:,:,:]
    
    return coordinate_points_new,output_array

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
    velocity_array_with_noise = interp_velocity_array + 1e-11*output_array
    #print('Done with generating noise')
    for i,t in enumerate(t_list[1:]):
        np.random.seed(seed= ((seed+i) % 4294967295))
        if (y[0]**2+y[1]**2)<2401 and -99<y[2]<1:
            y_old = y
            y = RK4_step(y,t,dt,seed, velocity_array_with_noise)+np.random.normal(scale=np.array([D_sigma,D_sigma,D_sigma]))
            if any(np.isnan(y).flatten()):
                y=y_old
        y_list.append(y)
    return t_list,y_list

def point_cloud(initial_points, time, dt):
    map_list = []
    for i,point in enumerate(initial_points):
        map_list.append(np.array(point+[i*1e7]+time))
    map_f = partial(generate_path,dt)
    with Pool(threads) as p:
        output = []
        for thing in p.imap(map_f,map_list):
            output.append(thing)
    return output
    
def point_cloud_tlist(initial_points, times, dt):
    map_list = []
    for i,point in enumerate(initial_points):
        map_list.append(np.array(point+[i*1e7]+times[i]))
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
