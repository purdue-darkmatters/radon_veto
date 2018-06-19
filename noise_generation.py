#pylint: disable = E1129, W0614

from radon_veto.config import *
import numpy as np
import scipy
from numba import jit,njit

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

def generate_and_save_noise_arrays():
    '''Generates and saves 2^n noise fields,
    based on relevant config file parameter.
    Please do not exceed 13, as I provisioned 4 digit file names only.
    As the symmetry group for a square cuboid is 16,
    you should never need to exceed 13 anyway.'''
    for i in range(2**noise_arrays_n):
        #pylint: disable=unused-variable
        coordinate_points_new,output_array = create_noise_field(i) 
        np.save('noise_{:%04d}'.format(i), output_array)

def generate_permutations(i):
    '''return dihedral_n, vert_n, arr_n

    (D4 group element number, Z2 group number, noise array number)
    
    As there are 2^n noise arrays, we can determine which noise array to use
    using a binary string.
    This is also useful as the symmetry group is of order 2^4.
    
    The smallest n binary digits represent the noise array number,
    followed by 1 digit representing vertical rotation,
    and the largest 3 digits represent the dihedral symmetries.'''

    arr_n_mask = 2**(noise_arrays_n)-1 
    #the binary representation of this is 1111... (n digits)
    arr_n = i & arr_n_mask # first n binary digits represent noise array number

    vert_n = (i // 2**(noise_arrays_n)) & 1 # next digit for vertical rotation
    dihedral_n = (i // 2**(noise_arrays_n+1))

    return dihedral_n, vert_n, arr_n

@njit
def rotz(input_arr):
    '''Rotates vector field 90 degrees about the z axis'''
    output_arr = np.full([input_arr.shape[1],input_arr.shape[0],input_arr.shape[2]], np.nan)
    for i in range(input_arr.shape[0]):
        for j in range(input_arr.shape[1]):
            for k in range(input_arr.shape[2]):
                i_p = (input_arr.shape[1]-1)-j
                j_p = i
                output_arr[i_p,j_p,k,0] = -1*input_arr[i,j,k,1]
                output_arr[i_p,j_p,k,1] = input_arr[i,j,k,0]
                output_arr[i_p,j_p,k,2] = input_arr[i,j,k,2]
    
    return output_arr

@njit
def rotx(input_arr):
    '''Rotates vector field 90 degrees about the x axis'''
    output_arr = np.full([input_arr.shape[0],input_arr.shape[2],input_arr.shape[1]], np.nan)
    for i in range(input_arr.shape[0]):
        for j in range(input_arr.shape[1]):
            for k in range(input_arr.shape[2]):
                j_p = (input_arr.shape[2]-1)-k
                k_p = j
                output_arr[i,j_p,k_p,0] = input_arr[i,j,k,0]
                output_arr[i,j_p,k_p,1] = -1*input_arr[i,j,k,2]
                output_arr[i,j_p,k_p,2] = input_arr[i,j,k,1]
    return output_arr

def flipx(input_arr):
    '''mirrors vector flip in the x-axis.'''
    output_arr = np.full(input_arr.shape, np.nan)
    for i in range(input_arr.shape[0]):
        for j in range(input_arr.shape[1]):
            for k in range(input_arr.shape[2]):
                i_p = (input_arr.shape[0]-1)-i
                output_arr[i_p,j,k,0] = -1*input_arr[i,j,k,0]
                output_arr[i_p,j,k,1] = input_arr[i,j,k,1]
                output_arr[i_p,j,k,2] = input_arr[i,j,k,2]
    return output_arr

def compose(f,args,n):
    for i in range(n): #pylint: disable=unused-variable
        args = f(args)
    return args

def transform(noise_array, dihedral_n, vert_n):
    '''transforms noise_array and stuff.'''
    if vert_n==1:
        noise_array = rotx(noise_array)
    
    if dihedral_n & 1:
        noise_array = np.flip(noise_array, 0)
    
    if dihedral_n // 2 > 0:
        noise_array = compose(rotz,noise_array,dihedral_n // 2)

    return noise_array

