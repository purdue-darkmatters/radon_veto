'''config file and convenience functions for radon_veto module'''
import numpy as np
from numba import njit

gridsize = np.array([2, 2, 1])
limit_box = np.array([[-50, 50], [-50, 50], [-100, 0]])
steps = (int(np.ceil((limit_box[0][1]-limit_box[0][0])/(gridsize[0]))),
         int(np.ceil((limit_box[1][1]-limit_box[1][0])/(gridsize[1]))),
         int(np.ceil((limit_box[2][1]-limit_box[2][0])/(gridsize[2]))))

x_list = np.linspace(limit_box[0][0]+gridsize[0]/2, limit_box[0][1]-gridsize[0]/2, steps[0])
y_list = np.linspace(limit_box[1][0]+gridsize[1]/2, limit_box[1][1]-gridsize[1]/2, steps[1])
z_list = np.linspace(limit_box[2][0]+gridsize[2]/2, limit_box[2][1]-gridsize[2]/2, steps[2])

@njit
def coord_from_index(index, grids, lim_box):
    '''Get coordinate values from indices'''
    return (grids[0]*index[0]+grids[0]/2+lim_box[0][0],
            grids[1]*index[1]+grids[1]/2+lim_box[1][0],
            grids[2]*index[2]+grids[2]/2+lim_box[2][0])

@njit
def index_from_coord(coord, grids, lim_box):
    '''Get indices from coordinate values'''
    return (int(np.floor((coord[0]-lim_box[0][0])/grids[0])),
            int(np.floor((coord[1]-lim_box[1][0])/grids[1])),
            int(np.floor((coord[2]-lim_box[2][0])/grids[2])))

@njit
def index_from_coord_float(coord, grids, lim_box):
    '''Get indices from coordinate values without rounding to integers'''
    return ((coord[0]-lim_box[0][0])/grids[0]-0.5,
            (coord[1]-lim_box[1][0])/grids[1]-0.5,
            (coord[2]-lim_box[2][0])/grids[2]-0.5)


threads = 20
timestep = 5e7

posrec_sigma = [0.3, 0.3, 0.17] # https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:jingqiang:sr1_wl_model_lows2 
#https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:analysis:sr1:positon_resolution

#There seem to be events below 96.9 and with radius outside 47.9. This needs investigating.

radius = 47.9
#radius = 48.5
height = 96.9
#height = 98
liquid_level = 0.25

subd = 6 #subdivisions for interpolation and noise generation

diffusion_constant = 5.95e-15 #cm^s/ns
#diffusion_constant = 1e-25 #cm^s/ns

noise_arrays_n = 4 #2^n noise arrays
noise_amplitude = 2.5e-11
#noise_amplitude = 0

tol = 0.01 #tolerance for convex hull equivalence checking.

use_static_arrays = True
noise_only = False
invert_velocities_config = False
# array_filename = 'interp_velocity_array.npy'
array_filename = '../Rn veto velocity fields/velocity_array_new_method_1000_5_filtered.npy'
invert_time_config = invert_velocities_config #This only inverts searching of events;
#velocity should also be inverted for this to make sense.

#clustering parameters
kernel_radius = 3
DBSCAN_radius = kernel_radius
DBSCAN_samples = 13
n_selection_po = 28 #1/n_selection points would be used
n_selection_bipo = 12 #1/n_selection points would be used
kde_rtol = 1e-6

#likelihood limit for kde and performance-related parameters
#likelihood_limit_po = 10.9
#likelihood_limit_bipo = 9.3
likelihood_limit_po = 11.2
likelihood_limit_bipo = 9.1
n_iter = 600
pointcloud_size = 192
#pointcloud_size = 20
corrected_likelihood_limit_po = likelihood_limit_po -\
                                np.log(n_iter*timestep)+np.log(pointcloud_size)
corrected_likelihood_limit_bipo = likelihood_limit_bipo -\
                                  np.log(n_iter*timestep) +\
                                  np.log(pointcloud_size)

@njit
def interp_index_from_coord(coord):
    '''Get interpolated index from coordinate values'''
    return (int(np.floor((coord[0]-limit_box[0][0])/(gridsize[0]/subd))),
            int(np.floor((coord[1]-limit_box[1][0])/(gridsize[0]/subd))),
            int(np.floor((coord[2]-limit_box[2][0])/(gridsize[0]/subd))))
