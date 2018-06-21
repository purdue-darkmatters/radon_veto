'''config file and convenience functions for radon_veto module'''
import numpy as np

gridsize = [2, 2, 1]
limit_box = [(-50, 50), (-50, 50), (-100, 0)]
steps = (int(np.ceil((limit_box[0][1]-limit_box[0][0])/(gridsize[0]))),
         int(np.ceil((limit_box[1][1]-limit_box[1][0])/(gridsize[1]))),
         int(np.ceil((limit_box[2][1]-limit_box[2][0])/(gridsize[2]))))

x_list = np.linspace(limit_box[0][0]+gridsize[0]/2, limit_box[0][1]-gridsize[0]/2, steps[0])
y_list = np.linspace(limit_box[1][0]+gridsize[1]/2, limit_box[1][1]-gridsize[1]/2, steps[1])
z_list = np.linspace(limit_box[2][0]+gridsize[2]/2, limit_box[2][1]-gridsize[2]/2, steps[2])


def coord_from_index(index, grids, lim_box):
    '''Get coordinate values from indices'''
    return (grids[0]*index[0]+grids[0]/2+lim_box[0][0],
            grids[1]*index[1]+grids[1]/2+lim_box[1][0],
            grids[2]*index[2]+grids[2]/2+lim_box[2][0])


def index_from_coord(coord, grids, lim_box):
    '''Get indices from coordinate values'''
    return (int(np.floor((coord[0]-lim_box[0][0])/grids[0])),
            int(np.floor((coord[1]-lim_box[1][0])/grids[1])),
            int(np.floor((coord[2]-lim_box[2][0])/grids[2])))


def index_from_coord_float(coord, grids, lim_box):
    '''Get indices from coordinate values without rounding to integers'''
    return ((coord[0]-lim_box[0][0])/grids[0]-0.5,
            (coord[1]-lim_box[1][0])/grids[1]-0.5,
            (coord[2]-lim_box[2][0])/grids[2]-0.5)


threads = 2
timestep = 5e7

posrec_sigma = [0.3, 0.3, 0.15]

#There seem to be events below 96.9 and with radius outside 47.9. This needs investigating.

#radius = 47.9
radius = 48.5
#height = 96.9
height = 97.5

subd = 6 #subdivisions for interpolation and noise generation

#diffusion_constant = 5.95e-15 #cm^s/ns
diffusion_constant = 1e-25 #cm^s/ns

noise_arrays_n = 2 #2^n noise arrays
noise_amplitude = 3e-11
#noise_amplitude = 0

use_static_arrays = True

def interp_index_from_coord(coord):
    '''Get interpolated index from coordinate values'''
    return (int(np.floor((coord[0]-limit_box[0][0])/(gridsize[0]/subd))),
            int(np.floor((coord[1]-limit_box[1][0])/(gridsize[0]/subd))),
            int(np.floor((coord[2]-limit_box[2][0])/(gridsize[0]/subd))))
