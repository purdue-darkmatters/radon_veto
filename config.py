import numpy as np

gridsize = [2, 2, 1]
limit_box = [(-50,50), (-50,50), (-100,0)]
steps = (int(np.ceil((limit_box[0][1]-limit_box[0][0])/(gridsize[0]))),int(np.ceil((limit_box[1][1]-limit_box[1][0])/(gridsize[1]))),int(np.ceil((limit_box[2][1]-limit_box[2][0])/(gridsize[2]))))

x_list = np.linspace(limit_box[0][0]+gridsize[0]/2, limit_box[0][1]-gridsize[0]/2, steps[0])
y_list = np.linspace(limit_box[1][0]+gridsize[1]/2, limit_box[1][1]-gridsize[1]/2, steps[1])
z_list = np.linspace(limit_box[2][0]+gridsize[2]/2, limit_box[2][1]-gridsize[2]/2, steps[2])


def coord_from_index(index, gridsize, limit_box):
    return (gridsize[0]*index[0]+gridsize[0]/2+limit_box[0][0],gridsize[1]*index[1]+gridsize[1]/2+limit_box[1][0],gridsize[2]*index[2]+gridsize[2]/2+limit_box[2][0])


def index_from_coord(coord, gridsize, limit_box):
    return (int(np.floor((coord[0]-limit_box[0][0])/gridsize[0])),int(np.floor((coord[1]-limit_box[1][0])/gridsize[1])),int(np.floor((coord[2]-limit_box[2][0])/gridsize[2])))


def index_from_coord_float(coord, gridsize, limit_box):
    return ((coord[0]-limit_box[0][0])/gridsize[0]-0.5,(coord[1]-limit_box[1][0])/gridsize[1]-0.5,(coord[2]-limit_box[2][0])/gridsize[2]-0.5)

threads=3
timestep = 5e7

posrec_sigma = [0.3,0.3,0.15]

radius = 47.9
height = 96.9

subd = 6 #subdivisions for interpolation and noise generation

diffusion_constant = 5.95e-15 #cm^s/ns

noise_arrays_n = 2 #2^n noise arrays
noise_amplitude = 1e-11

def interp_index_from_coord(coord):
    return (int(np.floor((coord[0]-limit_box[0][0])/(gridsize[0]/subd))), int(np.floor((coord[1]-limit_box[1][0])/(gridsize[0]/subd))), int(np.floor((coord[2]-limit_box[2][0])/(gridsize[0]/subd))))
