import numpy as np
from skimage import measure
from pathlib import Path
from stl import mesh
import imageio

def get_volume():
    inp_folder = Path('/export/scratch2/vladysla/Data/Simulation_paper_distr/Avocado/Training/s01_d09/segm')
    vol = np.zeros((380, 478, 478))
    for i in range(vol.shape[0]):
        vol[i] = imageio.imread(inp_folder / 'slice_{:06d}.tiff'.format(i))
    return vol

def create_mesh(vol, mat_num, step_size, fname='model.stl'):
    vol = (vol == mat_num)
    verts, faces, normals, values = measure.marching_cubes(vol, spacing=(0.228, 0.228, 0.228), step_size=step_size)
    model = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    
    for i, f in enumerate(faces):
        for j in range(3):
            model.vectors[i][j] = verts[f[j],:]
            
    volume, cog, inertia = model.get_mass_properties()
    print("Volume                                  = {0}".format(volume))
    print("Position of the center of gravity (COG) = {0}".format(cog))
            
    model.save(fname)
    return model
    
def show_mesh(model):
    from mpl_toolkits import mplot3d
    from matplotlib import pyplot

    figure = pyplot.figure()
    ax = figure.add_axes(mplot3d.Axes3D(figure, auto_add_to_figure=False))

    # Load the STL files and add the vectors to the plot
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(model.vectors))

    # Auto scale to the mesh size
    print(type(model.points))
    scale = model.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    pyplot.show()

if __name__ == "__main__":
    vol = get_volume()
    model = create_mesh(vol, 2, 5, 'meat.stl')
    model = create_mesh(vol, 3, 5, 'pit.stl')
    model = create_mesh(vol, 4, 5, 'air.stl')
    #show_mesh(model)
    
