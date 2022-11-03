import SimpleITK as sitk
import numpy as np
import imageio
from pathlib import Path

inp_folder = Path('/export/scratch2/vladysla/Data/Simulation_paper_distr/Avocado/Training/s01_d06/segm')
im = imageio.imread(inp_folder / 'slice_{:06d}.tiff'.format(0))
data = np.zeros((380, *im.shape), dtype=np.uint8)

for i in range(380):
    im = imageio.imread(inp_folder / 'slice_{:06d}.tiff'.format(i))
    data[i,:] = im

random_itk_image = sitk.GetImageFromArray(data)
random_itk_image.SetSpacing([0.1, 0.1, 0.1]) # Each pixel is 1.1 x 0.98 mm^2
sitk.WriteImage(random_itk_image, './image.mhd')
