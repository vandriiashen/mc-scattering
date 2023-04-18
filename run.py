import subprocess
import os
import shutil
from pathlib import Path
import numpy as np
import imageio
from tqdm import tqdm

def scanner_component_macro(f, component_type, translation, scanner_size, pixel_size):
    '''Generates pieces of scanner macro. This function is called by write_scanner_mac.
    Scanner macro consists of several similar sections that are generated here.
    
    :param f: Text stream where gate commands will be written
    :type f: :class:`_io.TextIOWrapper`
    :param component_type: One of the following components ['CTscanner', 'module', 'cluster', 'pixel']
    :type component_type: :class:`str`
    :param translation: Position of the center of the scanner, 3 coordinates in mm
    :type translation: :class:`tuple`
    :param scanner_size: Size of the scanner, 3 dimensions in mm
    :type scanner_size: :class:`tuple`
    :param pixel_size: Pixel size in mm. Pixel are assumed to be square, depth of a pixel is given by scanner size
    :type pixel_size: :class:`float`
    '''
    materials = {'CTscanner' : 'Vacuum', 'module' : 'Vacuum', 'cluster' : 'Vacuum', 'pixel' : 'CsI'}
    mother_components = {'CTscanner' : 'world', 'module' : 'CTscanner', 'cluster' : 'module', 'pixel' : 'cluster'}
    mother_type = mother_components[component_type]
    f.write("/gate/{}/daughters/name {}\n".format(mother_type, component_type))
    f.write("/gate/{}/daughters/insert box\n".format(mother_type))
    if component_type == 'CTscanner':
        f.write("/gate/{}/placement/setTranslation {} {} {} mm\n".format(component_type, *translation))
    if component_type != 'pixel':
        f.write("/gate/{}/geometry/setXLength {} mm\n".format(component_type, scanner_size[0]))
        f.write("/gate/{}/geometry/setYLength {} mm\n".format(component_type, scanner_size[1]))
    else:
        f.write("/gate/{}/geometry/setXLength {} mm\n".format(component_type, pixel_size))
        f.write("/gate/{}/geometry/setYLength {} mm\n".format(component_type, pixel_size))
    f.write("/gate/{}/geometry/setZLength {} mm\n".format(component_type, scanner_size[2]))
    f.write("/gate/{}/setMaterial {}\n".format(component_type, materials[component_type]))
    if component_type != 'pixel':
        f.write("/gate/{}/vis/forceWireframe\n".format(component_type))
        f.write("/gate/{}/vis/setColor white\n".format(component_type))
    else:
        f.write("/gate/{}/vis/setColor red\n".format(component_type))
        f.write("/gate/{}/vis/setVisible 0\n".format(component_type))
        
def write_scanner_mac(translation = (0., 0., 100.), scanner_size=(75., 82.5, 10.), pixel_size=0.30):
    '''Generates scanner macro CTScanner.mac.
    Keep in mind that if the depth of the scanner is too small, a lot of high-energy photons would not be detected.
    We set a large depth to get better signal and avoid computing photons that will not be registered.
    
    :param translation: Position of the center of the scanner, 3 coordinates in mm
    :type translation: :class:`tuple`
    :param scanner_size: Size of the scanner, 3 dimensions in mm
    :type scanner_size: :class:`tuple`
    :param pixel_size: Pixel size in mm. Pixel are assumed to be square, depth of a pixel is given by scanner size
    :type pixel_size: :class:`float`
    '''
    num_pixels = [int(scanner_size[i] / pixel_size) for i in range(2)]
    assert scanner_size[0] == num_pixels[0] * pixel_size
    assert scanner_size[1] == num_pixels[1] * pixel_size
    
    with open('macro/CTScanner.mac', 'w') as f:
        for component in ('CTscanner', 'module', 'cluster', 'pixel'):
            scanner_component_macro(f, component, translation, scanner_size, pixel_size)
            f.write("\n")
            
        f.write("/gate/pixel/repeaters/insert cubicArray\n")
        f.write("/gate/pixel/cubicArray/setRepeatNumberX 250\n".format(num_pixels[0]))
        f.write("/gate/pixel/cubicArray/setRepeatNumberY 275\n".format(num_pixels[1]))
        f.write("/gate/pixel/cubicArray/setRepeatNumberZ   1\n")
        f.write("/gate/pixel/cubicArray/setRepeatVector {} {} 0.0 mm\n".format(pixel_size, pixel_size))
        f.write("/gate/pixel/cubicArray/autoCenter true\n")
        
        f.write("/gate/systems/CTscanner/module/attach module\n")
        f.write("/gate/systems/CTscanner/cluster_0/attach cluster\n")
        f.write("/gate/systems/CTscanner/pixel_0/attach pixel\n")
        f.write("/gate/pixel/attachCrystalSD\n")
        
def write_phantom_mac(mat = 'Aluminium', radius = 20., height = 50., cavity_size = 0.2, cavity_scale = [1., 1., 1.], cavity_loc = [0., 0., 0.]):
    '''Generates phantom macro phantom.mac.
    This phantom is a cylinder made of a single material with an ellipsoid cavity inside.
    Ellipsoid axes are parameterized as R*a, R*b, R*c, where a, b, c are in the neighbourhood of 1 and define how close the ellipsoid is to a sphere with radius R
    
    :param mat: Material of the phantom.
    :type mat: :class:`str`
    :param radius: Radius of the cylinder in mm.
    :type radius: :class:`float`
    :param height: Height of the cylinder in mm.
    :type height: :class:`float`
    :param cavity_size: Size of the cavity in mm (R from the description).
    :type cavity_size: :class:`float`
    :param cavity_scale: Ellipsoid axes parameters (a, b and c from the description).
    :type cavity_scale: :class:`tuple`
    :param cavity_loc: Location of the cavity, 3 coordinates in mm
    :type cavity_loc: :class:`tuple`
    '''
    with open('data/cylinder/phantom.mac', 'w') as f:
        f.write("/gate/world/daughters/name phantom\n")
        f.write("/gate/world/daughters/insert cylinder\n")
        f.write("/gate/phantom/geometry/setRmin 0 mm\n")
        f.write("/gate/phantom/geometry/setRmax {} mm\n".format(radius))
        f.write("/gate/phantom/geometry/setHeight {} mm\n".format(height))
        f.write("/gate/phantom/geometry/setPhiStart 0 deg\n")
        f.write("/gate/phantom/geometry/setDeltaPhi 360 deg\n")
        f.write("/gate/phantom/setMaterial {}\n".format(mat))
        
        f.write("/gate/phantom/daughters/name cavity\n")
        f.write("/gate/phantom/daughters/insert ellipsoid\n")
        f.write("/gate/cavity/geometry/setXLength {} mm\n".format(cavity_size * cavity_scale[0]))
        f.write("/gate/cavity/geometry/setYLength {} mm\n".format(cavity_size * cavity_scale[1]))
        f.write("/gate/cavity/geometry/setZLength {} mm\n".format(cavity_size * cavity_scale[2]))
        f.write("/gate/cavity/setMaterial Vacuum\n")
        
        f.write("/gate/phantom/placement/setTranslation  0. 0. 0. mm\n")
        f.write("/gate/phantom/placement/setRotationAxis 1 0 0\n")
        f.write("/gate/phantom/placement/setRotationAngle 90 deg\n")
        f.write("/gate/phantom/attachPhantomSD\n")
        f.write("/gate/cavity/placement/setTranslation {} {} {} mm\n".format(*cavity_loc))
        f.write("/gate/cavity/attachPhantomSD\n")

def write_source_mac(material_par, total_particles = 10**9, cone_angle = 10., source_loc = [0., 0., -200.]):
    '''Generates source macro source.mac.
    Cone beam with a small focal spot. Default cone angle covers most of the default scanner size (excluding corners of the image)
    
    :param material_par: Material-specific dictionary that contains path to the spectrum file and kV range.
    :type material_par: :class:`dict`
    :param total_particles: Total number of photons to generate for a single projection.
    :type total_particles: :class:`float`
    :param cone_angle: Cone angle, in degrees.
    :type cone_angle: :class:`float`
    :param source_loc: Location of the source, 3 coordinates in mm
    :type source_loc: :class:`float`
    '''
    spectrum = np.loadtxt(material_par['spectrum_fname'], delimiter=',', skiprows=1)
    with open('macro/source.mac', 'w') as f:
        f.write("/gate/source/addSource xraygun\n")
        f.write("/gate/source/verbose 0\n")
        f.write("/gate/source/xraygun/setActivity {:d}. becquerel\n".format(total_particles))
        f.write("/gate/source/xraygun/gps/verbose 0\n")
        f.write("/gate/source/xraygun/gps/particle gamma\n")
        f.write("/gate/source/xraygun/gps/energytype Arb\n")
        f.write("/gate/source/xraygun/gps/histname arb\n")
        f.write("/gate/source/xraygun/gps/emin {} keV\n".format(material_par['kv_start']))
        f.write("/gate/source/xraygun/gps/emax {} keV\n".format(material_par['kv_end']))
        for i in range(material_par['kv_start'],material_par['kv_end']):
            f.write("/gate/source/xraygun/gps/histpoint {:.3f} {:d}\n".format(0.001 * spectrum[i,0], int(spectrum[i,1])))
        f.write(" /gate/source/xraygun/gps/arbint Lin\n")
        f.write("/gate/source/xraygun/gps/type Plane\n")
        f.write("/gate/source/xraygun/gps/shape Rectangle\n")
        f.write("/gate/source/xraygun/gps/halfx 5. um\n")
        f.write("/gate/source/xraygun/gps/halfy 5. um\n")
        f.write("/gate/source/xraygun/gps/mintheta 0  deg\n")
        f.write("/gate/source/xraygun/gps/maxtheta  {} deg\n".format(cone_angle))
        #f.write("/gate/source/xraygun/gps/maxtheta  0.0001 deg\n")
        f.write("/gate/source/xraygun/gps/centre {} {} {} mm\n".format(*source_loc))
        f.write("/gate/source/xraygun/gps/angtype iso\n")
        f.write("/gate/source/list\n")
        
def check_coords(p_x, p_y, dim_x, dim_y):
    '''Checks if pixel positions stay inside the expected range
    
    :param p_x: Pixel position along x axis
    :type p_x: :class:`int`
    :param p_y: Pixel position along y axis
    :type p_y: :class:`int`
    :param dim_x: Number of pixels along x axis
    :type dim_x: :class:`int`
    :param dim_y: Number of pixels along y axis
    :type dim_y: :class:`int`
    '''
    if p_x >= dim_x:
        print("Overflow: px = {}".format(p_x))
        p_x = dim_x - 1
    if p_x < 0:
        print("Underflow: px = {}".format(p_x))
        p_x = 0
    if p_y >= dim_y:
        print("Overflow: py = {}".format(p_y))
        p_y = dim_y - 1
    if p_y < 0:
        print("Underflow: py = {}".format(p_y))
        p_y = 0
    return p_x, p_y
    
def analyze_numpy(out_folder, num, pixel_dim, scanner_xy):
    '''Reads the file with raw data and makes an image with 2D distributions of registered photons and only scatterd photons
    
    :param out_folder: Folder containing raw data files
    :type out_folder: :class:`pathlib.PosixPath`
    :param num: Number of the file to read (multiple processes produce multiple files)
    :type num: :class:`int`
    :param pixel_dim: Number of pixels along x and y axes.
    :type pixel_dim: :class:`Tuple`
    :param scanner_xy: Width and height of the scanner in mm
    :type scanner_xy: :class:`Tuple`
    '''
    data = np.load(out_folder / "res{}.Singles.npy".format(num))
    
    dim_x = 250
    size_x = 75.
    c_x = 0.
    dim_y = 275
    size_y = 82.5
    c_y = 0.0
    proj = np.zeros((dim_y, dim_x), dtype=np.int32)
    proj_scat = np.zeros((dim_y, dim_x), dtype=np.int32)
    proj_r = np.zeros((dim_y, dim_x), dtype=np.int32)
    num_total = 0
    num_scat = 0
    
    # Physical position of every registered photon should be converted to a pixel number.
    # Image is created by summing all photons in every pixel
    print(data.dtype)
    p_x = np.floor((data['globalPosX'] - c_x + size_x/2) / size_x * dim_x)
    p_y = np.floor((data['globalPosY'] - c_y + size_y/2) / size_y * dim_y)
    p_id = p_y*dim_x + p_x
    p_id = p_id.astype(int)
    values, counts = np.unique(p_id, return_counts=True)
    for i in range(len(values)):
        y = values[i] // dim_x
        x = values[i] % dim_x
        x, y = check_coords(x, y, dim_x, dim_y)
        proj[y,x] = counts[i]
    num_total = proj.sum()
        
    # The same process repeats for photons that are marked with scattering flag
    values, counts = np.unique(p_id[data['comptonPhantom'] > 0], return_counts=True)
    for i in range(len(values)):
        y = values[i] // dim_x
        x = values[i] % dim_x
        x, y = check_coords(x, y, dim_x, dim_y)
        proj_scat[y,x] = counts[i]
    num_scat = proj_scat.sum()
    
    values, counts = np.unique(p_id[data['RayleighPhantom'] > 0], return_counts=True)
    for i in range(len(values)):
        y = values[i] // dim_x
        x = values[i] % dim_x
        x, y = check_coords(x, y, dim_x, dim_y)
        proj_r[y,x] = counts[i]
    num_r = proj_r.sum()
    
    # Raw data also have the number of scattering interactions every photon participated in.
    # Thus, we can divide a signal into primary (photons that never scattered) and a series of N scatterings.
    signal_terms = np.zeros((5,))
    signal_terms[0] = num_total - num_scat - num_r
    
    compt = data['comptonPhantom']
    compt = compt[compt > 0]
    values, counts = np.unique(compt, return_counts=True)
    for i in range(1,min(5, len(counts)+1)):
        signal_terms[i] += counts[i-1]
        
    ray = data['RayleighPhantom']
    ray = ray[ray > 0]
    values, counts = np.unique(ray, return_counts=True)
    for i in range(1,min(5, len(counts)+1)):
        signal_terms[i] += counts[i-1]
    
    print("Total particles = {}, scattered = {}, Rayleigh = {}".format(num_total, num_scat, num_r))
    print("Scattering fraction = {:0.3f}".format(num_scat / num_total))
    print("Rayleigh fraction = {:0.3f}".format(num_r / num_total))
    
    signal_terms /= num_total
    print("Signal series = ")
    print(",".join(["{:.2%}".format(val) for val in signal_terms]))
        
    imageio.imwrite(out_folder / "proj_{}.tiff".format(num), np.flip(proj, axis=0))
    imageio.imwrite(out_folder / "proj_scat_{}.tiff".format(num), np.flip(proj_scat, axis=0))
    imageio.imwrite(out_folder / "proj_r_{}.tiff".format(num), np.flip(proj_r, axis=0))
    
def run_simulation(out_folder, num_proc):
    '''Starts processes to perform simulations.
    Gate is a single-thread application, so we create multiple processes to split the computations and use multiple cores.
    
    :param out_folder: Folder to write raw data
    :type out_folder: :class:`pathlib.PosixPath`
    :param num_proc: Number of processes to create. gjs does not seem to work for small num_proc (<5).
    :type num_proc: :class:`int`
    '''
    my_env = os.environ.copy()
    my_env["GC_DOT_GATE_DIR"] = "."
    split_folder = Path(".Gate")
    shutil.rmtree(split_folder, ignore_errors=True)
    out_folder.mkdir(exist_ok=True)

    subprocess.Popen(["gjs", "-numberofsplits", str(num_proc), "-clusterplatform", "openmosix", "main.mac"], env=my_env)
    proc_it = 1
    procs = []
    proc_ids = []
    for i in range(num_proc):
        p = subprocess.Popen(["Gate", "-a", "[ResName,{}/res{}.npy]".format(out_folder.as_posix(), proc_it), ".Gate/main/main{:d}.mac".format(proc_it)])
        procs.append(p)
        proc_ids.append(proc_it)
        proc_it += 1
    for p in procs:
        p.wait()
        
    return proc_ids

def write_intermediary_images(out_folder, proc_ids, pixel_dim, scanner_xy):
    '''Makes projection images based on the data from every process.
    
    :param out_folder: Folder to write raw data
    :type out_folder: :class:`pathlib.PosixPath`
    :param proc_ids: List containing ids of processes. Numbers from this list will be used to read files with raw data.
    :type proc_ids: :class:`list`
    :param pixel_dim: Number of pixels along x and y axes.
    :type pixel_dim: :class:`Tuple`
    :param scanner_xy: Width and height of the scanner in mm
    :type scanner_xy: :class:`Tuple`
    '''
    for i in tqdm(proc_ids):
        analyze_numpy(out_folder, i, pixel_dim, scanner_xy)
            
def make_final_image(out_folder, proc_ids, pixel_dim):
    '''Makes projection images based on the data from all processes.
    Creates 4 files: all photons, photons that had Compton scattering, photons that had Rayleigh scattering, and scattered photons in general (Compton or Rayleigh)
    
    :param out_folder: Folder to write raw data
    :type out_folder: :class:`pathlib.PosixPath`
    :param proc_ids: List containing ids of processes. Numbers from this list will be used to read files with raw data.
    :type proc_ids: :class:`list`
    :param pixel_dim: Number of pixels along x and y axes.
    :type pixel_dim: :class:`Tuple`
    '''
    dim_x, dim_y = pixel_dim
    proj = np.zeros((dim_y, dim_x), dtype=np.int32)
    proj_scat = np.zeros((dim_y, dim_x), dtype=np.int32)
    proj_r = np.zeros((dim_y, dim_x), dtype=np.int32)
    for i in proc_ids:
        tmp = imageio.imread(out_folder / 'proj_{}.tiff'.format(i))
        tmp_scat = imageio.imread(out_folder / 'proj_scat_{}.tiff'.format(i))
        tmp_r = imageio.imread(out_folder / 'proj_r_{}.tiff'.format(i))
        proj += tmp
        proj_scat += tmp_scat
        proj_r += tmp_r
    imageio.imwrite(out_folder / "proj.tiff", proj)
    imageio.imwrite(out_folder / "proj_c.tiff", proj_scat)
    imageio.imwrite(out_folder / "proj_r.tiff", proj_r)
    imageio.imwrite(out_folder / "proj_scat.tiff", proj_scat+proj_r)

# Phantom materials and the corresponding source spectra
material_spec = {
    'Iron450': {
        'material' : 'Iron',
        'kv_start' : 10,
        'kv_end' : 450,
        'spectrum_fname' : 'data/spectra/spectrum_450kv.csv'
    },
    'Iron300': {
        'material' : 'Iron',
        'kv_start' : 10,
        'kv_end' : 300,
        'spectrum_fname' : 'data/spectra/spectrum_300kv.csv'
    },
    'Aluminium90': {
        'material' : 'Aluminium',
        'kv_start' : 10,
        'kv_end' : 90,
        'spectrum_fname' : 'data/spectra/spectrum_90kv.csv'
    },
    'Aluminium150': {
        'material' : 'Aluminium',
        'kv_start' : 10,
        'kv_end' : 150,
        'spectrum_fname' : 'data/spectra/spectrum_150kv.csv'
    },
    'Aluminium300': {
        'material' : 'Aluminium',
        'kv_start' : 10,
        'kv_end' : 300,
        'spectrum_fname' : 'data/spectra/spectrum_300kv.csv'
    },
    'PMMA90': {
        'material' : 'PMMA',
        'kv_start' : 10,
        'kv_end' : 90,
        'spectrum_fname' : 'data/spectra/spectrum_90kv.csv'
    },
    'PMMA150': {
        'material' : 'PMMA',
        'kv_start' : 10,
        'kv_end' : 150,
        'spectrum_fname' : 'data/spectra/spectrum_150kv.csv'
    },
    'PMMA40': {
        'material' : 'PMMA',
        'kv_start' : 10,
        'kv_end' : 40,
        'spectrum_fname' : 'data/spectra/spectrum_40kv.csv'
    }
}
    
if __name__ == "__main__":
    # Only compute samples in this range (this way you can stop and resume computations)
    start_sample = 0
    end_sample = 50
    # Create this number of processes to use multiple CPU cores
    num_proc = 60  
    # Generate volume properties using gen_volume_properties.py
    data_spec = np.loadtxt('data/data_spec_train.csv', delimiter=',')
    mat_par = material_spec['Aluminium150']
    total_particles = 10**9
    tmp_folder = Path('/export/scratch2/vladysla/GateSimOutput/')
    out_folder = Path('/export/scratch2/vladysla/al150_train_0_500/')
    
    # Default simulation settings
    scanner_loc = (0., 0., 100.)
    scanner_size = (75., 82.5, 10.)
    pixel_size = 0.30
    pixel_dim = (250, 275)
    
    # Smaller air gap
    #scanner_loc = (0., 0., 50.)
    #scanner_size = (62.5, 68.75, 10.)
    #pixel_size = 0.25
    #pixel_dim = (250, 275)
    
    out_folder.mkdir(exist_ok=True)
    (out_folder / 'proj').mkdir(exist_ok=True)
    (out_folder / 'compt').mkdir(exist_ok=True)
    (out_folder / 'rayleigh').mkdir(exist_ok=True)
    (out_folder / 'scat').mkdir(exist_ok=True)
    with open(out_folder / 'stats.csv', 'w') as f:
        f.write("proj_num,size,cyl_r,cyl_h,cav_r,cav_z,el_x,el_y,el_z\n")
    
    i = start_sample
    for comb in data_spec[start_sample:end_sample]:
        cav_sc = [0., 0., 0.]
        proj_num, size, cyl_r, cyl_h, cav_r, cav_z, cav_sc[0], cav_sc[1], cav_sc[2] = comb
        print(cyl_r, cyl_h, cav_z, cav_r, size, cav_sc)
        with open(out_folder / 'stats.csv', 'a') as f:
            f.write("{},{},{},{},{},{},{},{},{}\n".format(i, size, cyl_r, cyl_h, cav_r, cav_z, *cav_sc))
            
        write_scanner_mac(scanner_loc, scanner_size, pixel_size)
        write_phantom_mac(mat_par['material'], cyl_r, cyl_h, size, cavity_scale = cav_sc, cavity_loc = [cav_r, 0., cav_z])
        write_source_mac(mat_par, total_particles)
        
        proc_ids = run_simulation(tmp_folder, num_proc)
        write_intermediary_images(tmp_folder, proc_ids, pixel_dim, (scanner_size[0], scanner_size[1]))
        make_final_image(tmp_folder, proc_ids, pixel_dim)
        
        shutil.copy(tmp_folder / 'proj.tiff', out_folder / 'proj' / '{:04d}.tiff'.format(i))
        shutil.copy(tmp_folder / 'proj_c.tiff', out_folder / 'compt' / '{:04d}.tiff'.format(i))
        shutil.copy(tmp_folder / 'proj_r.tiff', out_folder / 'rayleigh' / '{:04d}.tiff'.format(i))
        shutil.copy(tmp_folder / 'proj_scat.tiff', out_folder / 'scat' / '{:04d}.tiff'.format(i))
        i += 1
