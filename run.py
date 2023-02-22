import subprocess
import os
import shutil
from pathlib import Path
import numpy as np
import imageio
from tqdm import tqdm

def scanner_component_macro(f, component_type, translation, scanner_size, pixel_size):
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
        #f.write("/gate/cavity/placement/setRotationAxis 1 0 0\n")
        #f.write("/gate/cavity/placement/setRotationAngle {} deg\n".format(cavity_rot[0]))
        #f.write("/gate/cavity/placement/setRotationAxis 0 1 0\n")
        #f.write("/gate/cavity/placement/setRotationAngle {} deg\n".format(cavity_rot[1]))
        #f.write("/gate/cavity/placement/setRotationAxis 0 0 1\n")
        #f.write("/gate/cavity/placement/setRotationAngle {} deg\n".format(cavity_rot[2]))
        f.write("/gate/cavity/attachPhantomSD\n")

def write_source_mac(material_par, total_particles = 10**9):
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
        f.write("/gate/source/xraygun/gps/maxtheta  10.0 deg\n")
        #f.write("/gate/source/xraygun/gps/maxtheta  0.0001 deg\n")
        f.write("/gate/source/xraygun/gps/centre 0. 0. -200. mm\n")
        f.write("/gate/source/xraygun/gps/angtype iso\n")
        f.write("/gate/source/list\n")
        
def check_coords(p_x, p_y, dim_x, dim_y):
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
    
def analyze_numpy(out_folder, num):
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
        p = subprocess.Popen(["Gate", "-a", "[ResName,res{}]".format(proc_it), ".Gate/main/main{:d}.mac".format(proc_it)])
        procs.append(p)
        proc_ids.append(proc_it)
        proc_it += 1
    for p in procs:
        p.wait()
        
    return proc_ids

def write_intermediary_images(out_folder, proc_ids):
    for i in tqdm(proc_ids):
        analyze_numpy(out_folder, i)
            
def make_final_image(out_folder, proc_ids):
    dim_x = 250
    dim_y = 275
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

material_spec = {
    'Iron450': {
        'material' : 'Iron',
        'kv_start' : 10,
        'kv_end' : 450,
        'spectrum_fname' : 'data/spectra/spectrum_450kv.csv'
    },
    'Aluminium150': {
        'material' : 'Aluminium',
        'kv_start' : 10,
        'kv_end' : 150,
        'spectrum_fname' : 'data/spectra/spectrum_150kv.csv'
    },
    'Plexiglass90': {
        'material' : 'Plexiglass',
        'kv_start' : 10,
        'kv_end' : 90,
        'spectrum_fname' : 'data/spectra/spectrum_90kv.csv'
    }
}
    
if __name__ == "__main__":
    start_sample = 57
    end_sample = 164
    num_proc = 10
    total_particles = 10**6
    data_spec = np.loadtxt('data/data_spec_test.csv', delimiter=',')
    tmp_folder = Path('/export/scratch1/vladysla/GateSimOutput/')
    out_folder = Path('/export/scratch1/vladysla/GateRes/')
    
    out_folder.mkdir(exist_ok=True)
    (out_folder / 'proj').mkdir(exist_ok=True)
    (out_folder / 'compt').mkdir(exist_ok=True)
    (out_folder / 'rayleigh').mkdir(exist_ok=True)
    (out_folder / 'scat').mkdir(exist_ok=True)
    with open(out_folder / 'stats.csv', 'w') as f:
        f.write("proj_num,size,cyl_r,cyl_h,cav_r,cav_z,el_x,el_y,el_z\n")
    
    i = start_sample
    mat_par = material_spec['Plexiglass90']
    for comb in data_spec[start_sample:end_sample]:
        cav_sc = [0., 0., 0.]
        proj_num, size, cyl_r, cyl_h, cav_r, cav_z, cav_sc[0], cav_sc[1], cav_sc[2] = comb
        print(cyl_r, cyl_h, cav_z, cav_r, size, cav_sc)
        with open(out_folder / 'stats.csv', 'a') as f:
            f.write("{},{},{},{},{},{},{},{},{}\n".format(i, size, cyl_r, cyl_h, cav_r, cav_z, *cav_sc))
            
        write_scanner_mac()
        write_phantom_mac(mat_par['material'], cyl_r, cyl_h, size, cavity_scale = cav_sc, cavity_loc = [cav_r, 0., cav_z])
        write_source_mac(mat_par, total_particles)
        
        proc_ids = run_simulation(tmp_folder, num_proc)
        write_intermediary_images(tmp_folder, proc_ids)
        make_final_image(tmp_folder, proc_ids)
        
        shutil.copy(tmp_folder / 'proj.tiff', out_folder / 'proj' / '{:04d}.tiff'.format(i))
        shutil.copy(tmp_folder / 'proj_c.tiff', out_folder / 'compt' / '{:04d}.tiff'.format(i))
        shutil.copy(tmp_folder / 'proj_r.tiff', out_folder / 'rayleigh' / '{:04d}.tiff'.format(i))
        shutil.copy(tmp_folder / 'proj_scat.tiff', out_folder / 'scat' / '{:04d}.tiff'.format(i))
        i += 1
