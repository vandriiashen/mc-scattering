import subprocess
import os
import shutil
from pathlib import Path
import numpy as np
import imageio
from tqdm import tqdm
from multiprocessing import Process

def write_source_mac():
    total_particles = 10**8
    spectrum = np.loadtxt('data/spectrum_90kv.csv', delimiter=',', skiprows=1)
    with open('macro/source.mac', 'w') as f:
        f.write("/gate/source/addSource xraygun\n")
        f.write("/gate/source/verbose 0\n")
        f.write("/gate/source/xraygun/setActivity {:d}. becquerel\n".format(total_particles))
        f.write("/gate/source/xraygun/gps/verbose 0\n")
        f.write("/gate/source/xraygun/gps/particle gamma\n")
        f.write("/gate/source/xraygun/gps/energytype Arb\n")
        f.write("/gate/source/xraygun/gps/histname arb\n")
        f.write("/gate/source/xraygun/gps/emin 10.00 keV\n")
        f.write("/gate/source/xraygun/gps/emax 150.00 keV\n")
        for i in range(10,90):
            f.write("/gate/source/xraygun/gps/histpoint {:.3f} {:d}\n".format(0.001 * spectrum[i,0], int(spectrum[i,1])))
        f.write(" /gate/source/xraygun/gps/arbint Lin\n")
        f.write("/gate/source/xraygun/gps/type Plane\n")
        f.write("/gate/source/xraygun/gps/shape Rectangle\n")
        f.write("/gate/source/xraygun/gps/halfx 5. um\n")
        f.write("/gate/source/xraygun/gps/halfy 5. um\n")
        f.write("/gate/source/xraygun/gps/mintheta 0  deg\n")
        f.write("/gate/source/xraygun/gps/maxtheta  3.6 deg\n")
        #f.write("/gate/source/xraygun/gps/maxtheta  0.0001 deg\n")
        f.write("/gate/source/xraygun/gps/centre -1.146 0.0927 -837. mm\n")
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

def analyze_root(i):
    f = ROOT.TFile.Open('out/benchmarkCT{}.root'.format(i))
    singles = f.Get('Singles')
    
    dim_x = 478
    size_x = 143.017
    c_x = 0.357
    dim_y = 380
    size_y = 113.696
    c_y = -0.207
    proj = np.zeros((dim_y, dim_x), dtype=np.int32)
    proj_scat = np.zeros((dim_y, dim_x), dtype=np.int32)
    num_total = 0
    num_scat = 0
    
    for entry in singles:
        p_x = int((entry.globalPosX - c_x + size_x/2) / size_x * dim_x)
        p_y = int((entry.globalPosY - c_y + size_y/2) / size_y * dim_y)
        p_x, p_y = check_coords(p_x, p_y, dim_x, dim_y)
        proj[p_y,p_x] += 1
        num_total += 1
        if entry.comptonPhantom > 0:
            proj_scat[p_y,p_x] += 1
            num_scat += 1
    print("Total particles = {}, scattered = {}".format(num_total, num_scat))
    print("Scattering fraction = {:0.3f}".format(num_scat / num_total))
        
    np.save("out/proj{}.npy".format(i), np.flip(proj, axis=0))
    np.save("out/proj_scat{}.npy".format(i), np.flip(proj_scat, axis=0))
    
def analyze_numpy(out_folder, i):
    data = np.load(out_folder / "res{}.Singles.npy".format(i))
    
    dim_x = 478
    size_x = 143.017
    c_x = 0.357
    dim_y = 380
    size_y = 113.696
    c_y = -0.207
    proj = np.zeros((dim_y, dim_x), dtype=np.int32)
    proj_scat = np.zeros((dim_y, dim_x), dtype=np.int32)
    num_total = 0
    num_scat = 0
    
    for entry in tqdm(data):
        p_x = int((entry[1] - c_x + size_x/2) / size_x * dim_x)
        p_y = int((entry[2] - c_y + size_y/2) / size_y * dim_y)
        p_x, p_y = check_coords(p_x, p_y, dim_x, dim_y)
        proj[p_y,p_x] += 1
        num_total += 1
        #comptonPhantom check
        if entry[5] > 0:
            proj_scat[p_y,p_x] += 1
            num_scat += 1
    print("Total particles = {}, scattered = {}".format(num_total, num_scat))
    print("Scattering fraction = {:0.3f}".format(num_scat / num_total))
        
    imageio.imwrite(out_folder / "proj_{}.tiff".format(i), np.flip(proj, axis=0))
    imageio.imwrite(out_folder / "proj_scat_{}.tiff".format(i), np.flip(proj_scat, axis=0))
    
def analyze_numpy2(out_folder, num):
    data = np.load(out_folder / "res{}.Singles.npy".format(num))
    
    dim_x = 478
    size_x = 143.017
    c_x = 0.357
    dim_y = 380
    size_y = 113.696
    c_y = -0.207
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
    
def run_simulation(out_folder):
    
    my_env = os.environ.copy()
    my_env["GC_DOT_GATE_DIR"] = "."
    split_folder = Path(".Gate")
    shutil.rmtree(split_folder, ignore_errors=True)
    
    out_folder.mkdir(exist_ok=True)
    
    num_proc = 12

    subprocess.Popen(["gjs", "-numberofsplits", str(num_proc), "-clusterplatform", "openmosix", "main.mac"], env=my_env)
    proc_num = 1
    procs = []
    proc_ids = []
    for i in range(num_proc):
        p = subprocess.Popen(["Gate", "-a", "[ResName,res{}]".format(proc_num), ".Gate/main/main{:d}.mac".format(proc_num)])
        procs.append(p)
        proc_ids.append(proc_num)
        proc_num += 1
    for p in procs:
        p.wait()
        
    return proc_ids

def write_intermediary_images(out_folder, proc_ids):
    for i in tqdm(proc_ids):
        analyze_numpy2(out_folder, i)

def write_intermediary_images_parallel(out_folder, proc_ids):
    procs = []
    for i in proc_ids:
        p = Process(target=analyze_numpy2, args=(out_folder, i))
        procs.append(p)
        p.start()
        
    for p in procs:
        p.join()
            
def make_final_image(out_folder, proc_ids):
    dim_x = 478
    dim_y = 380
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
    
def make_final_dose_distr(out_folder, proc_ids):
    res = np.zeros((100, 100, 100))
    for i in proc_ids:
        tmp = imageio.imread(out_folder / 'dose' / 'dose{}-Dose.mhd'.format(i))
        res += tmp
    imageio.mimwrite(out_folder / 'dose' / "Dose.tiff", res)
        
def combine_results():
    num_proc = 1000
    root_files = ["out/benchmarkCT{:d}.root".format(i+1) for i in range(num_proc)]
    p = subprocess.Popen(["hadd", "-f", "out/benchmarkCT.root", *root_files])
    p.wait()
    print("Combining done")
    
if __name__ == "__main__":
    out_folder = Path('/export/scratch1/vladysla/GateSimOutput/')
    write_source_mac()
    proc_ids = run_simulation(out_folder)
    #proc_ids = range(1, 2)
    write_intermediary_images(out_folder, proc_ids)
    make_final_image(out_folder, proc_ids)
    #make_final_dose_distr(out_folder, proc_ids)
