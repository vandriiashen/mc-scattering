# Monte-Carlo Simulation of a CT scan

Most parameters of the scan are defined in **run.py**: source properties, detector properties, sample geometry. The script creates multiple processes to run Gate (single-process application). The output of each process is transformed into projections. 4 projections are made for every scan: all X-ray photons, photons that experienced Rayleigh scattering, photons that experienced Compton scattering, and total scattering signal (Rayleigh+Compton).

macro/ subfolder contains scripts that control simulation. Files in this subfolder are edited by **run.py**.

**main.mac** can be executed separately for visualization.

# Gate Installation

First, install spack
```
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
source spack/share/spack/setup-env.sh
```

Some package use too much /tmp space, so
```
export TMPDIR=/export/scratch2/vladysla/spack/tmp
```
To install gate:
```
spack compiler find /opt/sw/gcc-11.3.0/bin/
spack install gate %gcc@11.3.0 ^nlohmann-json@3.10.5
```
