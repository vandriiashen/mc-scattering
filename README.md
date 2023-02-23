# Prepare for trouble and make it double

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
Some installation triggers problems with emacs on Fedora, so killall emacs is necessary
