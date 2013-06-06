from despotic import cloud
import h5py
import numpy as np
#import pylab as pl
from despotify_cube import despotify
import time

import sys
nprocs = int(sys.argv[1])
print "Running on %i processes" % nprocs
if len(sys.argv)>2:
    expand=int(sys.argv[2])
else:
    expand = 10
print "'Expand' is %i " % expand
if len(sys.argv)>3:
    runname=str(sys.argv[3])
else:
    runname='test'

path = os.getcwd()

# data from http://starformat.obspm.fr/starformat/project/TURB_BOX
with h5py.File(path+'/DF_hdf5_plt_cnt_0020_dens_downsampled','r') as ds:
    pppcube = np.array(ds['dens_downsampled'])
with h5py.File(path+'/DF_hdf5_plt_cnt_0020_velz_downsampled','r') as ds:
    ppvcube = np.array(ds['velz_downsampled'])

cloud_mass = 1e4 # msun
box_area = 10. # pc
vox_length = box_area * 3.08e18 / 256.
total_density = pppcube.sum()
# H2 cm^-3
cloud_mean_density = cloud_mass * 2e33/2.8/1.67e-24 / (total_density * vox_length**3)

# start with simple case
x,y = 128,128
nelts = 100
vgrid = np.linspace(ppvcube.min(),ppvcube.max(),nelts)
vdata = ppvcube[:,y-expand:y+expand+1,x-expand:x+expand+1]
pdata = pppcube[:,y-expand:y+expand+1,x-expand:x+expand+1] * cloud_mean_density

gmc = cloud(fileName=path+'/MilkyWayGMC_copy.desp')

gmc.sigmaNT = 1e5 # cm/s, instead of 2 as default
gmc.Tg = 20. # start at 20 instead of 15 K
gmc.Td = 20.

# add ortho-h2co
gmc.addEmitter('o-h2co', 1e-9)

t0 = time.time()
print "Beginning despotify"
results = despotify(pdata, vdata, vgrid, vox_length, cloud=gmc, nprocs=nprocs)
print "Completed despotify in %i seconds" % (time.time()-t0)
spectra,props,densspec = results

# pl.figure()
# onedshape = vgrid.shape + (np.prod(spectra[spectra.keys()[0]].shape[1:]),)
# for ii,key in enumerate(spectra):
#     pl.subplot(2,3,ii+1)
#     pl.plot(vgrid, spectra[key].reshape(onedshape), label=key)
#     pl.title(key)

try:
	import astropy.io.fits as pyfits
except ImportError:
	import pyfits
hdr = pyfits.Header()
hdr.update('CRPIX3', 1)
hdr.update('CRVAL3', vgrid[0])
hdr.update('CDELT3', vgrid[1]-vgrid[0])
for key in spectra:
    fitsfile = pyfits.PrimaryHDU(data=spectra[key], header=hdr)
    fitsfile.writeto(path+'/STARFORM_centralpixels_%sJanus%s.fits' % (key,runname), clobber=True)

fitsfile = pyfits.PrimaryHDU(data=densspec,header=hdr)
fitsfile.writeto(path+'/STARFORM_centralpixels_densityspectrumJanus%s.fits' % runname, clobber=True)
