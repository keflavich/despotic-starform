from despotic import cloud
import h5py
import numpy as np
import pylab as pl
from despotify_cube import despotify
import multiprocessing
pool = multiprocessing.Pool(4)

# data from http://starformat.obspm.fr/starformat/project/TURB_BOX
with h5py.File('DF_hdf5_plt_cnt_0020_dens_downsampled','r') as ds:
    pppcube = np.array(ds['dens_downsampled'])
with h5py.File('DF_hdf5_plt_cnt_0020_velz_downsampled','r') as ds:
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
expand = 30
vgrid = np.linspace(ppvcube.min(),ppvcube.max(),nelts)
vdata = ppvcube[:,y-expand:y+expand+1,x-expand:x+expand+1]
pdata = pppcube[:,y-expand:y+expand+1,x-expand:x+expand+1] * cloud_mean_density

gmc = cloud(fileName='/Users/adam/repos/despotic/cloudfiles/MilkyWayGMC.desp')

gmc.sigmaNT = 1e5 # cm/s, instead of 2 as default
gmc.Tg = 20. # start at 20 instead of 15 K
gmc.Td = 20.

# add ortho-h2co
gmc.addEmitter('o-h2co', 1e-9)

spectra,props = despotify(pdata, vdata, vgrid, vox_length, cloud=gmc, nprocs=16)

pl.figure()
onedshape = vgrid.shape + (np.prod(spectra[spectra.keys()[0]].shape[1:]),)
for ii,key in enumerate(spectra):
    pl.subplot(2,3,ii+1)
    pl.plot(vgrid, spectra[key].reshape(onedshape), label=key)
    pl.title(key)

import pyfits
hdr = pyfits.Header()
hdr.update('CRPIX3', 1)
hdr.update('CRVAL3', vgrid[0])
hdr.update('CDELT3', vgrid[1]-vgrid[0])
for key in spectra:
    fitsfile = pyfits.PrimaryHDU(data=spectra[key], header=hdr)
    fitsfile.writeto('STARFORM_centralpixels_%s.fits' % key, clobber=True)
