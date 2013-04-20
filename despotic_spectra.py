from despotic import cloud
import h5py
import numpy as np
import pylab as pl

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
expand = 1
vgrid = np.linspace(ppvcube.min(),ppvcube.max(),nelts)
vdata = ppvcube[:,y-expand:y+expand+1,x-expand:x+expand+1]
pdata = pppcube[:,y-expand:y+expand+1,x-expand:x+expand+1] * cloud_mean_density

vinds = np.empty(vdata.shape)
volume_spectra = np.empty(vgrid.shape + vdata.shape[1:])
dens_spectra = np.empty(vgrid.shape + vdata.shape[1:])
for jj in xrange(vdata.shape[1]):
    for kk in xrange(vdata.shape[2]):
vinds = np.array([np.digitize(vdata[:,jj,kk], vgrid) for jj in xrange(vdata.shape[1]) for kk in xrange(vdata.shape[2])])
volume_spectrum = np.bincount(vinds, minlength=nelts)
dens_spectrum = np.bincount(vinds, weights=pdata.flat, minlength=nelts)

pl.figure(3)
pl.clf()
pl.plot(vgrid,volume_spectrum,label='volume')
pl.plot(vgrid,dens_spectrum,label='density')
pl.legend(loc='best')

gmc = cloud(fileName='/Users/adam/repos/despotic/cloudfiles/MilkyWayGMC.desp')

gmc.sigmaNT = 1e5 # cm/s, instead of 2 as default
gmc.Tg = 20. # start at 20 instead of 15 K
gmc.Td = 20.

# add ortho-h2co
gmc.addEmitter('o-h2co', 1e-9)


tau11 = np.empty(pdata.shape)
tau22 = np.empty(pdata.shape)
tex11 = np.empty(pdata.shape)
tex22 = np.empty(pdata.shape)
tb11 = np.empty(pdata.shape)
tb22 = np.empty(pdata.shape)

print "Shape: ",tau11.shape

import progressbar
pb = progressbar.ProgressBar()

for ii in pb(xrange(tau11.size)):
    gmc.nH = pdata.flat[ii]
    gmc.colDen = gmc.nH * vox_length
    line = gmc.lineLum('o-h2co')
    ind = np.unravel_index(ii, tau11.shape)
    tau11[ind] = line[0]['tau']
    tau22[ind] = line[2]['tau']
    tex11[ind] = line[0]['Tex']
    tex22[ind] = line[2]['Tex']
    tb11[ind] = line[0]['intTB']
    tb22[ind] = line[2]['intTB']

props = {'tau11':tau11,
    'tau22':tau22,
    'tex11':tex11,
    'tex22':tex22,
    'tb11':tb11,
    'tb22':tb22,}

pl.figure()
spectra = {}
for ii,key in enumerate(props):
    speccube = np.empty(vgrid.shape+tau11.shape[1:])
    for jj,kk in np.nditer(np.indices(tau11.shape[1:]).tolist()):
        speccube[jj,kk] = np.bincount(vinds, weights=props[key][:,jj,kk],
                minlength=nelts)
    spectra[key] = speccube
    pl.subplot(2,3,ii)
    pl.plot(vgrid, spectra[key])
    pl.title(key)
