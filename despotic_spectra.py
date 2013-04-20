from despotic import cloud
import h5py
import numpy as np
import pylab as pl

with h5py.File('DF_hdf5_plt_cnt_0020_dens_downsampled','r') as ds:
    pppcube = np.array(ds['dens_downsampled'])
with h5py.File('DF_hdf5_plt_cnt_0020_velz_downsampled','r') as ds:
    ppvcube = np.array(ds['velz_downsampled'])

cloud_mean_density = 200 # H2 cm^-3

# start with simple case
x,y = 128,128
nelts = 100
expand = 0
vgrid = np.linspace(ppvcube.min(),ppvcube.max(),nelts)
vdata = ppvcube[:,y-expand:y+expand+1,x-expand:x+expand+1]
pdata = np.exp(pppcube[:,y-expand:y+expand+1,x-expand:x+expand+1]) * cloud_mean_density
vinds = np.digitize(vdata.flat, vgrid)
volume_spectrum = np.bincount(vinds, minlength=nelts)
dens_spectrum = np.bincount(vinds, weights=pdata.flat, minlength=nelts)

pl.figure(3)
pl.clf()
pl.plot(vgrid,volume_spectrum,label='volume')
pl.plot(vgrid,dens_spectrum,label='density')
pl.legend(loc='best')

gmc = cloud(fileName='/Users/adam/repos/despotic/cloudfiles/MilkyWayGMC.desp')

gmc.sigmaNT = 1e5 # cm/s, instead of 2 as default
gmc.Tg = 20 # start at 20 instead of 15 K
gmc.Td = 20

# add ortho-h2co
gmc.addEmitter('o-h2co', 1e-9)

gmc.colDen = 5e21 # use a moderately high column, but not as high as the default

tau11 = np.empty(pdata.size)
tau22 = np.empty(pdata.size)
tex11 = np.empty(pdata.size)
tex22 = np.empty(pdata.size)
tb11 = np.empty(pdata.size)
tb22 = np.empty(pdata.size)
for ii in xrange(tau11.size):
    gmc.nH = pdata[ii]
    line = gmc.lineLum('o-h2co')
    tau11[ii] = line[0]['tau']
    tau22[ii] = line[2]['tau']
    tex11[ii] = line[0]['Tex']
    tex22[ii] = line[2]['Tex']
    tb11[ii] = line[0]['intTB']
    tb22[ii] = line[2]['intTB']

props = {'tau11':tau11,
    'tau22':tau22,
    'tex11':tex11,
    'tex22':tex22,
    'tb11':tb11,
    'tb22':tb22,}

spectra = {}
for ii,key in enumerate(props):
    spectra[key] = np.bincount(vinds, weights=props[key], minlength=nelts)
    pl.subplot(2,3,ii)
    pl.plot(vgrid, spectra[key])
    pl.title(ii)
