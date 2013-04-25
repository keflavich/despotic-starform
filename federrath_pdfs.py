import os
import glob
import numpy as np
from agpy import readcol

pdfs = {}

for root,dirs,filenames in os.walk('/Users/adam/work/h2co/simulations/ginsburg_pdfs'):
    if 'pdf_time0.txt' in filenames: 
        fn = 'pdf_time0.txt'
        name = root.split("/")[-1]
        data = np.loadtxt(root+"/"+fn,skiprows=11)
        pdfs[name] = data[:,2]

dens_s = data[:,0]

info = readcol('ginsburg_pdfs/models_pdfs.txt',asRecArray=True)

scaled_pdfs = {k:dens_s+np.log(info['density'][info['model']==k]/(2.8*1.67e-24)) for k in pdfs if k in info.model}

import pylab as pl

pl.figure(1)
pl.clf()
for k in pdfs:
    pl.semilogy(dens_s, pdfs[k])

pl.figure(2)
pl.clf()
for k in scaled_pdfs:
    pl.semilogy(scaled_pdfs[k], pdfs[k])
