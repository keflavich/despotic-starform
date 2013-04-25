import os
import glob
import numpy as np
from agpy import readcol
#import scipy.stats as ss
import scipy.optimize as so

pdfs = {}

for root,dirs,filenames in os.walk('/Users/adam/work/h2co/simulations/ginsburg_pdfs'):
    if 'pdf_time0.txt' in filenames: 
        fn = 'pdf_time0.txt'
        name = root.split("/")[-1]
        data = np.loadtxt(root+"/"+fn,skiprows=11)
        pdfs[name] = data[:,2]

dens_s = data[:,0]

info = readcol('ginsburg_pdfs/models_pdfs.txt',asRecArray=True)

recentered_densities = {k:dens_s - (((dens_s)*(p)).sum()/(p).sum()) for k,p in pdfs.iteritems()}
scaled_densities = {k:dens_s+np.log(info['density'][info['model']==k]/(2.8*1.67e-24)) for k in pdfs if k in info.model}
mass_pdfs = {k:(np.exp(d)*pdfs[k])/(np.exp(d)*pdfs[k]).sum() for k,d in scaled_densities.iteritems()}
recentered_mass_densities = {k:dens_s - (((dens_s)*(p)).sum()/(p).sum()) for k,p in mass_pdfs.iteritems()}

import pylab as pl
pl.rc('font',size=20)

# default
# keys = pdfs.keys()

def gaussian(x,mu,sig,amp):
    return amp*np.exp(-(x-mu)**2/(2*sig**2))

nfig = 5
for ii,mach in enumerate(np.unique(info['RMS_Mach'])):

    keys = info['model'][(info['resolution']==512)*(info['RMS_Mach']==mach)]

    pl.figure(1+nfig*ii)
    pl.clf()
    pl.title('$\mathcal{M}=%i$' % mach)
    for k in keys:
        pl.semilogy(dens_s, pdfs[k], label=k)
        pl.xlabel("Overdensity $s=\\ln(\\rho/\\rho_0)$")
        pl.ylabel("Volumetric PDF $p(V)$")
    pl.axis([-5,10,1e-3,1])
    pl.legend(loc='best',prop={'size':18})

    pl.figure(4+nfig*ii)
    pl.clf()
    pl.title("Recentered Densities $\mathcal{M}=%i$" % mach)
    for k in keys:
        pl.semilogy(recentered_densities[k], pdfs[k], label=k)
        pl.xlabel("Overdensity $s=\\ln(\\rho/\\rho_0)$")
        pl.ylabel("Volumetric PDF $p(V)$")
    pl.axis([-10,10,1e-3,1])
    pl.legend(loc='best',prop={'size':18})

    pl.figure(2+nfig*ii)
    pl.clf()
    pl.title('$\mathcal{M}=%i$' % mach)
    for k in keys: #scaled_densities:
        pl.semilogy(scaled_densities[k], pdfs[k], label=k)
        pl.xlabel("Density ln($n$ cm$^{-3}$)")
        pl.ylabel("Volumetric PDF $p(V)$")
    pl.axis([-5,17,1e-3,1])
    pl.legend(loc='best',prop={'size':18})

    pl.figure(3+nfig*ii)
    pl.clf()
    pl.title('$\mathcal{M}=%i$' % mach)
    for k in keys: #scaled_densities:
        pl.semilogy(scaled_densities[k], mass_pdfs[k], label=k)
        pl.xlabel("Density ln($n$ cm$^{-3}$)")
        pl.ylabel("Mass PDF $p(M)$")
    pl.axis([-5,17,1e-3,1])
    pl.legend(loc='best',prop={'size':18})

    pl.figure(5+nfig*ii)
    pl.clf()
    ax = pl.subplot(111)
    pl.title("Recentered Mass Densities $\mathcal{M}=%i$" % mach)
    # Shink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    for k in keys:
        (center,width,amp),covmat = so.curve_fit(gaussian, recentered_mass_densities[k], mass_pdfs[k]) 
        p = pl.semilogy(recentered_mass_densities[k], mass_pdfs[k],
                label="%s: $\\mu=%0.1f$, $\\sigma=%0.1f$" % (k,center,width), alpha=0.5, linewidth=2)
        pl.semilogy(recentered_mass_densities[k], gaussian(recentered_mass_densities[k],center,width,amp), color=p[0].get_color(), linestyle='--')
        pl.xlabel("Overdensity $s=\\ln(\\rho/\\rho_0)$")
        pl.ylabel("Mass PDF $p(M)$")
    pl.axis([-10,10,1e-3,ax.get_ylim()[1]])
    ax.legend(loc='center left',prop={'size':14},bbox_to_anchor=[1.0,0.5])
    pl.savefig("federrath_pdfs_recentered_massweighted_fitted_mach%i.png" % mach)

    pl.show()
