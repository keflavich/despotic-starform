import os
import itertools
#import glob
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

def twogaussian(x,mu,sig,amp,mu2,sig2,amp2):
    return amp*np.exp(-(x-mu)**2/(2*sig**2))+amp2*np.exp(-(x-mu2)**2/(2*sig2**2))

nfig = 5
for ii,mach in enumerate(np.unique(info['RMS_Mach'])):

    keys = info['model'][(info['resolution']==512)*(info['RMS_Mach']==mach)]
    keys = [k for k in keys if '1024' not in k and '5885' not in k]

    pl.figure(1+nfig*ii)
    pl.clf()
    ax = pl.subplot(111)
    # Shink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    pl.title('$\mathcal{M}=%i$' % mach)
    for k in keys:
        pl.semilogy(dens_s, pdfs[k], label=k)
        pl.xlabel("Overdensity $s=\\ln(\\rho/\\rho_0)$")
        pl.ylabel("Volumetric PDF $p(V)$")
    pl.axis([-10,6,1e-3,1])
    #pl.legend(loc='best',prop={'size':18})
    ax.legend(loc='center left',prop={'size':14},bbox_to_anchor=[1.0,0.5])
    pl.savefig("federrath_pdfs_volume_mach%i.png" % mach)

    pl.figure(4+nfig*ii)
    pl.clf()
    ax = pl.subplot(111)
    # Shink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    pl.title("Recentered Densities $\mathcal{M}=%i$" % mach)
    for k in keys:
        pl.semilogy(recentered_densities[k], pdfs[k], label=k)
        pl.xlabel("Overdensity $s=\\ln(\\rho/\\rho_0)$")
        pl.ylabel("Volumetric PDF $p(V)$")
    pl.axis([-10,10,1e-3,1])
    #pl.legend(loc='best',prop={'size':18})
    ax.legend(loc='center left',prop={'size':14},bbox_to_anchor=[1.0,0.5])

    pl.figure(2+nfig*ii)
    pl.clf()
    ax = pl.subplot(111)
    # Shink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    pl.title('$\mathcal{M}=%i$' % mach)
    for k in keys: #scaled_densities:
        pl.semilogy(scaled_densities[k], pdfs[k], label=k)
        pl.xlabel("Density ln($n$ cm$^{-3}$)")
        pl.ylabel("Volumetric PDF $p(V)$")
    pl.axis([-5,17,1e-3,1])
    #pl.legend(loc='best',prop={'size':18})
    ax.legend(loc='center left',prop={'size':14},bbox_to_anchor=[1.0,0.5])

    pl.figure(3+nfig*ii)
    pl.clf()
    ax = pl.subplot(111)
    # Shink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    pl.title('$\mathcal{M}=%i$' % mach)
    for k in keys: #scaled_densities:
        pl.semilogy(scaled_densities[k], mass_pdfs[k], label=k)
        pl.xlabel("Density ln($n$ cm$^{-3}$)")
        pl.ylabel("Mass PDF $p(M)$")
    pl.axis([-5,17,1e-3,1])
    #pl.legend(loc='best',prop={'size':18})
    ax.legend(loc='center left',prop={'size':14},bbox_to_anchor=[1.0,0.5])
    pl.savefig("federrath_pdfs_massweighted_mach%i.png" % mach)

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
    pl.axis([-7,7,1e-3,ax.get_ylim()[1]])
    ax.legend(loc='center left',prop={'size':14},bbox_to_anchor=[1.0,0.5])
    pl.savefig("federrath_pdfs_recentered_massweighted_fitted_mach%i.png" % mach)

    pl.show()

# fitting the Mach 10 PDF with a 2-peak gaussian
k='HDr512cM10s4774g'
mach = 10
pl.figure(26)
pl.clf()
ax = pl.subplot(111)
# Shink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
(center,width,amp),covmat = so.curve_fit(gaussian, recentered_mass_densities[k], mass_pdfs[k])
(center1,width1,amp1,center2,width2,amp2),covmat = so.curve_fit(twogaussian,
        recentered_mass_densities[k], mass_pdfs[k],
        p0=[center,width,amp,1.4,1,amp])
p = pl.semilogy(recentered_mass_densities[k], mass_pdfs[k],
        label="%s" % (k), alpha=0.5, linewidth=3)
pl.semilogy(recentered_mass_densities[k],
        gaussian(recentered_mass_densities[k],center,width,amp),
        color='g', linestyle='--', label="$\\mu=%0.1f$, $\\sigma=%0.1f$" % (center,width))
pl.semilogy(recentered_mass_densities[k],
        twogaussian(recentered_mass_densities[k],center1,width1,amp1,center2,width2,amp2),
        color='r', linestyle='-',linewidth=2, alpha=0.5)
pl.semilogy(recentered_mass_densities[k],
        gaussian(recentered_mass_densities[k],center1,width1,amp1),
        color='r', linestyle=':', label="$\\mu=%0.1f$, $\\sigma=%0.1f$" % (center1,width1))
pl.semilogy(recentered_mass_densities[k],
        gaussian(recentered_mass_densities[k],center2,width2,amp2),
        color='r', linestyle='--', label="$\\mu=%0.1f$, $\\sigma=%0.1f$" % (center2,width2))
pl.axis([-7,7,1e-3,0.1])
ax.legend(loc='center left',prop={'size':14},bbox_to_anchor=[1.0,0.5])
pl.xlabel("Overdensity $s=\\ln(\\rho/\\rho_0)$")
pl.ylabel("Mass PDF $p(M)$")
pl.title("Recentered Mass Density $\mathcal{M}=%i$" % mach)
pl.savefig("federrath_mach10_rescaled_massweighted_fitted.png")

pl.figure(27)
pl.clf()
xdens = np.logspace(0,8)
volume_mean_densities = {k: (np.exp(scaled_densities[k])*pdfs[k]).sum() for k in pdfs}
mass_mean_densities = {k: (np.exp(scaled_densities[k])*mass_pdfs[k]).sum() for k in mass_pdfs}
markers = {'s': 'o', 'c': 'x', 'm': '+'}
for k in scaled_densities:
    if '512' not in k:
        continue
    marker = markers[k[6]] if k[6] in 'scm' else markers[k[7]]
    pl.plot(volume_mean_densities[k],mass_mean_densities[k],marker=marker, markeredgewidth=2, alpha=0.5) # label=k,

import sys
sys.path.append("/Users/adam/work/turbulence/")
import hopkins_pdf

linestyles = itertools.cycle(['-','--','-.',':'])
for sigma in [1,2,3]:
    pl.loglog(xdens, xdens*np.exp(sigma**2), label='$\\sigma_s=%0.1f$' % sigma,
            linestyle=linestyles.next(), linewidth=2, alpha=0.5)
    T = hopkins_pdf.T_of_sigma(sigma,logform=True)
    pl.loglog(xdens, xdens*np.exp(sigma**2*(1+T)**3/(1+3*T+2*T**2)), label='$\\sigma_s=%0.1f$,$T=%0.2f$' % (sigma,T),
            linestyle=linestyles.next(), linewidth=2, alpha=0.5)

pl.errorbar([15],[2.2e4],xerr=np.array([[7,140]]).T,yerr=np.array([[12000,9400]]).T, label="G43.16-0.03", color='b', alpha=0.5, marker='o')
pl.errorbar([15],[2.4e4],xerr=np.array([[7,140]]).T,yerr=np.array([[8000,8600]]).T, label="G43.17+0.01", color='k', alpha=0.5, marker='o')
    
pl.loglog(xdens,xdens,'k--')
pl.axis([1,1e6,1,1e6])
pl.legend(loc='best',prop={'size':18})
pl.xlabel("Volume-weighted density $n_V(\\mathrm{H}_2)$ cm$^{-3}$")
pl.ylabel("Mass-weighted density $n_M(\\mathrm{H}_2)$ cm$^{-3}$")
pl.draw()
pl.savefig("VolumeVsMassWeighting.png")
