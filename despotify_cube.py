import despotic
import numpy as np
import itertools
try:
    from parallel_map import parallel_map
    parallelOK = True
except ImportError:
    parallelOK = False
try:
    from progressbar import ProgressBar,Percentage,Bar
    from progressbar import AdaptiveETA as ETA
except ImportError:
    from progressbar import ProgressBar,Percentage,Bar
    from progressbar import ETA


def despotify(pcube, vcube, vgrid, voxel_size=3.08e18, species='o-h2co',
              cloud=None, cloudfile='MilkyWayGMC.desp', cloudfile_path=None,
              output_linenumbers=[0,2],
              output_properties=['tau','Tex','intTB'],
              nprocs=1):
    """
    Turn a simulated ppp cube into a ppv cube using despotic for the radiative
    transfer

    Note that it is "despot-ify", not "de-spotify".

    Parameters
    ----------
    pcube : np.ndarray
        3-dimensional array containing values with units of density in n(H2) cm^-3
    vcube : np.ndarray
        3-dimensional array containing Z-velocity values, i.e. the velocity
        should be in the direction of the 0'th axis (because python arrays are
        inverted).  Expected unit is km/s, but it doesn't matter as long as the
        velocity units match the vgrid units
    vgrid : np.ndarray
        1-dimensional array containing the output velocity grid.  Must have
        same units as vcube.
    voxel_size : float
        1-dimensional size of a voxel in cm.  Used to convert from density to
        column
    species : str
        A string identifying the LAMDA species name, e.g. 'o-h2co', 'co', etc.
    cloud : None or despotic.cloud
        Can pass in a despotic cloud instance that will be modified by the
        specified cube density.  Otherwise, will be read from file.
    cloudfile : str
        The filename specifying the default cloud file to use
    cloudfile_path : str or None
        If none, defaults to despotic.__path__/cloudfiles/
    output_linenumbers : iterable
        A list of integer indices for which line numbers should be output as
        cubes
    output_properties : iterable
        A list of strings identifying the line properties to output as cubes
    nprocs : int
        Number of processes.  If <=1, will not try to parallelize.  If >1, will
        use python's multiprocessing toolkit to try to parallelize the code
        execution.

    Returns
    -------
    A data cube of dimensions [velocity,position,position] for each line in
    output_linenumbers for each property in output_properties
    """

    if pcube.shape != vcube.shape:
        raise ValueError('Cube Size mismatch: {0},{1}'.format(str(pcube.shape),
                         str(vcube.shape)))
    if vgrid.ndim > 1:
        raise ValueError('Velocity grid must be 1-dimensional')

    imshape = pcube.shape[1:]
    # need +1 because bincount can put things at element 0 or element (last+1)
    nelts = vgrid.size+1
    outcubeshape = (nelts,) + imshape

    print "Shapes: imshape ",imshape," outcubeshape ",outcubeshape,' nelts ',nelts

    vinds = np.empty(vcube.shape, dtype='int64')
    dens_spectra = np.empty(outcubeshape)
    for jj,kk in np.ndindex(imshape):
        vinds[:,jj,kk] = np.digitize(vcube[:,jj,kk], vgrid)
        dens_spectra[:,jj,kk] = np.bincount(vinds[:,jj,kk],
                                            weights=pcube[:,jj,kk],
                                            minlength=nelts)

    print "Made dens_spectra"

    cloudfile_path = cloudfile_path or despotic.__path__[0]+"/cloudfiles/"

    if cloud is None:
        cloud = despotic.cloud(fileName="{0}/{1}".format(cloudfile_path,
                               cloudfile))

    # property cubes prior to gridding have same shape as input cubes
    # use dict() instead of {} for python2.6 compatibility
    prop_cubes = dict([
        ("{0}{1}".format(pr,ln), np.empty(pcube.shape))
        for ln,pr in itertools.product(output_linenumbers, output_properties)])

    print "made prob_cubes"

    pb = ProgressBar(widgets=[Percentage(), ETA(), Bar()],
                     maxval=pcube.size).start()

    if parallelOK and nprocs > 1:
	print "Running in parallel"
        # for parallelization
        def model_a_pixel(density, cloud=cloud):
            cloud.nH = density
            cloud.colDen = cloud.nH * voxel_size
            line = cloud.lineLum(species)
            results = [("{0}{1}".format(pr,ln),
                        line[ln][pr])
                       for ln,pr in itertools.product(output_linenumbers,
                                                      output_properties)]
            pb.update(pb.currval+1)
            return dict(results)

        # run the parallel modeling
        result = parallel_map(model_a_pixel, pcube.flat, nprocs)

        #
        pb.finish()

        # put the modelsinto their locations
        for ind,out in enumerate(result):
            zi,yi,xi = np.unravel_index(ind, pcube.shape)
            for ln,pr in itertools.product(output_linenumbers,
                                           output_properties):
                key = "{0}{1}".format(pr,ln)
                prop_cubes[key][zi,yi,xi] = out[key]

    else: 
	print "Running in serial"
        for (zi,yi,xi),nH in np.ndenumerate(pcube):
            #print "At pixel %i,%i,%i" % (zi,yi,xi)
            cloud.nH = pcube[zi,yi,xi]
            cloud.colDen = cloud.nH * voxel_size
            #print "Running lineLum on ",species
            line = cloud.lineLum(species) #,verbose=True,veryverbose=True) 
            #print "Finished lineLum"

            for ln,pr in itertools.product(output_linenumbers,
                                           output_properties):
                key = "{0}{1}".format(pr,ln)
                prop_cubes[key][zi,yi,xi] = line[ln][pr]

            pb.update(pb.currval+1)
        pb.finish()
        print "Finished cube loop"

    # spectral cubes have outcubeshape
    spectra_cubes = {}
    spectra_cubes = dict([
        ("{0}{1}".format(pr,ln), np.empty(outcubeshape))
        for ln,pr in itertools.product(output_linenumbers, output_properties)])

    for key in prop_cubes:
        for jj,kk in itertools.product(*map(xrange,imshape)):
            spectra_cubes[key][:,jj,kk] = \
                np.bincount(vinds[:,jj,kk],
                            weights=prop_cubes[key][:,jj,kk],
                            minlength=nelts)

    return spectra_cubes,prop_cubes,dens_spectra

