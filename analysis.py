# Based on a script from Stas Bevc (C) 2011
# http://www.sicmm.org/~stas/rdf-script/rdf_from_pdb.py

import numpy
import ase
import sys

def rdf(images, nbins, side=None, output='./rdf.out'):
    """Calculate the radial distribution function of a given ase trajectory"""
    print('Calculating g(r)...')


    if not (isinstance(images, list) or isinstance(images, ase.atoms.Atoms)):
            raise TypeError('images must be a list or an ase.atoms.Atoms istance')

    if not type(images) is list:
        images = [images]
    maxbin = nbins # number of bins

    if side is None:
        side = images[0].get_cell()[0,0]
    
    sideh = side/2.0
    dr = float((sideh)/maxbin) # bin width
    # I allocate a temporary larger histogram to avoid indexing problems later
    hist = numpy.zeros((maxbin), dtype=int) 
    rdf = {}
    nstep = len(images)
    
    for image in images:
        if not all(image.get_pbc()):
            image.set_pbc(True)
            image.set_cell([side, side, side])

        # read atom coordinates from PDB
        positions = image.get_positions(wrap=True)
        
        # loop over particle pairs
        npart = len(image)
        print("looping over particles",npart)
        for i in range(npart):

            posdiff = positions[i] - positions[i+1:]
            posdiff[posdiff>sideh] -= side
            posdiff[posdiff<-sideh] += side
            rij_array = numpy.sum(numpy.abs(posdiff)**2,axis=-1)**(1./2)
            update, bins =  numpy.histogram(rij_array, bins=maxbin, range=(0.0,
                sideh), density=False)
            hist += update
    
    # normalize
    print "Normalizing ... "
    phi = npart/numpy.power(side, 3.0) # number density (N*V)
    norm = 2.0 * numpy.pi * dr * phi * nstep * npart
    
    for i in range(0, len(bins) - 1):
        dr = bins[i+1] - bins[i]
        rrr = bins[i] + dr/2.0
        val = hist[i]/ norm / ((rrr * rrr) + (dr * dr) / 12.0)
        rdf.update({rrr:val})
    
#    return rdf

#-------------------------------------------------------------------#

# write RDF into file
    #boxsize = 36.845
    #numbins = 384 # number of bins
    #cm = True # calculate RDF from center of mass of molecule
    #numAT = 4 # number of atoms in molecule
    #pdbsdir = "./pdbs/" # directory with PDB files
    outfile = output

    #rdf = makeRDF(pdbsdir, boxsize, numbins, cm, numAT)
    print "Writing output file ... " +outfile
    outfile = open(outfile, "w")
    for r in sorted(rdf.iterkeys()): # sort before writing into file
        outfile.write("%15.8g %15.8g\n"%(r, rdf[r]))
    outfile.close()

    return rdf

if __name__ == "__main__":
    rdf(sys.argv[1],int(sys.argv[2]),slicemin=int(sys.argv[3]),slicemax=int(sys.argv[4]),output=sys.argv[5])
