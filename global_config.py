#!/usr/bin/python
# Always run this script to set the global variables correctly

# ConfigMap available as a global fucntion
try:
	G = ConfigMap('GLOBAL')
except Exception:
	print "Please define the parser object"


# Define the rf-wavelength vector
COEFF0, COEFF1, NPIX = float(G['coeff0']), float(G['coeff1']), int(G['npix'])
wl = 10**(COEFF0 + np.arange(NPIX)*COEFF1)

zmin = float(G['zmin'])
run = int(G['run'])

# CATFILE := cat + DR_version + Z(type) + Z_min + RunType + .fits
cat_file = G['cat_file'] + '_' + G['dr_version'] +'_' + G['drq_z'] + '_' + str(zmin) + '_' + str(run) +'.fits'

drq_z = G['drq_z']

author = G['author']
