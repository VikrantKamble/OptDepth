from ConfigParser import SafeConfigParser
import numpy as np

parser = SafeConfigParser()
parser.read('config.ini')

print parser.sections()
# define a helper function to recover all the options in a given section
def ConfigMap(section):
	dict1 = {}
	options = parser.options(section)
	for option in options:
		dict1[option] = parser.get(section, option)
		if not dict1[option]:
			print 'Got an empty string! Please check your .ini file'
			return 'FAIL'
			break
	return dict1

G = ConfigMap('GLOBAL')

# Define the rf-wavelength vector
COEFF0, COEFF1, NPIX = float(G['coeff0']), float(G['coeff1']), int(G['npix'])
wl = 10**(COEFF0 + np.arange(NPIX)*COEFF1)

# CATFILE := cat + DR_version + Z(type) + Z_min + RunType + .fits
cat_file = G['cat_file'] + '_' + G['dr_version'] +'_' + G['drq_z'] + '_' + str(G['zmin']) + '_' + str(G['run']) +'.fits'

drq_z = G['drq_z']

author = G['author']

