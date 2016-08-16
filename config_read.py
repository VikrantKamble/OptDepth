from ConfigParser import SafeConfigParser
import sys

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

# SET UP GLOBAL VARIABLES CORRECTLY
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