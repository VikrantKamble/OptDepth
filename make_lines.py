import numpy as np
import fitsio

from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=20) 
rc('ytick', labelsize=20)
rc('figure', figsize=(19.8,14))


fname = "~/Work/tau/bigcomp.fits"
A = fitsio.read(fname,1)

# get line data file
FULL = np.loadtxt('lines_quasar.dat', delimiter=';' , dtype={'names' : ('line_name', 'lambda'), 'formats' : ('S30', 'S30')}).T
lines, lamlam = FULL['line_name'], FULL['lambda']

# Plot everything

plot(A['WAVE'], A['FLUX'], linewidth=2, color='red')

for i,j in enumerate(lines):
	temp = lamlam[i].split(',')
	if len(temp) == 1:
		foo = float(temp[0])
	else:
		myarr = np.array([float(x) for x in temp])
		foo = np.average(myarr)
	plot([foo, foo], [1,18.5], color='k', linewidth=0.6)
	text(foo, 19, r'$%s \ %s$' %(j, lamlam[i]), horizontalalignment='center', verticalalignment='bottom',rotation='vertical', fontsize=23)

xlim(1745, 4160)
ylim(0,28)
xlabel(r'$\lambda_{rf}$', fontsize=20)
tight_layout()
show()

savefig('lines_quasar_2.pdf')
