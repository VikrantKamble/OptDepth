import config_read as cfg
import fitsio
import analyze
import plotting
import CompGen
import imp

imp.reload(CompGen)
imp.reload(plotting)
imp.reload(cfg)
imp.reload(analyze)

def do_all():
	global read_flag, A, spec, ivar, tb, sn
	try: 
		read_flag
	except NameError:
		print 'The fits file does not exists. Reading now'

		# Sets up the global parser object and some global variables
		A = fitsio.FITS(cfg.cat_file)

		spec, ivar, tb =  A[0].read(), A[1].read(), A[2].read()
		sn_ind =  np.where((cfg.wl > 1280) & (cfg.wl < 1480))[0]

		sn = np.sum(spec[:,sn_ind]*np.sqrt(ivar[:,sn_ind]), axis=1)/np.sum(ivar[:,sn_ind] > 0, axis=1)

		# BAD SPECTRA
		ivar[139526][:] = 0

		print 'Input catalog file reading finished!'

		read_flag = 1
	
	# GENERATE COMPOSITES
	comp_folder = CompGen.CompGen(spec, ivar, tb, sn)

	# PLOT COMPOSITES
	plotting.plotcomp(comp_folder)

	# # RUN OPTICAL DEPTH CODE ON COMPOSITES
	analyze.runall(comp_folder)

if __name__=="__main__":
	do_all()
