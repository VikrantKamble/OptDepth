import os
import ConfigParser
config = ConfigParser.ConfigParser()
file = open("config.ini", "w")

config.add_section("MakeCat")
config.set("MakeCat", 'drQCat', '/uufs/astro.utah.edu/common/home/u0882817/Work/Opt_old/data/DR12Q.fits')
config.set("MakeCat", 'drQ_DataExt', 1)
config.set("MakeCat", 'drQ_Z', 'Z_PCA')
config.set("MakeCat", 'DLA_file', '/uufs/astro.utah.edu/common/home/u0882817/Work/Opt_old/data/DLAs.dat')
config.set("MakeCat", 'spec_dir', '/uufs/chpc.utah.edu/common/home/sdss02/dr13/eboss/spectro/redux/v5_9_0/spectra/lite')
config.set("MakeCat", 'skyline_file','/uufs/astro.utah.edu/common/home/u0882817/Work/Opt_old/data/skymask_DR9.dat')
config.set("MakeCat", 'wav_range', [1000, 3000])
config.set("MakeCat", 'cat_file', os.environ['HOME'] + '/Work/tau/cat/cat_zpca_test.fits')
config.set("MakeCat", 'ParamsToWrite', ['PLATE','MJD','FIBERID','Z_PCA', 'Z_VI', 'ALPHA_NU','REWE_CIV', 'FWHM_CIV'])
config.set("MakeCat", 'Author', 'Vikrant Kamble')


config.write(file)
file.close()



