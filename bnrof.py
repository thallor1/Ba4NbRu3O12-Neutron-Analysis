import numpy as np 
from mantid.simpleapi import * 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import re
import CifFile
import lmfit
from lmfit import Model,Parameters
from mantid.geometry import CrystalStructure, ReflectionGenerator, ReflectionConditionFilter
import MDUtils as mdu
'''
Below are some of the functions used throughout the analysis of Ba4NbRu3O12 data. 
They are defined here so as to not clog up the notebooks. 
'''


def load_iexy(fnames):
	#Sequentially loads IEXY files from DAVE, appending data to the last one.
	# Takes either a str or a list of strs
	if type(fnames)==str:
		#single fname, simple.
		iexy = np.genfromtxt(fnames)
	elif type(fnames)==list:
		#multiple filenames
		iexy = np.empty((0,4))
		for i in range(len(fnames)):
			iexy = np.vstack((iexy,np.genfromtxt(fnames[i])))
	return iexy

class IEXY_data:
	'''
	Class for handling iexy data form DAVE
	'''
	def __init__(self,fnames=0,scale_factor=1,Ei=5.0,self_shield=1.0):
		if fnames:
			iexy = load_iexy(fnames)
			self.intensity = iexy[:,0]*scale_factor
			self.err = iexy[:,1]*scale_factor
			self.q = iexy[:,2]
			self.energies = iexy[:,3]
			self.Ei=Ei


	def delete_indices(self,indices):
		#Tool to mask pixels in IEXY. Useful for
		# Masking or stitching
		indices=np.unique(indices)
		self.intensity = np.delete(self.intensity,indices)
		self.err = np.delete(self.err,indices)
		self.q = np.delete(self.q,indices)
		self.energies = np.delete(self.energies,indices)

	def get_indices(self,indices):
		#gets data from specified indices of object
		intensity=self.intensity[indices]
		err = self.err[indices]
		q = self.q[indices]
		energies = self.energies[indices]
		return intensity,err,q,energies

	def scale_iexy(self,scale_factor):
		#Scales the data to a precomputed scale factor
		self.intensity = self.intensity*scale_factor
		self.err=self.err*scale_factor

	def sub_bkg(self,bkg_iexy,self_shield=1.0):
		#subtracts another iexy from this one using the nearest value
		# in the background IEXY
		# TODO- implement interpolation rather than neearest neighbor
		for i in range(len(self.intensity)):
			closest_arg = np.argmin(np.abs(bkg_iexy.q-self.q[i])+np.abs(bkg_iexy.energies-self.energies[i]))
			self.intensities[i]=self.intensitiy[i]-self_shield*bkg_iexy.intensity[closest_arg]
			self.err[i]=np.sqrt(self.err[i]**2 + (self_shield*bkg_iexy.err[closest_arg])**2 )
	def bose_subtract(self,highT_iexy,tlow,thigh):
		#Performs bose-einstein temperature subtraction using a highT iexy dataset
		self.intensity= self.intensity

	def normalize_to_bragg(self,ref_iexy,res_Q,res_E,ref_Q_res,ref_E_res,bragg_I,bragg_I_ref):
		#Normalizes one dataset to another using bragg peak intensities
		# Requires resolution in Q, E for both datasets
		# Assumes reference dataset is already normalized and that both have
		# been adjusted for energy-dependent transmission.
		# Should use something like DAVE to get the intensity of the peak normalized to the same monitor
		scale_factor = (bragg_I_ref*ref_E_res*ref_Q_res)/(bragg_I*E_res*Q_res)
		self.intensity*=scale_factor
		self.err*=scale_factor

	def take_q_cut(self,q_range,e_range,plot=True):
		#Simple function to take a cut of an iexy object
		#Q range in form of [min,max,num_bins]
		#E range in form of [min,max]
		#returns Q,I(Q),Err(Q)
		q_cut_i = np.intersect1d(np.where(self.q>=q_range[0]),np.where(self.q<=q_range[1]))
		e_slice_i = np.intersect1d(np.where(self.energies>=e_range[0]),np.where(self.energies<=e_range[1]))
		all_i = np.unique(np.append(q_cut_i,e_slice_i))
		slice_I, slice_err, slice_Q, slice_E = self.get_indices(all_i)
		bin_edges = np.linspace(q_range[0],q_range[1],q_range[2]+1)
		q_cut =[]
		q_cut_err=[]
		q_bin_centers=[]
		for i in range(len(bin_edges)-1):
			ind_in_bin = np.intersect1d(np.where(slice_Q>bin_edges[i]),np.where(slice_Q[i]<bin_edges[i]))
			cut_val = np.mean(slice_I[ind_in_bin])
			cut_err = np.sqrt(np.sum(slice_err[ind_in_bin]**2))/len(ind_in_bin)
			q_cut.append(cut_val)
			q_cut_err.append(cut_err)
			q_bin_centers.append(np.mean([bin_edges[i],bin_edges[i+1]]))
		if plot==True:
			plt.figure()
			plt.title('Q-cut from E=['+str(e_range[0])+','+str(e_range[1])+']')
			plt.xlabel('|Q|$\AA^{-1}$')
			plt.ylabel('Intensity (arb.)')
			plt.xlim(q_range[0],q_range[1])
			plt.ylim(np.min(q_cut),np.median(q_cut)*4.0)
			plt.show()
		return q_bin_centers,q_cut,q_cut_err

	def absorb_correct(self,rho_abs,vol,num_formula_units=False,d=1.0):
		#Corrects dataset for energy dependent absorption
		#d is path traveled in cm
		# rho is total absorption cross section of material
		# vol is the unit cell volume
		# num_formula_units is the number of formula units in the unit cell
		ref_lambda= 3956.0/2200.0
		lambda_i = 9.045 / np.sqrt(self.Ei)
		lambda0=ref_lambda
		energies_f = self.Ei - self.energies
		lambda_f = 9.045/np.sqrt(energies_f)
		if num_formula_units==False:
			print('WARNING: Number of formula units per unit cell must be specified for this calculation. ')
			return False
		for i in range(len(self.intensity)):
			ratio = (lambda_i + lambda_f[i])/(2.0*lambda0)
			transmission = np.exp(-0.5*d*rho_abs*ratio/vol)
			self.intensity[i]=self.intensity[i]/transmission
			self.err[i]=self.err[i]/transmission

	def take_cut(self,cut_axis='x',cut_params=[0,10,0.1],integrated_axis_extents=[0,1],plot=True):
		#Simple function to take a cut of an iexy object
		#  Define if the axis being cut is X or Y
		#  cut_params in format of [min,max,resolution]
		#  integrated_axis_extents defines integration region of other axis [min,max

		I_all = self.intensity 
		x_all = self.q 
		y_all = self.energies
		err_all = self.err
		if cut_axis=='x':
			integrated_i = np.intersect1d(np.where(y_all<=integrated_axis_extents[1]),np.where(y_all>=integrated_axis_extents[0]))
		elif cut_axis=='y':
			integrated_i = np.intersect1d(np.where(x_all<=integrated_axis_extents[1]),np.where(x_all>=integrated_axis_extents[0]))
		else:
			print('Invalid cut axis argument- only x or y permitted.')
			return 0
		I_all = I_all[integrated_i]
		x_all = x_all[integrated_i]
		y_all = y_all[integrated_i]
		err_all = err_all[integrated_i]
		#Integrate the relavant axis and errors
		if cut_axis=='x':
			#sort points into x_bins, then integrate Y
			x_bins = np.arange(cut_params[0],cut_params[1]+cut_params[2]/2.0,cut_params[2])
			x = x_bins[1:]-(x_bins[1]-x_bins[0])/2.0
			y = np.zeros(len(x))
			err = np.zeros(len(x))
			for i in range(len(x_bins)-1):
				ind = np.intersect1d(np.where(x_all>=x_bins[i]),np.where(x_all<=x_bins[i+1]))
				bin_errs = err_all[ind]
				bin_I = I_all[ind]
				if len(ind)>0:
					y[i] = np.average(I_all[ind],weights=1.0/bin_errs)
					err[i]=np.sqrt(np.sum(bin_errs**2))/len(bin_errs)
		elif cut_axis=='y':
			#sort points into x_bins, then integrate Y
			x_bins = np.arange(cut_params[0],cut_params[1]+cut_params[2]/2.0,cut_params[2])
			x = x_bins[1:]-(x_bins[1]-x_bins[0])/2.0
			y = np.zeros(len(x))
			err = np.zeros(len(x))
			for i in range(len(x_bins)-1):
				ind = np.intersect1d(np.where(y_all>=x_bins[i]),np.where(y_all<=x_bins[i+1]))
				bin_errs = err_all[ind]
				bin_I = I_all[ind]
				if len(ind)>0:
					y[i] = np.average(I_all[ind],weights=1.0/bin_errs)
					err[i]=np.sqrt(np.sum(bin_errs**2))/len(bin_errs)
		#If the user chose too fine a resolution there will be zero bins- remove these
		bad_bins=np.where(y==0)[0]
		x = np.array(x)
		y=np.array(y)
		err=np.array(err)
		x = np.delete(x,bad_bins)
		y = np.delete(y,bad_bins)
		err = np.delete(err,bad_bins)

		return x,y,err

	def rebin_iexy(self,x_bins,y_bins,return_new=True):
		#Given arrays for x and y bin edges, rebins the dataset appropriately.
		# If return new is set to true, returns a new object
		# If changed to false, edits the current object
		x_res = np.abs(x_bins[0]-x_bins[1])/2.0
		y_res = np.abs(y_bins[1]-y_bins[0])/2.0
		x_bin_centers = x_bins[1:]-x_res 
		y_bin_centers = y_bins[1:]-y_res 
		I_new=[]
		err_new =[]
		x_new =[]
		y_new =[]
		for i in range(len(x_bins)-1):
			for j in range(len(y_bins)-1):
				#find all intensities that lie in the bin
				xmin=x_bins[i]
				xmax=x_bins[i+1]
				ymin=y_bins[j]
				ymax=y_bins[j+1]
				x_ind = np.intersect1d(np.where(self.q >=xmin),np.where(self.q<xmax))
				y_ind = np.intersect1d(np.where(self.energies >=ymin),np.where(self.energies<ymax))
				ind = np.intersect1d(x_ind,y_ind)
				if len(ind)>0:
					I_arr = np.array(self.intensity[ind])
					err_arr = np.array(self.err[ind])
					zero_errs = np.where(err_arr==0)[0]
					err_arr[zero_errs]=1e8
					weights = 1.0/err_arr
					weights[zero_errs]=0.0 
					if np.sum(weights)==0:
						weights=np.ones(len(weights))
					I_bin=np.average(I_arr,weights=weights)
					err_bin = np.sqrt(np.sum(err_arr**2))/len(err_arr)
					x_new.append(x_bin_centers[i])
					y_new.append(y_bin_centers[j])
					I_new.append(I_bin)
					err_new.append(err_bin)
				else:
					x_new.append(x_bin_centers[i])
					y_new.append(y_bin_centers[j])
					I_new.append(np.nan)
					err_new.append(np.nan)
		if return_new==True:
			copy_obj = copy.deepcopy(self)
			copy_obj.q=np.array(x_new)
			copy_obj.energies = np.array(y_new)
			copy_obj.intensity = np.array(I_new)
			copy_obj.err = np.array(err_new)
			return copy_obj
		else:
			self.q=np.array(x_new) 
			self.energies = np.array(y_new)
			self.intensity = np.array(I_new)
			self.err=np.array(err_new)



	def transform2D(self,q_decimals=4,e_decimals=8):
		#Good for cleaning up dataset, bins data in Q and E
		# returns grid objects suitable for pcolormesh
		Q_arr = np.sort(np.around(np.unique(self.q),q_decimals))
		E_arr = np.sort(np.around(np.unique(self.energies),e_decimals))
		Q_grid,E_grid = np.meshgrid(Q_arr,E_arr)
		I_grid,err_grid = np.zeros(np.shape(Q_grid)),np.zeros(np.shape(Q_grid))
		for i in range(len(self.intensity)):
			q = self.q[i]
			e = self.energies[i]
			Int = self.intensity[i]
			err = self.err[i]
			q_index = np.argmin(np.abs(Q_arr-q))
			e_index = np.argmin(np.abs(E_arr-e))
			I_grid[e_index,q_index]=Int
			err_grid[e_index,q_index]=err
		return Q_grid,E_grid,I_grid,err_grid

	def plot_slice(self,axis_extents=False,vmin=0,vmax=1e4,cmap='rainbow',title='IEXY Slice plot',xlabel='|Q|',ylabel='E'):
		#Plots a slice of the dataset, returns the figure and ax 
		# axes extents in form of [xmin,xmax,ymin,ymax]
		if axis_extents==False:
			axis_extents=[-1e10,1e10,-1e10,1e10]
		fig,ax=plt.subplots(1,1)
		ax.set_title(title)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		X,Y,I,E = self.transform2D()
		#Z = scipy.interpolate.griddata((x_arr,y_arr),z_arr,(X,Y),method='linear')
		cax=fig.add_axes([1.0,0.1,0.05,0.75])
		mesh=ax.pcolormesh(X,Y,I,vmin=vmin,vmax=vmax,cmap=cmap)
		fig.colorbar(mesh,cax=cax,orientation='vertical')
		plt.show()
		return fig,ax

	def scale_to_FF(self,mag_ion):
		#Scales to the catalogued MANTID magneitc form factor
		#Assumes that the x is Q, y is E (i.e. powder)
		q_arr = self.q 
		q_min = np.min(q_arr)
		q_max = np.max(q_arr)
		q_all = np.linspace(q_min,q_max,1000)
		Q_ff, FF = THfuncs.get_MANTID_magFF(q_all,mag_ion)
		for i in range(len(self.intensity)):
			q_ind = np.argmin(np.abs(q_all-self.q[i]))
			FF_ind = FF[q_ind]
			self.intensity[i]*=FF_ind
			self.err[i]*=FF_ind 

	def save_IEXY(self,fname):
		#Saves manipulated data to a new file for later
		q_arr = np.array(self.q)
		e_arr = np.array(self.energies)
		I_arr = np.array(self.intensity)
		err_arr = np.array(self.err)
		mat = np.array([I_arr,err_arr,q_arr,e_arr])
		np.savetxt(fname,mat.T)

	def convert_to_MD(self):
		#Converts the IEXY to an MDhisto workspace for use with other algorithms.
		q_arr = np.array(self.q)
		e_arr = np.array(self.energies)
		I_arr = np.array(self.intensity)
		err_arr = np.array(self.err)
		err_arr[np.isnan(I_arr)]=0
		I_arr[np.isnan(I_arr)]=0
		#Need to do the sorting systematically. First get all elements in the lowest energy bin
		#First get all elements in the lowest E-bin
		new_I_arr=[]
		new_err_arr=[]
		for i in range(len(np.unique(np.around(e_arr,3)))):
			e_val=np.unique(np.around(e_arr,3))[i]
			e_bin_indices = np.where(np.around(e_arr,3)==e_val)[0]
			curr_q_arr = q_arr[e_bin_indices]
			curr_I_arr=I_arr[e_bin_indices]
			curr_err_arr=err_arr[e_bin_indices]
			q_sorted = np.argsort(curr_q_arr)
			new_I_arr.append(curr_I_arr[q_sorted])
			new_err_arr.append(curr_err_arr[q_sorted])
		I_arr=np.array(new_I_arr)
		err_arr=np.array(new_err_arr)
		extents_str = str(np.min(self.q))+','+str(np.max(self.q))+','+str(np.min(self.energies))\
						+','+str(np.max(self.energies))
		num_bin_str = str(len(np.unique(self.q)))+','+str(len(np.unique(self.energies)))
		
		out_ws = CreateMDHistoWorkspace(Dimensionality=2,Extents=extents_str,SignalInput=I_arr,ErrorInput=err_arr,NumberOfBins=num_bin_str,NumberOfEvents=np.ones(len(self.intensity)),Names='Dim1,Dim2',Units='MomentumTransfer,EnergyTransfer')
		return out_ws

 

def stitch_iexy(primary_iexy,secondary_iexy):
	#Takes two iexy objects with overlapping regions in coordinate space
	# and stiches them together, with the primary one preferred.
	max_e_primary = np.max(primary_iexy.energies)
	min_e_primary = np.min(primary_iexy.energies)
	min_q_primary = np.min(primary_iexy.q)
	max_q_primary = np.max(primary_iexy.q)
	#Remove elements that fall in these ranges from the secondary
	overlap_indcies =reduce(np.intersect1d, (np.where(secondary_iexy.q>=min_q_primary)[0],\
		np.where(secondary_iexy.q<=max_q_primary)[0],np.where(secondary_iexy.energies>=min_e_primary)[0]\
		,np.where(secondary_iexy.energies<=max_e_primary)[0]))
	secondary_iexy.intensities=np.delete(secondary_iexy.intensities,overlap_indcies)
	secondary_iexy.err=np.delete(secondary_iexy.err,overlap_indcies)


def fourier(q,FF,A,d):
	result= A*np.array(FF)*(np.sin(np.array(q)*d)/(np.array(q)*d))
	return result
def self_fourier(FF,C):
	result = FF*C
	return result

def gaussian_peak(x,sf,b,c):
	peak= np.array((sf/(c*np.sqrt(2.0*np.pi)))*(np.exp(-(x-b)**2 / (2.0*c**2))))
	#ensure that the integral remains equal to sf
	'''
	integral_now =np.sqrt(2.0)*sf*np.abs(c)*np.sqrt(np.pi)
	ratio = integral_now / sf
	peak=peak/ratio
	'''
	return peak

def incoherent_b(q_calc,b_inc,N_sample):
	#calculates incoherent powder scattering xc given a cross section and sample quantity
	return (N_sample/(4.0*np.pi))*np.ones(len(q_calc))*b_inc

def gen_ref_list_Material(material,material_th,maxQ=5.0,mind=0.2,maxd=30.0):
	#Given a material object, returns the following:
	# Q_reflection, HKL_reflection, Multiplicity of reflection, structure factor (barn / f.u.)
	generator = ReflectionGenerator(material)
	# Create list of unique reflections between 0.7 and 3.0 Angstrom
	hkls = generator.getUniqueHKLsUsingFilter(1, 30.0, ReflectionConditionFilter.StructureFactor)

	# Calculate d and F^2
	dValues = generator.getDValues(hkls)
	fSquared = generator.getFsSquared(hkls)

	pg = material.getSpaceGroup().getPointGroup()

	# Make list of tuples and sort by d-values, descending, include point group for multiplicity.
	reflections = sorted([(hkl, d, fsq, len(pg.getEquivalents(hkl))) for hkl, d, fsq in zip(hkls, dValues, fSquared)],
									key=lambda x: x[1], reverse=True)
	q_unique = []
	hkl_unique = []
	multiplicity = []
	sf_unique = []
	for reflection in reflections:
		hkl = reflection[0]
		d = reflection[1]
		f2 = reflection[2]#*0.01 # For conversion from fm^2 to barn
		M = reflection[3]
		Qtau = material_th.Qmag_HKL(hkl[0],hkl[1],hkl[2])
		if f2>0.01 and Qtau<maxQ:
			q_unique.append(Qtau)
			hkl_unique.append(hkl)
			multiplicity.append(M)
			sf_unique.append(f2*M*0.01)# For conversion from fm^2 to barn
	return np.array(q_unique),np.array(hkl_unique),np.array(multiplicity),np.array(sf_unique)

def integrate_elastic(test_md,deltaE,plot_results=False):
	#This function returns a cut of the integreated intensity over some energy window.
	dims = test_md.getNonIntegratedDimensions()
	q = mdu.dim2array(dims[0])
	e = mdu.dim2array(dims[1])
	I = np.copy(test_md.getSignalArray())
	Err = np.sqrt(np.copy(test_md.getErrorSquaredArray()))
	#Get the integrated elastic line
	#delE = [-3.1e-3,3.1e-3]
	E,Q = np.meshgrid(e,q)
	el_ind = np.array([np.abs(e)<=deltaE][0])
	Qel,Eel,Iel,Errel = Q[:,el_ind],E[:,el_ind],I[:,el_ind],Err[:,el_ind]
	el_e = e[el_ind]
	if plot_results==True:
		plt.figure()
		plt.pcolormesh(Qel,Eel,Iel,vmin=0,vmax=np.nanmean(Iel)*4.0,cmap='Spectral_r',shading='nearest')
		plt.xlabel(r"$Q$ ($\AA^{-1}$)")
		plt.ylabel(r"$\hbar\omega$ (eV)")
	int_I = []
	int_Q = []
	int_err = []
	for i,qpt in enumerate(q):
		Icut = Iel[i,:]
		Errcut = Errel[i,:]
		int_I_pt = np.trapz(x=el_e,y=Icut)
		int_err_pt = get_trapz_err(x=el_e,errs=Errcut)    
		int_I.append(int_I_pt)
		int_Q.append(qpt)
		int_err.append(int_err_pt)
	int_I,int_Q,int_err = np.array(int_I),np.array(int_Q),np.array(int_err)
	return int_Q,int_I,int_err

def get_md_Material_bragg_normalization_factors(test_md,Ei,energy_fwhm,material,material_th,aluminum_material=False,scale_guess=5.0e-2,\
												peak_width_guess=0.035,alum_scale_guess=0.0,sample_mass=1.0,allow_aluminum=True,\
												banned_Q_regimes=False,fit_method='powell',allow_q_shift=False,plot_results=True,\
											   form_unit_norm=True,plot_savedir=''):
	#This function calculates the overall normalization factor to scale 
	# the observed scattering on an inelastic spectrometer like SEQ to 
	# absolute units. 
	if banned_Q_regimes==False:
		banned_Q_regimes=[]
	if aluminum_material==False:
		Aluminum = Material('Aluminum.cif',suppress_print=True,nist_data='nist_scattering_table.txt')
	else:
		Aluminum = aluminum_material
	dims = test_md.getNonIntegratedDimensions()
	q = mdu.dim2array(dims[0])
	e = mdu.dim2array(dims[1])
	I = np.copy(test_md.getSignalArray())
	Err = np.sqrt(np.copy(test_md.getErrorSquaredArray()))
	prefactor_sample = (4.0*np.pi*(2.0*np.pi)**3)/(4.0*np.pi*material_th.cell_vol) # 4 pi comes from powder average
	#print(f"Prefactor_sample {prefactor_sample:.2f}")
	#print(f"1/ Prefactor = {1.0/prefactor_sample:.2f}")
	#Get the integrated elastic line
	#delE = [-3.1e-3,3.1e-3]
	E,Q = np.meshgrid(e,q)
	el_ind = np.array([np.abs(e)<=energy_fwhm*2.0][0])
	Qel,Eel,Iel,Errel = Q[:,el_ind],E[:,el_ind],I[:,el_ind],Err[:,el_ind]
	el_e = e[el_ind]
	int_I = []
	int_Q = []
	int_err = []
	for i,qpt in enumerate(q):
		Icut = Iel[i,:]
		Errcut = Errel[i,:]
		int_I_pt = np.trapz(x=el_e,y=Icut)
		int_err_pt = get_trapz_err(x=el_e,errs=Errcut) # should be per eV     
		int_I.append(int_I_pt)
		int_Q.append(qpt)
		int_err.append(int_err_pt)
	int_I,int_Q,int_err = np.array(int_I),np.array(int_Q),np.array(int_err)

	#Calculate the structure factors of all the relevant reflections
	q_unique,hkl_unique,mult,sfs = gen_ref_list_Material(material,material_th,maxQ=np.max(int_Q))

	#scale the SFs by the powder averaging effect
	sfs*=prefactor_sample/q_unique**2
	
	#Model the intensity as gaussian distributions with the same FWHM around each reflection
	def gauss_peak(q,sf,q0,sigma):
		peak = np.array((sf/(sigma*np.sqrt(2.0*np.pi)))*np.exp(-(q-q0)**2/(2.0*sigma**2)))
		return peak
	def calc_diff_pattern(q,sigma,I_scale,delQ=0):
		q_list = int_Q
		sf_list = sfs
		Iout = np.zeros(len(q))
		for i,q_pt in enumerate(q_unique):
			Ipeak = gauss_peak(q-delQ,sf_list[i],q_pt,sigma)/I_scale
			Iout+=Ipeak
		Iout[np.isnan(Iout)]=0
		return Iout
	#Could easily add in aluminum here. 
	peakmodel = Model(calc_diff_pattern,independent_vars=['q'])
	peakparams=peakmodel.make_params()
	peakparams.add('sigma',vary=True,value=0.01,min=0.007,max=0.035)
	peakparams.add('I_scale',vary=True,value=1.0e3,min=0.0,max=1e8)
	peakparams.add('delQ',value=0,vary=False)
	w=1.0/int_err
	#Iterate through the illegal Q-regimes
	for qwindow in banned_Q_regimes:
		if len(banned_Q_regimes)>0:
			#Check for case that no region is disallowed. 
			qmin = qwindow[0]
			qmax = qwindow[1]
			badi = np.intersect1d(np.where(int_Q<qmin)[0],np.where(int_Q>qmax)[0])
			w[badi]=0
			
	peakfit = peakmodel.fit(int_I,params=peakparams,q=int_Q,method='powell',nan_policy='omit',weights=w)
	datscale = peakfit.params['I_scale'].value
	#Per f.u. rather than unit cell:
	if form_unit_norm ==True:
		datscale/=material_th.formula_units
	if plot_results==True:
	#print(peakfit.fit_report())
		markersize=2
		fig,ax =plt.subplots(2,1,figsize=(3.54,4),sharex=True,height_ratios=[2,1])
		ax[0].errorbar(int_Q,int_I,yerr=int_err,color='k',marker='o',ms=markersize,ls=' ',zorder=3,mfc='w',mec='k',capsize=2)
		ax[0].bar(q_unique,1.5*np.pi*sfs/datscale,width=np.abs(np.max(q)-np.min(q[0]))/100,color='g',zorder=1)
		q_eval = np.linspace(0,3.5,1000)
		result_eval = peakmodel.eval(q=q_eval,params=peakfit.params)
		result_sub = peakmodel.eval(q=int_Q,params=peakfit.params)
		ax[0].plot(q_eval,result_eval,'r-',zorder=2)
		ax[1].errorbar(int_Q,int_I - result_sub,yerr=int_err,color='b',marker='o',ms=markersize)
		ax[1].plot(np.linspace(0,np.max(int_Q),1000),np.zeros(1000),'k--')
		ax[0].set_ylabel('$\int I(Q)Q^2d\Omega d\omega$')
		ax[1].set_ylabel('$(2\pi)^3|F_{HKL}(Q)|^2(4\pi V_0 Q^2)^-1$')
		ax[1].set_xlabel('$Q$ $\AA^{-1}$')
		print("Saving "+plot_savedir)
		fig.savefig(plot_savedir,bbox_inches='tight')
	outdict = {}
	outdict['result_scale']=datscale#*1e3 # norm to per eV
	return outdict
def get_trapz_err(x,errs,xlim=False):
	#Given the x values, errors, and limits of a trapzoidal integral returns the error bar of the 
	# result that would be given by np.trapz
	if xlim==False:
		xlim=[np.nanmin(x)-0.1,np.nanmax(x)+0.1]
	integral=0
	int_err=0
	good_i = np.intersect1d(np.where(x>=xlim[0])[0],np.where(x<=xlim[1])[0])
	x=x[good_i]
	errs=errs[good_i]
	for i in range(len(errs)-1):
		delX = np.abs(x[0]-x[i+1])
		term=delX * np.sqrt(errs[i]**2 + errs[i+1]**2)/2.0
		int_err+=term
	int_err = np.sqrt(np.sum(errs**2))
	return int_err


def bin_1D(x,y,yerr,bins,statistic='mean',fill=False):
	#Given specified bins, returns the binned coordinates and errors
	x,y,yerr=np.array(x),np.array(y),np.array(yerr)
	x_bin=[]
	y_bin=[]
	yerr_bin = []
	x=x[~np.isnan(y)]
	yerr = yerr[~np.isnan(y)]
	y=y[~np.isnan(y)]
	for i in range(len(bins)-1):
		val_ind = np.intersect1d(np.where(x>bins[i]),np.where(x<=bins[i+1]))

		if len(val_ind)>0 and fill==False:
			x_bin.append(np.mean([bins[i],bins[i+1]]))
			y_bin.append(np.average(np.array(y)[val_ind],weights=1.0/(np.array(yerr)[val_ind])))
			yerr_bin.append(np.sqrt(np.sum((np.array(yerr)[val_ind])**2))/len(val_ind))
		elif fill==True:
			x_bin.append(np.mean([bins[i],bins[i+1]]))
			yerr=np.ones(len(yerr))
			y_bin.append(np.average(np.array(y)[val_ind],weights=1.0/(np.array(yerr)[val_ind])))
			yerr_bin.append(np.sqrt(np.sum((np.array(yerr)[val_ind])**2))/len(val_ind))
	return np.array(x_bin),np.array(y_bin),np.array(yerr_bin)



def import_NIST_table(nist_file='nist_scattering_table.txt'):
	#Imports the nist table of scattering data as a python dictionary.
	# Columns represetn the following:
	#     Isotope 	conc 	Coh b 	Inc b 	Coh xs 	Inc xs 	Scatt xs 	Abs xs
	f = open(nist_file,'r')
	f_lines=f.readlines()
	f.close()
	lines={}
	column_labels=f_lines[0]
	for i in range(len(f_lines))[1:]:
		#Skipping the first line, append all of the results to our dictionary
		line = f_lines[i].strip('\r\n').split('\t')
		line_strip = [element.strip(' ') for element in line]
		element = line_strip[0]
		data = line_strip[1:] 
		lines[element]=data 
	return lines

def get_cif_dict(cif_file):
	'''
	A wrapper for the "ReadCif" untility from externaly python library.
	Given a cif, returns a dictionary with relevant attributes defined by the parameters in the line. 
	'''
	cif_import = CifFile.ReadCif(cif_file)
	if len(cif_import.dictionary)>1:
		#Multiple phases specified in cif file- get the one with data inside
		for i in range(len(cif_import.dictionary)):
			curr_key = cif_import.keys()[i]
			try:
				key = curr_key 
				cell_a = cif_import[curr_key]['_cell_length_a']
				#If this worked, this is good
				break
			except KeyError:
				cell_a = 0
	else:
		key = cif_import.keys()[0]
	return cif_import[key]
	
class Material:
	'''
	Class for calculation of scattering properties of materials from CIF files.
	Requires a cif file to initialize, as well as the NIST table of scattering data
	The following methods are available:
		1. Position generator for one unit cell
		2. Structure factor for particular [HKL]
		3. Absortption for given energy tranfer at particular Ei
		4. Contains basaic geometric information (the following, contained in lattice):
			a
			b
			c
			alpha
			beta
			gamma
			cell_vol
			space_group

	'''
	def __init__(self,cif_file,nist_data='nist_scattering_table.txt',b_arr=False,suppress_print=False):
		#Initializes the class
		if b_arr==False:
			scatt_dict = import_NIST_table(nist_data)
			self.b_arr=False
		else:
			scatt_dict=b_arr
			self.b_arr=True
			self.scatt_dict=b_arr
		#Make a dictionary of the unique atomic positions from the cif file, get their scattering lengths
		cif_f = open(cif_file,'r')
		f_lines = cif_f.readlines()
		cif_f.close()
		cif_obj = get_cif_dict(cif_file)

		#The cif obj contains all relevant information.
		
		#Collect the a, b, c, alpha, beta, gamma values. 
		a = float(cif_obj['_cell_length_a'].split('(')[0])
		b = float(cif_obj['_cell_length_b'].split('(')[0])
		c = float(cif_obj['_cell_length_c'].split('(')[0])
		alpha = float(cif_obj['_cell_angle_alpha'].split('(')[0])
		beta = float(cif_obj['_cell_angle_beta'].split('(')[0])
		gamma = float(cif_obj['_cell_angle_gamma'].split('(')[0])
		cell_vol = float(cif_obj['_cell_volume'].split('(')[0])
		#Generate reciprocal lattice 
		alpha_r = alpha*np.pi/180.0
		beta_r = beta*np.pi/180.0
		gamma_r = gamma*np.pi/180.0
		avec = np.array([a,0.0,0.0])
		bvec = np.array([b*np.cos(gamma_r),b*np.sin(gamma_r),0])
		cvec = np.array([c*np.cos(beta_r),c*(np.cos(alpha_r)-np.cos(beta_r)*np.cos(gamma_r))/(np.sin(gamma_r)),\
			c*np.sqrt(1.0-np.cos(beta_r)**2-((np.cos(alpha_r)-np.cos(beta_r)*np.cos(gamma_r))/np.sin(gamma_r))**2)])
		V_recip=np.dot(avec,np.cross(bvec,cvec))
		astar = np.cross(bvec,cvec)/V_recip
		bstar = np.cross(cvec,avec)/V_recip
		cstar = np.cross(avec,bvec)/V_recip

		#The parameter that defines the space group is often inconsistent. Find something that contains the string
		# '_space_group' but not 'xyz'
		space_group = 'Undefined'
		for i in range(len(cif_obj)):
			key_str = cif_obj.keys()[i]
			if (('_space_group' in key_str) or ('_space_group_name_h-m' in key_str)) and ('xyz' not in key_str) and ('number' not in key_str) and ('symop_id' not in key_str):
				#found the key to the space group in the dictionary.
				space_key = key_str 
				space_group = cif_obj[key_str]
				continue 
		self.avec = avec 
		self.bvec = bvec 
		self.cvec = cvec 
		self.u = astar
		self.v = bstar
		self.w = cstar
		self.a = a 
		self.b = b 
		self.c = c 
		self.alpha= alpha 
		self.beta = beta 
		self.gamma = gamma 
		self.cell_vol = cell_vol
		self.astar = np.linalg.norm(astar) 
		self.bstar = np.linalg.norm(bstar)
		self.cstar = np.linalg.norm(cstar) 
		self.cell_vol_recip = V_recip 
		self.space_group = space_group
		self.fname = cif_file 
		self.nist_file=nist_data
		self.scatt_dict = scatt_dict
		self.cif_dict = cif_obj
		# extracts some additional info in the cif file about the general unit cell
		f_lines = self.gen_flines()


		chem_sum = cif_obj['_chemical_formula_sum']
		try:
			formula_weight = float(cif_obj['_chemical_formula_weight'])
		except:
			#Need to calculate from chemsum, not implemented. yet..
			print("WARNING: Chemical weight not in cif file. Placeholder value used but should be updated manually using: \n Material.formula_weight=(val)")
			formula_weight = 100.0

		formula_units = int(cif_obj['_cell_formula_units_Z'])
		self.chem_sum = chem_sum 
		self.formula_weight=formula_weight
		self.formula_units = formula_units
		if suppress_print==False:
			print('\n######################\n')
			print('a = '+str(self.a)+' Ang')
			print('b = '+str(self.b)+' Ang')
			print('c = '+str(self.c)+' Ang')
			print('alpha = '+str(self.alpha)+' Ang')
			print('beta = '+str(self.beta)+' Ang')
			print('gamma = '+str(self.gamma)+' Ang')
			print(self.space_group)
			print('Space group: '+self.space_group)
			print('Unit Cell Volume ='+str(self.cell_vol))
			print('Formula weight = '+str(self.formula_weight))
			print('Formula units per unit cell = '+str(self.formula_units))
			print(cif_file+' imported successfully.'+'\n')
			print('###################### \n')

	def gen_flines(self):
		#simply retuns the flines string array
		cif_f = open(self.fname,'r')
		f_lines = cif_f.readlines()
		cif_f.close()
		return f_lines

	def generate_symmetry_operations(self):
		#Collect the symmetry equivalent sites. File format is 'site_id, symmetry equiv xyz'
		#Check for sure where the position is
		#returns the symmetry operations as array of strings, i.e. ['+x','+y','-z']

		symm_arr = self.cif_dict['_symmetry_equiv_pos_as_xyz']
		for i in range(len(symm_arr)):
			if type(symm_arr[i])==str:
				symm_arr[i]=symm_arr[i].split(',')
			else:
				pass
		#Now operations in format of ['+x','+y','+z']
		return symm_arr

	def gen_unique_coords(self):
		#Get the relevant atomic coordinates and displacement aprameters for unique positions
		f_lines = self.gen_flines()

		coords = {}
		#Make a dictionary of the ions and their positions- dctionary of format 
		# ['IonLabel': '['ion', fract_x, fract_y, fract_z, occupancy, Uiso, Uiso_val, Multiplicity]]

		atom_site_type_symbol_arr =  self.cif_dict['_atom_site_type_symbol']
		atom_site_label_arr = self.cif_dict['_atom_site_label']
		atom_site_fract_x_arr = self.cif_dict['_atom_site_fract_x']
		atom_site_fract_y_arr = self.cif_dict['_atom_site_fract_y']
		atom_site_fract_z_arr = self.cif_dict['_atom_site_fract_z']
		atom_site_occupancy_arr = self.cif_dict['_atom_site_occupancy']
		#The following may or may not be in the cif file
		try:
			atom_site_thermal_displace_type_arr = self.cif_dict['_atom_site_thermal_displace_type']
		except:
			atom_site_thermal_displace_type_arr = np.zeros(len(atom_site_type_symbol_arr))
		try:
			atom_site_U_iso_or_equiv_arr = self.cif_dict['_atom_site_U_iso_or_equiv']
		except:
			atom_site_U_iso_or_equiv_arr = np.zeros(len(atom_site_type_symbol_arr))
		try:
			atom_site_symmetry_multiplicity_arr = self.cif_dict['_atom_site_multiplicity']
		except:
			atom_site_symmetry_multiplicity_arr = np.zeros(len(atom_site_type_symbol_arr))
		for i in range(len(atom_site_label_arr)):
			ion = atom_site_type_symbol_arr[i]
			label = atom_site_label_arr[i]
			fract_x = atom_site_fract_x_arr[i]
			fract_y = atom_site_fract_y_arr[i]
			fract_z = atom_site_fract_z_arr[i]
			occupancy = atom_site_occupancy_arr[i]
			thermal_displace_type = atom_site_thermal_displace_type_arr[i]
			Uiso = atom_site_U_iso_or_equiv_arr[i]
			sym_mult = atom_site_symmetry_multiplicity_arr[i]
			coords[label]=[ion,fract_x,fract_y,fract_z,occupancy,thermal_displace_type,Uiso,sym_mult]
			#Note these are all still strings


		# The cif file tells the number of atom types in the cell- we can check the results of our symmetry operations with this.
		expected={}
		try:
			ion_list = self.cif_dict['_atom_type_symbol']
			ion_number = self.cif_dict['_atom_type_number_in_cell']
		except:
			ion_list = [0]
			ion_number = [0]
		for i in range(len(ion_list)):
			expected[ion_list[i]]=float(ion_number[i])
		#store this information for later
		self.expected_sites=expected
		return coords 
	def gen_unit_cell_positions(self):
			# Now we must generate all of the positions. We'll generate a list of format:
		#    [ion, x, y, z]
		# which makes structure factor calclation easier. 
		coords = self.gen_unique_coords()
		scatt_dict = self.scatt_dict
		symm_ops = self.generate_symmetry_operations()

		positions=[]
		structure_array=[]
		#for every ion, iterate over the symmetry operations. If the position already exists, it is not unique and not added to the posisions array 
		for position in coords:
			ion_coords = coords[position]
			ion = ion_coords[0]
			#For some reason O has a - next to it..
			ion = ion.replace('-','')
			ion = ion.replace('+','')
			ion = "".join([x for x in ion if x.isalpha()])
			x = float(ion_coords[1].replace(' ','').split('(')[0])
			y = float(ion_coords[2].replace(' ','').split('(')[0])
			z = float(ion_coords[3].replace(' ','').split('(')[0])
			try:
				if self.b_arr==False:
					b_el = float(scatt_dict[ion][1])
				else:
					b_el = float(scatt_dict[ion])
			except KeyError:
				print('Ion '+ion+' not found in NIST Tables or included b_arr. Include argument b_arr with elastic scattering lengths in fm when declaring Material object.')
				break
				return 0
			occupancy = float(ion_coords[4])

			for j in range(len(symm_ops)):
				symmetry = symm_ops[j]
				#replace the x with our new x, etc for y and z 
				x_sym = symmetry[0]
				x_eval_str = x_sym.replace('x',str(x))
				x_eval_str = x_eval_str.replace('y',str(y))
				x_eval_str = x_eval_str.replace('z',str(z))
				x_eval_str = x_eval_str.replace('/','*1.0/')
				y_sym = symmetry[1]
				y_eval_str = y_sym.replace('x',str(x))
				y_eval_str = y_eval_str.replace('y',str(y))
				y_eval_str = y_eval_str.replace('z',str(z))
				y_eval_str = y_eval_str.replace('/','*1.0/')

				z_sym = symmetry[2]
				z_eval_str = z_sym.replace('x',str(x))
				z_eval_str = z_eval_str.replace('y',str(y))
				z_eval_str = z_eval_str.replace('z',str(z))
				z_eval_str = z_eval_str.replace('/','*1.0/')
				x_pos = eval(x_eval_str)
				y_pos = eval(y_eval_str)
				z_pos = eval(z_eval_str)
				#assume that atoms can be no closer than 0.1 Ang
				z_pos = round(z_pos,2)
				x_pos = round(x_pos,2)
				y_pos = round(y_pos,2)				
				if x_pos==0.0:
					x_pos=0.0
				if y_pos==0.0:
					y_pos=0.0
				if z_pos==0.0:
					z_pos=0.0
				if x_pos<0.0:
					x_pos+=1.
				if x_pos>=1.0:
					x_pos-=1.
				if y_pos<0.0:
					y_pos+=1.
				if y_pos>=1.00:
					y_pos-=1.
				if z_pos<-0.0:
					z_pos+=1.
				if z_pos>=1.00:
					z_pos-=1.0
				

				occ = occupancy
				pos = [round(x_pos,2),round(y_pos,2),round(z_pos,2),occ]
				if pos not in positions:
					positions.append(pos)
					structure_array.append([ion,b_el,x_pos,y_pos,z_pos,occ])

		#Now we have all of the positions!
		self.unit_cell_xyz = structure_array
		return structure_array

	def plot_unit_cell(self,cmap='jet'):
		#NOTE: only supports up to 10 different atoms. 
		structure = self.gen_unit_cell_positions()
		unique_ions = np.unique(np.array(structure)[:,0])
		norm=matplotlib.colors.Normalize(vmin=0,vmax=len(unique_ions))
		figure = plt.figure(1,figsize=(8,8))
		MAX = np.max([self.a,self.b,self.c])
		ax = figure.add_subplot(111,projection='3d')
		used_ions = []
		for i in range(len(structure)):
			x = structure[i][2]*self.a
			y = structure[i][3]*self.b
			z = structure[i][4]*self.c
			b_val = structure[i][1]
			occupancy = structure[i][5]
			ion = structure[i][0]
			ion_i = np.where(unique_ions==ion)[0][0]

			color = np.array(matplotlib.cm.jet(norm(ion_i+0.5)))

			if ion not in used_ions:
				ax.scatter(x,y,z,c=color,label=ion,s=5.0*(np.abs(b_val)))
				used_ions.append(ion)
			else:
				ax.scatter(x,y,z,c=color,alpha=occupancy,s=5.0*(np.abs(b_val)))
		#Weirdly need to plot white points to fix the aspect ratio of the box
		# Create cubic bounding box to simulate equal aspect ratio
		max_range = np.max([self.a,self.b,self.c])
		mid_x = (self.a) * 0.5
		mid_y = (self.b) * 0.5
		mid_z = (self.c) * 0.5
		ax.set_xlim(mid_x - max_range, mid_x + max_range)
		ax.set_ylim(mid_y - max_range, mid_y + max_range)
		ax.set_zlim(mid_z - max_range, mid_z + max_range)
		ax.legend()
		plt.show()
		return figure, ax
	def gen_reflection_list(self,max_tau=20,maxQmag=1e10,b_dict=False):
		#Calculates the structure factor for all reflections in the unit cell. 
		# returns an array of arrays of format [H K L Freal Fi |F|^2 ]

		#NOTE add in occupancy later
		#Need to convert from fractional to real coordinates 
		#Returns in units of per unit cell 
		structure = self.gen_unit_cell_positions()

		F_HKL = 0.0
		#Generate array of Q-vectors
		taulim = np.arange(-max_tau+1,max_tau)
		xx, yy, zz = np.meshgrid(taulim,taulim,taulim)
		x = xx.flatten()
		y = yy.flatten()
		z = zz.flatten()
		#array of reciprocal lattice vectors; 4th column will be structure factor^2
		tau = np.array([x,y,z, np.zeros(len(x))]).transpose()  

		ion_list = np.array(structure)[:,0]
		occupancy_arr = np.array(structure)[:,5].astype(float)
		b_array = occupancy_arr*np.array(structure)[:,1].astype(float)

		#Imported from the NIST site so this is in femtometers, for result to be barn divide by 10
		b_array=b_array

		unit_cell_pos = np.array(structure)[:,2:5].astype(float)

		a_vec = self.avec
		b_vec =self.bvec
		c_vec = self.cvec
		u_vec = self.u
		v_vec = self.v
		w_vec = self.w

		i=0
		bad_ind = []
		for i in range(len(tau)):
			q_vect = tau[i][0:3]
			qmag = self.Qmag_HKL(q_vect[0],q_vect[1],q_vect[2])

			if qmag>maxQmag:
				bad_ind.append(i)
				tau[i,3]=0.0
			else:
				SF = 0
				#Sum over all ions in the cell
				for j in range(len(unit_cell_pos)):
					pos = unit_cell_pos[j]
					SF = SF + occupancy_arr[j]*b_array[j]*np.exp(2.0j*np.pi*np.inner(q_vect,pos))
				tau[i,3]=np.linalg.norm(SF)**2 / 100.0 # Fm^2 to barn
		# Eliminate tiny values
		tau[:,3][tau[:,3] < 1e-8] = 0.0
		low_reflect_i = np.where(tau[:,3]==0.0)[0]
		zero_ind = np.where(tau[:,3]==0.0)[0]
		# Divide but the number of formula units per unit cell to put it in units of barn / f.u.
		tau[:,3]=tau[:,3]#/self.formula_units
		self.HKL_list = tau
		return tau

	def calc_F_HKL(self,H,K,L):
		# Directly calculates a SF^2 for an arbitrary HKL index
		structure = self.gen_unit_cell_positions()

		F_HKL = 0.0
		#Generate array of Q-vectors
		tau = np.array([H,K,L,0])  

		ion_list = np.array(structure)[:,0]
		occupancy_arr = np.array(structure)[:,5].astype(float)
		b_array = occupancy_arr*np.array(structure)[:,1].astype(float)

		#Imported from the NIST site so this is in femtometers, for result to be barn divide by 10
		b_array=b_array*0.1

		unit_cell_pos = np.array(structure)[:,2:5].astype(float)

		a_vec = self.avec
		b_vec =self.bvec
		c_vec = self.cvec
		u_vec = self.u
		v_vec = self.v
		w_vec = self.w

		i=0
		bad_ind = []
		SF = 0
		for j in range(len(unit_cell_pos)):
			q_vect = np.array([H,K,L])
			pos = unit_cell_pos[j]
			SF = SF + occupancy_arr[j]*b_array[j]*np.exp(2.0j*np.pi*np.inner(q_vect,pos))
		tau[3]=np.linalg.norm(SF)**2
		# Divide but the number of formula units per unit cell to put it in units of barn / f.u.
		return tau 

	def fetch_F_HKL(self,H,K,L):
		# Returns the SF^2 of a particular reflection
		try:
			HKL = self.HKL_list
		except AttributeError:
			HKL = self.gen_reflection_list()
		index = np.argmin(np.abs(H-HKL[:,0])+np.abs(K-HKL[:,1])+np.abs(L-HKL[:,2]))
		#print('|F|^2 for ['+str(H)+str(K)+str(L)+'] ='+str(HKL[index][3])+' (fm^2/sr)')
		return HKL[index]
	
	def Qmag_HKL(self,H,K,L):
		#Returns the magnitude of the q-vector of the assosciated HKL indec in Ang^-1
		qvec = 2.0*np.pi*np.array(H*self.u+K*self.v+L*self.w)
		qmag = np.linalg.norm(qvec)
		return qmag 
	def twotheta_hkl(self,H,K,L,E,mode='deg'):
		#Simply feeds into equivalent Q_HKL function then converts to twotheta
		qmag = self.Qmag_HKL(H,K,L)
		lamb = 9.045 / np.sqrt(E)
		twotheta = np.arcsin(qmag*lamb/(4.0*np.pi))*2.0
		if mode=='deg':
			return twotheta*180.0/np.pi 
		else:
			return twotheta

	def theory_bragg_I(self,obsQ,intQ,intE,H,K,L,sample_mass):
		#Returns a scale factor to scale the data into  'barn/(eV mol sr fu)'
		#Makes a very rough assumption that Q^2 is constant through the integral (clearly it's not)
		#If you want to avoid this you need to evaluate int(Q^2), and input 1.0 as obsQ

		'''
		Input params:
			obsQ - center of bragg peak 
			intQ - integral in Q of bragg peak
			intE - integral in E of bragg peak (In meV!!)
			HKL, indices of bragg peak
			sample_mass- mass of sample in grams

		Returns:
			A scaling factor to normalize your dataset to the bragg peak.In units of fm^2
			Multiply by multiplicity afterwards if needed
		'''
		observed_Qsqr = obsQ**2 *(intQ)
		obs_E_int =intE #To become eV rather than meV
		I_obs = observed_Qsqr*obs_E_int
		f_HKL = self.fetch_F_HKL(H,K,L)[-1]
		#Convert to barn^2 / Sr
		f_HKL = f_HKL#*0.01
		#We want it per fu, so scale the fHKL to reflect this
		#f_HKL/=formula_scale
		density = self.formula_weight
		N = sample_mass/density 
		numerator = (4.0*np.pi)*I_obs*N 
		denom = f_HKL * (((2*np.pi)**3)/self.cell_vol)
		scaling_factor = denom/numerator 

		return scaling_factor

	def calc_sample_absorption(self,Ei,deltaE,d_eff,abs_dict=False,suppress_print=False):
		#Given a value or array of Ei and deltaE, returns the absorption per formula unit for the material. 
		#Also requires an effective distance

		#Can override the tabulated absorption if a dictionary in the format of abs_dict=['ion_str':absorption cross section] is give
		if abs_dict==False:
			scatt_dict = import_NIST_table(self.nist_file)

		lambda_i = np.sqrt(81.81 / Ei)
		lambda_f = np.sqrt(81.81/(Ei-deltaE))
		lambda0 = 3956.0/2200.0 
		rI = lambda_i / lambda0 
		rF = lambda_f / lambda0 
		cell_V = self.cell_vol 
		formula = self.chem_sum 
		num_units = self.formula_units
		formula_list = formula.split()
		atoms =[]
		for string in formula_list:
			num = ''.join(x for x in string if x.isdigit())
			ion = ''.join(x for x in string if x.isalpha())
			if not num:
				num=1
			atoms.append([ion,int(num)])
		sigma_abs = 0.0
		for i in range(len(atoms)):
			if abs_dict==False:
				abs_xc_str = self.scatt_dict[atoms[i][0]][-1]
				
				abs_xc = float(abs_xc_str.split('(')[0])
				sigma_abs = sigma_abs+(atoms[i][1]*abs_xc)
			else:
				abs_xc = abs_dict[atoms[i][0]]
				sigma_abs=sigma_abs + (atoms[i][1]*abs_xc)
		self.rho_abs = sigma_abs
		sigma_abs=sigma_abs*num_units
		if suppress_print==False:
			print('Mean elastic path length for Ei='+str(round(Ei,2))+'meV = '+str(round(1.0/(sigma_abs*rI/cell_V),2))+' cm')
		transmission_vs_energy_i = np.exp(-d_eff * sigma_abs * rI / cell_V)
		transmission_vs_energy_f = np.exp(-d_eff * sigma_abs * rF / cell_V)
		geo_mean_tranmission = np.sqrt(transmission_vs_energy_i*transmission_vs_energy_f)

		return geo_mean_tranmission


def genQslice(Qmin,Qmax,Qbins):
	QSlice = '|Q|, '+str(Qmin)+','+str(Qmax)+','+str(Qbins)
	return QSlice
def genEslice(Emin,Emax,Ebins):
	Eslice = 'DeltaE, '+str(Emin)+','+str(Emax)+','+str(Ebins)
	return Eslice

def import_files_to_MD(file_arr,MT_arr=False,Q_slice='[0,3,0.1]',E_slice='[-10,10,0.1]',self_shield=1.0,numEvNorm=True,absorb_correct=False,\
						absorb_arr=False,van_factor=1.0):
	# Takes file array and turns it into mdhisto
	# option available to correct for absorption, the absorb_arr needs to contain the following:
	'''
	Ei
	cell_vol Ang^3
	mean neutron path 
	absorption cross section 
	plot transmission (True, False)
	E0_shift
	'''

	if type(file_arr)==str:
		file_arr=[file_arr]
	if type(MT_arr)==str:
		MT_arr=[MT_arr]
	if type(file_arr)==list:
		matrix_ws = LoadNXSPE(file_arr[0])
	if MT_arr!=False:
		matrix_ws_MT = LoadNXSPE(MT_arr[0])
		i=1
		while len(MT_arr)>i:
			x=LoadNXSPE(MT_arr[i])
			matrix_ws_MT = matrix_ws_MT + x
			i=i+1
		matrix_ws_MT=matrix_ws_MT/i
	i=1
	while len(file_arr)>i:
		x = LoadNXSPE(file_arr[i])
		matrix_ws=matrix_ws+x
		i=i+1
	matrix_ws = matrix_ws/i

	#Convert to MD
	md_smpl=ConvertToMD(matrix_ws,Qdimensions='|Q|')
	if MT_arr!=False:
		md_MT=ConvertToMD(matrix_ws_MT,Qdimensions='|Q|')
		#Normalize to event.

	#Bin both
	cut2D_smpl = BinMD(md_smpl,AxisAligned=True,AlignedDim0=Q_slice,AlignedDim1=E_slice)
	#Normalize to event
	if numEvNorm==True:
		cut2D_smpl = normalize_MDHisto_event(cut2D_smpl)
	if MT_arr!=False:
		cut2D_MT = BinMD(md_MT,AxisAligned=True,AlignedDim0=Q_slice,AlignedDim1=E_slice)
		#normalize to event
		if numEvNorm==True:
			cut2D_MT = normalize_MDHisto_event(cut2D_MT)
		cut2D = cut2D_smpl-self_shield*cut2D_MT
		cut2D = cut2D*van_factor
	else:
		cut2D = cut2D_smpl*van_factor
	#Normalize by num events (already done earlier in an update)
	sets = [cut2D]
	'''
	if numEvNorm==True:
		for dataset in sets:
			dataset = normalize_MDHisto_event(dataset)
			dataset=dataset.clone()
	'''
	#Correct for absorption if desired
	if absorb_correct==True:
		Ei= absorb_arr[0]
		cell_vol = absorb_arr[1]
		neutron_d = absorb_arr[2]
		sigma_abs = absorb_arr[3]
		plot_transmission=absorb_arr[4]
		absorb_shift = absorb_arr[5]
		absorb_scaled_cut2D = correct_absorb_MD(cut2D,Ei,cell_vol,neutron_d,sigma_abs,plot_transmission,absorb_shift)
		return_cut2D=absorb_scaled_cut2D.clone()
	else:
		return_cut2D = cut2D.clone()
	return return_cut2D
def normalize_MDHisto_event(cut2D_normalize):
	#Normalizes a binned workspace by number of events
	sets = [cut2D_normalize]
	for dataset in sets:
		non_normalized_intensity = np.copy(dataset.getSignalArray())
		non_normalized_err = np.sqrt(np.copy(dataset.getErrorSquaredArray()))
		num_events = np.copy(dataset.getNumEventsArray())
		normalized_intensity=non_normalized_intensity/num_events
		normalized_error = non_normalized_err/num_events
		dataset.setSignalArray(normalized_intensity)
		dataset.setErrorSquaredArray(normalized_error**2)
	return sets[0]

def correct_absorb_MD_Material(absorbMD,Ei,material_object,d_eff,abs_dict=False,plot_transmission=False):
	#Assumes that a material object has already been created, calculates the proper absorption values. 
	md_absorption=absorbMD.clone() 
	I = np.copy(md_absorption.getSignalArray())
	err =np.sqrt(np.copy(md_absorption.getErrorSquaredArray()))
	dims = md_absorption.getNonIntegratedDimensions()
	q = mdu.dim2array(dims[0])
	e = mdu.dim2array(dims[1])
	
	transmission = material_object.calc_sample_absorption(d_eff=d_eff,deltaE=e,Ei=Ei,abs_dict=abs_dict) 
	for i in range(len(e)):
		I[:,i]/=transmission[i]
		err[:,i]/=transmission[i]
	md_absorption.setSignalArray(I)
	md_absorption.setErrorSquaredArray(err**2)  
	if plot_transmission==True:
		plt.figure()
		plt.xlabel('Energy (meV)')
		plt.ylabel('Transmission (fraction)')
		plt.title('Neutron transmission vs energy for Ei='+str(Ei)+' meV')
		plt.plot(e,transmission,color='r')
		plt.show()
	return md_absorption
