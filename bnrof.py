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
import scipy 
import traceback
'''
Below are some of the functions used throughout the analysis of Ba4NbRu3O12 data. 
They are defined here so as to not clog up the notebooks. 
'''

def sub_highQ_cut(in_MD,qlim,elim,scale=1.0):
	work_MD = in_MD.clone()
	dims = work_MD.getNonIntegratedDimensions()
	qdim,edim = mdu.dim2array(dims[0]),mdu.dim2array(dims[1])
	eres = np.abs(edim[1]-edim[0])
	sliced_MD = slice_box_QE_MD(work_MD,[0,np.max(qlim)],elim).clone()
	#Subtract the high Q cut from the rest of the measurement
	e,I,err = cut_MDHisto_powder(work_MD,'DeltaE',[elim[0],elim[1],eres],qlim)

	I_slice = np.copy(sliced_MD.getSignalArray())
	#This is slow but we can simply iterate through every point...
	for i in range(len(I_slice[:,0])):
		for j in range(len(I_slice[0,:])):
			energy = edim[j]
			near_i = np.argmin(np.abs(e-energy))
			cut_I = I[near_i]
			I_slice[i,j]-=cut_I*scale
	sliced_MD.setSignalArray(I_slice)
	return sliced_MD


def tempsubtract_cut2D(lowT_cut2D_T,highT_cut2D_T,tLow,tHigh,numEvNorm=False,vmin=0,vmax=5):
	#Same as normal bose einstein temperature subtraction but with cut2d workspaces instead of filenames.
	#Normalize by num events
	#Scale the highT dataset by bose-population factor
	highT_cut2D_tempsub = highT_cut2D_T.clone()
	lowT_cut2D_tempsub = lowT_cut2D_T.clone()
	hight_plot_tempsub = highT_cut2D_tempsub.clone()
	#pre_H_fig,pre_H_ax = fancy_plot_cut2D(hight_plot_tempsub, vmin=0,vmax=10,title='T=120K pre-scale')

	dims = lowT_cut2D_tempsub.getNonIntegratedDimensions()
	q_values = mdu.dim2array(dims[0])
	energies = mdu.dim2array(dims[1])
	if numEvNorm==True:
		lowT_cut2D_tempsub=normalize_MDHisto_event(lowT_cut2D_tempsub)
		highT_cut2D_tempsub = normalize_MDHisto_event(highT_cut2D_tempsub)
	kb=8.617e-2
	bose_factor_lowT = (1-np.exp(-energies/(kb*tLow)))
	bose_factor_highT = (1-np.exp(-energies/(kb*tHigh)))
	#Only makes sense for positive transfer
	bose_factor_lowT[np.where(energies<0)]=0
	bose_factor_highT[np.where(energies<0)]=0
	highT_Intensity = np.copy(highT_cut2D_tempsub.getSignalArray())
	highT_err = np.sqrt(np.copy(highT_cut2D_tempsub.getErrorSquaredArray()))
	bose_factor = bose_factor_highT/bose_factor_lowT
	highT_Intensity_corrected = bose_factor*highT_Intensity
	highT_err_corrected = bose_factor*highT_err
	highT_Intensity_corrected[np.where(highT_Intensity_corrected==0)]=0
	highT_err_corrected[np.where(highT_err_corrected==0)]=0
	highT_Intensity_corrected[np.isnan(highT_Intensity_corrected)]=0
	highT_err_corrected[np.isnan(highT_err_corrected)]=0

	highT_cut2D_tempsub.setSignalArray(highT_Intensity_corrected)
	highT_cut2D_tempsub.setErrorSquaredArray(highT_err_corrected**2)
	highT_plot_cut2D = highT_cut2D_tempsub.clone()
	lowt_tempsub_plot = lowT_cut2D_tempsub.clone()
	#pre_L_fig, pre_L_ax = fancy_plot_cut2D(lowt_tempsub_plot,vmin=vmin,vmax=vmax,title='T=5K')
	#post_H_fig, post_H_ax = fancy_plot_cut2D(highT_plot_cut2D,vmin=vmin,vmax=vmax,title='T=120K post-scale')
	#Don't really know if MANTID handles the subtraction well...
	lowT_cut2D_intensity = np.copy(lowT_cut2D_tempsub.getSignalArray())
	lowT_cut2D_err = np.sqrt(np.copy(lowT_cut2D_tempsub.getErrorSquaredArray()))

	mag_intensity = lowT_cut2D_intensity - highT_Intensity_corrected
	mag_err = np.sqrt(lowT_cut2D_err**2 + highT_err_corrected**2)

	cut2D_mag_tempsub= lowT_cut2D_tempsub.clone()
	cut2D_mag_tempsub.setSignalArray(mag_intensity)
	cut2D_mag_tempsub.setErrorSquaredArray(mag_err**2)
	cut2D_mag_tempsub_plot = cut2D_mag_tempsub.clone()
	#manget_fig,magnet_ax = fancy_plot_cut2D(cut2D_mag_tempsub_plot,vmin=vmin,vmax=vmax,title='magnetism')
	return cut2D_mag_tempsub



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
		Q_ff, FF = get_MANTID_magFF(q_all,mag_ion)
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


def stitch_MD(MD_arr,q_limits=False,e_limits=False,q_res=0.02,e_res=0.1,mode='overwrite'):
	#Overlaps different MDHisto datasets if one is preferred for any reason. 
	#Will go in order of priority- like so
	#Make sure that all MDHistoWorkspaces have the same binning distances

	#Assumes the workspaces are normalized to match eachother- if not, can do so with the norm_WS argument.

	# [highest_MD, second_MD,...., lowest MD]

	# Creates a grid with the finest resolution of the datasets. Empty bins in coarser datasets are filled with a linear interpolation of the adjacent non-zero pixels.
	# Of course, the errors of all of the pixels that were interpolated from need to be increased to account for this- rebinning should give the same error back as before stitching.

	# if q and e limits are defined, they are taken as the min/max of each MD to use. Some datasets are weird at edge of kinematics so it's best to cut them off.
	# Taken in format of [[qmin1,qmax1], [qmin2, qmax2], ...., [qmin_low,qmax_low]]


	# First get all Q, E values necessary.
	q_vals = []
	e_vals = []
	q_tot_min=0
	q_tot_max=0
	e_tot_min=0
	e_tot_max=0
	e_res_tot=1e8
	q_res_tot=1e8
	for md in MD_arr:
		dims=md.getNonIntegratedDimensions()
		q=mdu.dim2array(dims[0])
		e=mdu.dim2array(dims[1])
		q_res = np.abs(q[1]-q[0])
		e_res = np.abs(e[1]-e[0])
		qmin=np.min(q)
		if qmin<q_tot_min:
			q_tot_min=qmin 
		qmax=np.max(q)
		if qmax>q_tot_max:
			q_tot_max=qmax
		emin=np.min(e)
		if emin<e_tot_min:
			e_tot_min=emin
		emax=np.max(e)
		if emax>e_tot_max:
			e_tot_max=emax 
		#only retain lowest resolution 
		if q_res<q_res_tot:
			q_res_tot=q_res
		if e_res<e_res_tot:
			e_res_tot=e_res
	q_vals_output=np.arange(q_tot_min,q_tot_max+q_res_tot/2.0,q_res_tot)
	e_vals_output=np.arange(e_tot_min,e_tot_max+e_res_tot/2.0,e_res_tot)
	Q,E = np.meshgrid(e_vals_output,q_vals_output)
	new_I, new_err = np.zeros(np.shape(Q)),np.zeros(np.shape(Q))
	for i in range(len(MD_arr))[::-1]:
		#Cycle through MD workspaces, giving the points in the grid the closest values in intensity and errro
		MD = MD_arr[i]

		I = np.copy(MD.getSignalArray())
		err = np.sqrt(np.copy(MD.getErrorSquaredArray()))
		dims = MD.getNonIntegratedDimensions()
		num_use_mat = np.zeros(np.shape(I))
		q = mdu.dim2array(dims[0])
		e = mdu.dim2array(dims[1])
		Q_curr,E_curr = np.meshgrid(q,e)
		#elements that are zero or nan are out of the kinematic range and must not be appended to the final output. 
		I[np.isnan(I)]=0
		banned_i = np.where(I==0)
		#Now cycle through the output grid and match the closest index. If the intensity has already been appended to the output_arr,
		# be sure to adjust the error for the number of times it's been assigned at the end. (Goes as sqrt(N))
		if q_limits!=False:
			q_min_curr = q_limits[i][0]
			q_max_curr = q_limits[i][1]
		else:
			q_min_curr = np.min(q)-1.0
			q_max_curr = np.max(q)+1.0
		if e_limits!=False:
			e_min_curr = e_limits[i][0]
			e_max_curr = e_limits[i][1]
		else:
			e_min_curr = np.min(e)-1.0
			e_max_curr = np.max(e)+1.0
		for j in range(len(q_vals_output)):
			for k in range(len(e_vals_output)):
				#For each point in the grid, get the closest value in the relevant MDHisto
				e_grid_val = e_vals_output[k]
				q_grid_val = q_vals_output[j]
				closest_e_ind = np.argmin(np.abs(e_grid_val-e))
				closest_q_ind = np.argmin(np.abs(q_grid_val-q))
				banned_condition = (closest_q_ind in banned_i[0] and closest_e_ind in banned_i[1])
				q_val = q[closest_q_ind]
				e_val = e[closest_e_ind]
				I_val = I[closest_q_ind,closest_e_ind]
				Err_val = err[closest_q_ind,closest_e_ind]
				if not np.isnan(I_val) and I_val!=0 and q_val>=q_min_curr and q_val<=q_max_curr and e_val<=e_max_curr and e_val>=e_min_curr:
					if num_use_mat[closest_q_ind,closest_e_ind]==0:
						new_I[j,k]=I_val 
						new_err[j,k]=Err_val 

					if num_use_mat[closest_q_ind,closest_e_ind]>0 and mode!='weighted_mean':
						#Need to adjust the errors. First bring it back to one use. 
						new_I[j,k]=I_val
						scaled_err = Err_val/np.sqrt(num_use_mat[closest_q_ind,closest_e_ind])
						new_err[j,k]=scaled_err*np.sqrt(num_use_mat[closest_q_ind,closest_e_ind]+1.0)
					num_use_mat[closest_q_ind,closest_e_ind]+=1.0
	new_I = new_I.T 
	new_err = new_err.T
	#Create the output workspace
	extents_str = str(np.min(q_vals_output))+','+str(np.max(q_vals_output))+','+str(np.min(e_vals_output))+','+str(np.max(e_vals_output))
	num_bins_str = str(len(q_vals_output))+','+str(len(e_vals_output)) 

	output_ws = CreateMDHistoWorkspace(Dimensionality=2,SignalInput=new_I,ErrorInput=new_err,Extents=extents_str,NumberOfBins=num_bins_str,Units='MomentumTransfer,EnergyTransfer',Names='|Q|,DeltaE')
	return output_ws

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

def undo_normalize_MDHisto_event(cut2D_normalize):
	#Un-Normalizes a binned workspace by number of events
	sets = [cut2D_normalize]
	for dataset in sets:
		non_normalized_intensity = np.copy(dataset.getSignalArray())
		non_normalized_err = np.sqrt(np.copy(dataset.getErrorSquaredArray()))
		num_events = np.copy(dataset.getNumEventsArray())
		normalized_intensity=non_normalized_intensity*num_events
		normalized_error = non_normalized_err*num_events
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



def csv_to_MD(in_csv,skiprows=4,Ei=False,twoThetaMin=False,plot=False,transpose=False):
	#Given a csv file, imports as an MDHistoworkspace. Useful for DFT!!
	in_dat = np.loadtxt(open(in_csv,'rb'),delimiter=',',skiprows=skiprows)
	if transpose==True:
		in_dat = np.transpose(in_dat)
	Q = in_dat[:,0]
	E = in_dat[:,1]
	I = in_dat[:,2]
	Err = np.ones(np.shape(I)) #meaningless but zero gives errors
	qres = np.abs(np.unique(Q)[1]-np.unique(Q)[0])
	eres = np.abs(np.unique(E)[1]-np.unique(E)[0])

	csv_to_iexy = IEXY_data()

	csv_to_iexy.intensity=I
	csv_to_iexy.err=Err
	csv_to_iexy.q=Q
	csv_to_iexy.energies=E
	if plot==True:
		fig,ax =csv_to_iexy.plot_slice(vmin=0,vmax=np.mean(I)/2.0,axis_extents=[0,np.max(Q),0,np.max(E)])
		ax.set_xlim(0,np.max(Q))
		ax.set_ylim(0,np.max(E))

	csv_MD = csv_to_iexy.convert_to_MD()
	if Ei!=False:
		if twoThetaMin==False:
			twoThetaMin=3.5
		csv_MD=mask_minQ_fixedEi_MD(csv_MD,twoThetaMin,Ei)
	out_MD = csv_MD.clone()
	return out_MD

def minQmacs(Ef,twoTheta,deltaE=0):
	#Returns the lowest available Q for a given Ef and energy transfer
	Ei = deltaE +Ef 
	ki = np.sqrt(Ei/2.07)
	kf = np.sqrt(Ef/2.07)
	Q = np.sqrt( ki**2 + kf**2 - 2*ki*kf*np.cos(twoTheta*np.pi/180.0))
	return Q, deltaE

def minQseq(Ei,twoTheta,deltaE=0):
	#Returns lowest Q for Ei
	deltaEmax = Ei*0.9
	if deltaE==0:
		deltaE = np.linspace(0,deltaEmax,1000)

	Ef = Ei - deltaE

	ki = np.sqrt(Ei/2.07)
	kf = np.sqrt(Ef/2.07)
	Q = np.sqrt( ki**2 + kf**2 - 2*ki*kf*np.cos(twoTheta*np.pi/180))
	return Q, deltaE

def minQseq_multEi(Ei_arr,twoTheta,deltaE=0):
	#returns lowest accessible Q for a number of Ei's
	if not len(Ei_arr)>1:
		print('This function only takes an array of incident energies.') 
		return 0

	Eiarr=np.array(Ei_arr)
	if deltaE==0:
		deltaE=np.linspace(0,np.max(Ei_arr)*0.9,1000)
	Q_final_arr=[]
	#for every deltaE find which Ei has the lowest accessible Q. 
	# if the deltaE>0.9*Ef then this Ei is impossible
	for i in range(len(deltaE)):
		delE=deltaE[i]
		minQ=1000.0 #placeholder values
		for j in range(len(Eiarr)):
			Ei=Eiarr[j]
			if Ei>=0.9*delE:
				#allowed
				Ef = Ei - delE
				ki = np.sqrt(Ei/2.07)
				kf = np.sqrt(Ef/2.07)
				Q = np.sqrt( ki**2 + kf**2 - 2*ki*kf*np.cos(twoTheta*np.pi/180))
			else:
				Q=10.0
			if Q<minQ:
				minQ=Q
		Q_final_arr.append(minQ)
	return np.array(Q_final_arr),np.array(deltaE)

def mask_minQ_fixedEi_MD(seq_MD,twoThetaMin,Ei):
	#Remove areas outside of kinematic limit of SEQ, or any instrument with fixed Ei
	I = np.copy(seq_MD.getSignalArray())
	err = np.sqrt(np.copy(seq_MD.getErrorSquaredArray()))
	if type(Ei)==list:
		Q_arr,E_max = minQseq_multEi(Ei,twoThetaMin)
	else:
		Q_arr,E_max = minQseq(Ei,twoThetaMin)
	out_MD = seq_MD.clone()
	dims = seq_MD.getNonIntegratedDimensions()
	q_values = mdu.dim2array(dims[0])
	energies = mdu.dim2array(dims[1])
	for i in range(len(I)):
		q_cut = I[i]
		q_val = q_values[i]
		err_cut = err[i]
		kinematic_E = E_max[np.argmin(np.abs(q_val-Q_arr))]
		q_cut[np.where(energies>kinematic_E)]=np.nan
		err_cut[np.where(energies>kinematic_E)]=np.nan
		I[i]=q_cut
		err[i]=err_cut
	out_MD.setSignalArray(I)
	out_MD.setErrorSquaredArray(err**2)
	return out_MD
def csv_to_MD_mat(in_csv,q_arr,e_arr,skiprows=0,Ei=False,twoThetaMin=False,plot=False,transpose=False):
	#Same as csv_to_MD but instead of the format being four columns, it's a matrix.
	#The extents are defined by q_arr and e_arr which are in the format of [min,max]
	in_dat = np.loadtxt(open(in_csv,'rb'),delimiter=',',skiprows=skiprows)
	if transpose==True:
		in_dat==np.transpose(in_dat)
	q = np.linspace(q_arr[0],q_arr[1],np.shape(in_dat)[0])
	e = np.linspace(e_arr[0],e_arr[1],np.shape(in_dat)[1])
	Q,E = np.meshgrid(q,e)
	I = in_dat.T 
	#Make the mdhistoworkspace 
	I_arr=I.flatten()
	err_arr=I_arr/I_arr
	extents_str = str(np.min(q))+','+str(np.max(q))+','+str(np.min(e))\
						+','+str(np.max(e))
	num_bin_str = str(len(np.unique(q)))+','+str(len(np.unique(e)))
	out_ws = CreateMDHistoWorkspace(Dimensionality=2,Extents=extents_str,SignalInput=I_arr,\
				ErrorInput=err_arr,NumberOfBins=num_bin_str,NumberOfEvents=np.ones(len(I_arr))\
				,Names='Dim1,Dim2',Units='MomentumTransfer,EnergyTransfer')
	return out_ws
def scale_dftMD_to_highQ(md_obs,md_dft,qrange,erange,Ei,plot_fit=True,inputK=False,inputE=False,inputA=False,noscale=False):
	#Function to try to correct for energy dependent errors in DFT calculations using a high Q
	# measurement
	md_obs = md_obs.clone()
	md_dft = md_dft.clone()
	dims_obs = md_obs.getNonIntegratedDimensions()
	e_obs = mdu.dim2array(dims_obs[1])
	eres = np.abs(e_obs[0]-e_obs[1])
	e,i,err = cut_MDHisto_powder(md_obs,'DeltaE',[erange[0],erange[1],eres],[qrange[0],qrange[1]])
	e2,i2,err2 = cut_MDHisto_powder(md_dft,'DeltaE',[erange[0],erange[1],eres],[qrange[0],qrange[1]])
	Ei_arr = np.ones(len(e))*Ei
	Ef_arr = np.ones(len(e))*Ei-e
	lambda_i_list = np.sqrt(81.82/Ei_arr)
	lambda_f_list = np.sqrt(81.82/Ef_arr)
	ki_list = 2.0*np.pi/lambda_i_list
	kf_list = 2.0*np.pi/lambda_f_list
	def calc_I(I_dft,I_meas,A,T,Ei):
		#Helper function only to be used in the dft scaling function.
		#rescale I to have the correct energy dependence
		K = (I_meas / (A*I_dft))*(1.0/(1.0+(ki_list**2 + kf_list**2)))
		K = 1.0
		I=I_dft*K
		outI = I*T
		outErr = err2*T
		mc_term = (1.0-T)*(ki_list**2 + kf_list**2) * I       
		total_I = (mc_term+outI)*A
		return total_I
	box_bragg = slice_box_QE_MD(md_obs,[0,4],[4,40]).clone()
	box_dft = slice_box_QE_MD(md_dft,[0,4],[4,40]).clone()
	mult_model = Model(calc_I,independent_vars=['I_dft','I_meas'])
	params=mult_model.make_params()
	#Fix A to be dsome input value if desired
	if type(inputA)!=bool:
		params.add('A',value=inputA,min=0.1,max=100.0,vary=False)
	else:
		params.add('A',value=6.0,min=0.1,max=100.0,vary=True)
	params.add('T',value=1.0,min=0.9,max=1.1)
	params.add('Ei',value=Ei,vary=False)
	weights = 1.0/err
	if type(inputK)== bool:
		result = mult_model.fit(i,I_dft=i2,I_meas=i,params=params,weights=weights,method='powell')
		T_result = result.params['T'].value
		A_result = result.params['A'].value
		#Get a list of scaling factors for every omega, rescale I_dft
		K = (i/(A_result*i2))*(1.0/(T_result+(1.0-T_result)*(ki_list**2 + kf_list**2)))
		#Finally, apply this scaling factor to the entire MD datset for every omega. Easiest to do by just making new 
		#MDs with the regions of interest.
	else:
		e=inputE
		K=inputK #For use with low temp measurements
		T_result=1.0
		A_result=1.0
	newI = np.copy(box_dft.getSignalArray())
	interp_K = scipy.interpolate.interp1d(x=e,y=K,bounds_error=False)
	dims_new = box_dft.getNonIntegratedDimensions()
	e_new = mdu.dim2array(dims_new[1])
	scaled_i2 = np.copy(i2)
	for ii in range(len(e_new)):
		omega = e_new[ii]
		K_omega = interp_K(x=omega)
		#K_omega=1.0
		if noscale!=True:
			newI[:,ii]=newI[:,ii]*K_omega
	if noscale!=True:
		scaled_dft = T_result*i2*A_result*interp_K(e2)
	else:
		scaled_dft = i2*A_result
	box_dft.setSignalArray(newI)
	mc_term = (1.0-T_result)*(ki_list**2 + kf_list**2) * scaled_dft/T_result
	if plot_fit==True and type(inputK)==bool:
		print(result.fit_report())
		fig,ax=plt.subplots(figsize=(8,5))
		ax.errorbar(e,i,yerr=err,color='k',label='Data 300K',marker='o',capsize=3,ls='--')
		ax.errorbar(e2,T_result*i2*A_result,yerr=err2,color='b',label='DFT 300K',marker='o',capsize=3,ls=' ')
		ax.plot(e2,mc_term,'g',label='Multiple Scattering')
		ax.plot(e2,scaled_dft,label='Renormalized DFT',color='b')
		ax.plot(e2,mc_term+scaled_dft,label='Best fit',color='r')
		ax.set_xlabel('E (meV)')
		ax.legend()
		ax.set_ylim(0,30)
		ax.set_ylabel('I (barn/eV/fu/sr)')
		ax.set_title('DFT F($\omega$) adjustment for E$_i$='+str(Ei)+' meV')
	return box_dft,box_bragg,T_result,A_result,e2,interp_K(e2)

def energyscale_MD(in_md,omegas,K):
	#Scales an MD in energy by a given factor of K across the whole dataset.
	work_MD = in_md.clone()
	I = np.copy(work_MD.getSignalArray())
	Err = np.sqrt(np.copy(work_MD.getSignalArray()))
	dims = work_MD.getNonIntegratedDimensions()
	e = mdu.dim2array(dims[1])
	interpK = scipy.interpolate.interp1d(y=K,x=omegas,bounds_error=False)

	for i in range(len(e)):
		I[:,i]*=interpK(e[i])
		Err[:,i]*=interpK(e[i])
	work_MD.setSignalArray(I)
	work_MD.setErrorSquaredArray(Err**2)
	return work_MD

def imitate_MD(md1,md2,mode='nearest-neighbor'):
	#Given two mdhistoworkspaces, rescales md1 to the same dimensionality as md2 using a nearest neighbor approach
	md_in = md1.clone()
	md_highdim = md2.clone()
	dims = md_in.getNonIntegratedDimensions()
	q= mdu.dim2array(dims[0])
	e= mdu.dim2array(dims[1])
	Q,E = np.meshgrid(q,e)
	outMD = md_in.clone()
	ref_I = np.copy(md_in.getSignalArray())
	out_I = outMD.getSignalArray()
	out_I = out_I * np.nan
	outMD.setSignalArray(out_I)
	out_err = np.sqrt(np.copy(outMD.getErrorSquaredArray()))
	out_err = out_err * np.nan
	outMD.setErrorSquaredArray(out_err)
	#Now we have an output mdhisto of the correct dimensionality consisting of only NaN values. 
	dims2= md_highdim.getNonIntegratedDimensions()
	q2 = mdu.dim2array(dims2[0])
	e2 = mdu.dim2array(dims2[1])
	I_orig = np.copy(md_highdim.getSignalArray())
	Err_orig = np.sqrt(np.copy(md_highdim.getErrorSquaredArray()))
	for i in range(len(q)):
		for j in range(len(e)):
			q_ind = np.argmin(np.abs(q2-q[i]))
			e_ind = np.argmin(np.abs(e2-e[j]))
			I_reference = ref_I[i,j]
			I_nearest = I_orig[q_ind,e_ind]
			Err_nearest = Err_orig[q_ind,e_ind]
			if I_reference!=0 and not np.isnan(I_reference):
				out_I[i,j]=I_nearest
				out_err[i,j]=Err_nearest
	errsqr = out_err**2
	outMD.setSignalArray(out_I)
	outMD.setErrorSquaredArray(errsqr)
	return outMD

def sub_nearest_MD(md_left,md_right,mode='subtract'):
	#allows for subtraction of MDHistos with unequal bin sizes. Uses closest value in coarser grid.
	out_MD = md_left.clone()
	dims = out_MD.getNonIntegratedDimensions()
	qLeft = mdu.dim2array(dims[0])
	eLeft = mdu.dim2array(dims[1])

	sub_dims = md_right.getNonIntegratedDimensions()
	q_sub = mdu.dim2array(sub_dims[0])
	e_sub = mdu.dim2array(sub_dims[1])

	I_Left=np.copy(md_left.getSignalArray())
	I_new = np.copy(I_Left)
	Ierr_left = np.sqrt(np.copy(md_left.getErrorSquaredArray()))
	new_err = np.copy(Ierr_left)
	I_sub = np.copy(md_right.getSignalArray())
	I_sub_err = np.sqrt(np.copy(md_right.getErrorSquaredArray()))
	for i in range(len(qLeft)):
		for j in range(len(eLeft)):
			q_arg = np.argmin(np.abs(q_sub-qLeft[i]))
			e_arg = np.argmin(np.abs(e_sub-eLeft[j]))
			I_new[i,j]=I_Left[i,j]-I_sub[q_arg,e_arg]
			err_sub = I_sub_err[q_arg,e_arg]
			err_net = np.sqrt(Ierr_left[i,j]**2 + err_sub**2)
			new_err[i,j]=err_net 
	out_MD.setSignalArray(I_new)
	out_MD.setErrorSquaredArray(new_err**2)
	return out_MD



def cut_MDHisto_powder(workspace_cut1D,axis,extents,integration_range, auto_plot=False, extra_text='',plot_min=0,plot_max=1e-4,debug=False):
	#Takes an MD Histo and returns x, y for a cut in Q or E
	#only for powder data
	# Workspace - an mdhistoworkspace
	# axis - |Q| or DeltaE (mev)
	# extents - min, max, step_size for cut axis- array
	# Integration range- array of min, max for axis being summed over

	#Normalize by num events
	sets = [workspace_cut1D]
	intensities = np.copy(workspace_cut1D.getSignalArray())*1.

	errors = np.sqrt(np.copy(workspace_cut1D.getErrorSquaredArray()*1.))
	errors[np.isnan(intensities)]=1e30
	#clean of nan values
	intensities[np.isnan(intensities)]=0
	if debug==True:
		print('random row of intensities')
		print(intensities[3])

	dims = workspace_cut1D.getNonIntegratedDimensions()
	q = mdu.dim2array(dims[0])
	e = mdu.dim2array(dims[1])

	if axis=='|Q|':
		#First limit range in E
		e_slice = intensities[:,np.intersect1d(np.where(e>=integration_range[0]),np.where(e<=integration_range[1]))]
		slice_errs = errors[:,np.intersect1d(np.where(e>=integration_range[0]),np.where(e<=integration_range[1]))]
		#Integrate over E for all values of Q
		integrated_intensities = []
		integrated_errs=[]
		for i in range(len(e_slice[:,0])):
			q_cut_vals = e_slice[i]
			q_cut_err = slice_errs[i]

			q_cut_err=q_cut_err[np.intersect1d(np.where(q_cut_vals!=0)[0],np.where(~np.isnan(q_cut_vals)))]
			q_cut_vals=q_cut_vals[np.intersect1d(np.where(q_cut_vals!=0)[0],np.where(~np.isnan(q_cut_vals)))]


			if len(q_cut_vals>0):
				integrated_err=np.sqrt(np.nansum(q_cut_err**2))/len(q_cut_vals)
				integrated_intensity=np.average(q_cut_vals,weights=1.0/q_cut_err)
				integrated_errs.append(integrated_err)
				integrated_intensities.append(integrated_intensity)
			else:
				integrated_err=0
				integrated_intensity=0
				integrated_errs.append(integrated_err)
				integrated_intensities.append(integrated_intensity)

		q_vals = q
		binned_intensities = integrated_intensities
		binned_errors = integrated_errs
		bin_x = q_vals
		bin_y = binned_intensities
		bin_y_err = binned_errors
		other = '$\hbar\omega$'
		# Now bin the cut as specified by the extents array
		extent_res = np.abs(extents[1]-extents[0])
		bins = np.arange(extents[0],extents[1]+extents[2]/2.0,extents[2])
		bin_x,bin_y,bin_y_err = bin_1D(q,bin_y,bin_y_err,bins,statistic='mean')
	elif axis=='DeltaE':
		#First restrict range across Q
		q_slice = intensities[np.intersect1d(np.where(q>=integration_range[0]),np.where(q<=integration_range[1]))]
		slice_errs = errors[np.intersect1d(np.where(q>=integration_range[0]),np.where(q<=integration_range[1]))]
		#Integrate over E for all values of Q
		integrated_intensities = []
		integrated_errs=[]
		for i in range(len(q_slice[0])):
			e_cut_vals = q_slice[:,i]
			e_cut_err = slice_errs[:,i]
			e_cut_err = e_cut_err[np.intersect1d(np.where(e_cut_vals!=0)[0],np.where(~np.isnan(e_cut_vals)))]
			e_cut_vals=e_cut_vals[np.intersect1d(np.where(e_cut_vals!=0)[0],np.where(~np.isnan(e_cut_vals)))]


			if len(e_cut_vals)>0:
				integrated_err=np.sqrt(np.nansum(e_cut_err**2))/len(e_cut_vals)
				integrated_intensity=np.average(e_cut_vals,weights=1.0/e_cut_err)
				integrated_errs.append(integrated_err)
				integrated_intensities.append(integrated_intensity)
			else:
				integrated_errs.append(0)
				integrated_intensities.append(0)
		bin_x = e
		bin_y = integrated_intensities
		bin_y_err = integrated_errs

		bins = np.arange(extents[0],extents[1]+extents[2]/2.0,extents[2])
		bin_x,bin_y,bin_y_err = bin_1D(e,bin_y,bin_y_err,bins,statistic='mean')
		other = '|Q|'
	else:
		print('Invalid axis option (Use \'|Q|\' or \'DeltaE\')')
		return False
	if auto_plot==True:
		#Attempts to make a plot of the cut. Limits and such will be off

		cut_fig, cut_ax = plt.subplots(1,1,figsize=(8,6))
		cut_ax.set_title(axis+' Cut')
		cut_ax.set_xlabel(axis)
		cut_ax.set_ylabel('Intensity (arb.)')
		cut_ax.errorbar(x=bin_x,y=bin_y,yerr=bin_y_err,marker='o',color='k',ls='--',mfc='w',mec='k',mew=1,capsize=3)
		cut_ax.set_ylim(np.nanmin(bin_x)*0.8,np.nanmax(bin_y)*1.5)
		cut_ax.text(0.7,0.95,other+'='+str(integration_range),transform=cut_ax.transAxes)
		cut_ax.text(0.7,0.85,extra_text,transform=cut_ax.transAxes)
		cut_fig.show()

	return np.array(bin_x),np.array(bin_y),np.array(bin_y_err)


def slice_box_QE_MD(input_md,qlim,elim):
	#Given limits that the user wants to keep, returns a new workspace with only the selected box
	working_MD = input_md.clone()
	dims=working_MD.getNonIntegratedDimensions()
	q = mdu.dim2array(dims[0])
	e = mdu.dim2array(dims[1])
	I= np.copy(working_MD.getSignalArray())
	err = np.sqrt(np.copy(working_MD.getErrorSquaredArray()))
	qmin= qlim[0]
	qmax = qlim[1]
	emin=elim[0]
	emax=elim[1]
	working_MD = mask_QE_box_MD(working_MD,[np.min(q),np.max(q)],[np.min(e),emin]).clone()
	working_MD = mask_QE_box_MD(working_MD,[np.min(q),qmin],[np.min(e),np.max(e)]).clone()
	working_MD = mask_QE_box_MD(working_MD,[np.min(q),np.max(q)],[emax,np.max(e)]).clone()
	working_MD = mask_QE_box_MD(working_MD,[qmax,np.max(q)],[np.min(e),np.max(e)]).clone()
	# For efficienty now we don't need the rest of the data. 
	return working_MD



def mask_QE_box_MD(input_md,qlim,elim):
	#given limits in Q, E, masks an MDHistoworkspace. Useful for masking regions with bad data.
	working_MD = input_md.clone()
	dims = working_MD.getNonIntegratedDimensions()
	q = mdu.dim2array(dims[0])
	e = mdu.dim2array(dims[1])
	I = np.copy(working_MD.getSignalArray())
	err = np.sqrt(np.copy(working_MD.getErrorSquaredArray()))
	qmin=qlim[0]
	qmax=qlim[1]
	emin=elim[0]
	emax=elim[1]

	for i in range(len(I[:,0])):
		for j in range(len(I[0])):
			point = I[i,j]
			q_curr = q[i]
			e_curr = e[j]
			if q_curr<=qmax and q_curr>=qmin and e_curr<=emax and e_curr>=emin :
				#In the 'box'
				I[i,j]=np.nan
				#Don't touch the error- should be effectively inf but we'll leave it alone.
				err[i,j]=np.inf
	working_MD.setSignalArray(I)
	working_MD.setErrorSquaredArray(err**2)
	return working_MD



def MDfactorizationv2(workspace_MDHisto,mag_ion='Ir4',q_lim=False,e_lim=False,Ei=50.0,twoThetaMin=3.5,plot_result=True,method='powell',fname='placeholder.jpg',\
						fast_mode=False,overwrite_prev=False,allow_neg_E=True,g_factor=2.0,debug=False,fix_Qcut=False,fix_Ecut=False):
	#Does a rocking curve foreach parameter to determine its uncertainty
	#Assumes that z is already flatted into a 1D array
	#This is specifically for the factorization technique
	#below contents are from old version of function 

	#Version 2 includes an updated definition of F(omega)
	if overwrite_prev==True and os.path.exists(fname):
		#Delete the file
		os.remove(fname)

	dims = workspace_MDHisto.getNonIntegratedDimensions()
	q_values = mdu.dim2array(dims[0])
	energies = mdu.dim2array(dims[1])
	intensities = np.copy(workspace_MDHisto.getSignalArray())
	errors = np.sqrt(np.copy(workspace_MDHisto.getErrorSquaredArray()))
	if q_lim!=False and e_lim!=False:
		qmin=q_lim[0]
		qmax=q_lim[1]
		emin=e_lim[0]
		emax=e_lim[1]
	elif q_lim==False and e_lim!=False:
		#e_lim has not been defined.
		qmin=np.min(q_values)
		qmax=np.max(q_values)
		emin=e_lim[0]
		emax=e_lim[1]
	elif e_lim!=False and q_lim==False:
		#q-lim hasn't been defined
		qmin=q_lim[0]
		qmax=q_lim[1]
		emin=np.min(energies)
		emax=np.max(energies)
	else:
		qmin=np.min(q_values)
		qmax=np.max(q_values)
		emin=np.min(energies)
		emax=np.max(energies)

	intensities = intensities[np.intersect1d(np.where(q_values>=qmin),np.where(q_values<=qmax))]
	intensities = intensities[:,np.intersect1d(np.where(energies>=emin),np.where(energies<=emax))]
	errors = errors[np.intersect1d(np.where(q_values>=qmin),np.where(q_values<=qmax))]
	errors = errors[:,np.intersect1d(np.where(energies>=emin),np.where(energies<=emax))]
	energies = energies[np.intersect1d(np.where(energies>=emin),np.where(energies<=emax))]
	q_values = q_values[np.intersect1d(np.where(q_values>=qmin),np.where(q_values<=qmax))]

	#Remove areas outside of kinematic limit of SEQ 
	#This is outside of the scope of the script.
	if twoThetaMin!=False and Ei!=False:
		if twoThetaMin!=False and (type(Ei) != list):
			Q_arr,E_max = minQseq(Ei,twoThetaMin)
		elif type(Ei)!=list:
			Q_arr,E_max = minQseq(Ei,twoThetaMin)
				
		if type(Ei)==list:
			Q_arr,E_max = minQseq_multEi(Ei,twoThetaMin)
		for i in range(len(intensities)):
			q_cut = intensities[i]
			q_val = q_values[i]
			err_cut = errors[i]
			kinematic_E = E_max[np.argmin(np.abs(q_val-Q_arr))]
			q_cut[np.where(energies>kinematic_E)]=np.nan
			err_cut[np.where(energies>kinematic_E)]=np.nan
			intensities[i]=q_cut
			errors[i]=err_cut
	


	x = q_values
	y = energies
	z = intensities
	bad_i = np.isnan(z)
	intensities[np.isnan(z)]=0
	errors[np.isnan(z)]=1e10
	errors[errors==0]=1e10
	errors[np.isnan(errors)]=1e10

	#Take big cuts of the dataset to get a good guess
	q_res = np.abs(q_values[1]-q_values[0])
	e_res = np.abs(energies[1]-energies[0])
	
	q_vals_guess = np.copy(x)
	q_cut_guess = np.zeros(len(x))
	q_cut_guess_errs=np.zeros(len(x))
	e_cut_guess = np.zeros(len(y))
	e_cut_guess_errs = np.zeros(len(y))
	for i in range(len(q_cut_guess)):
		q_i = intensities[i,:]
		qerr_i = errors[i,:]
		qpt = np.average(q_i,weights=1.0/qerr_i)
		q_cut_guess[i]=qpt 
		q_cut_guess_errs[i]=qpt*np.mean(q_i/qerr_i)
	for i in range(len(e_cut_guess)):
		e_i = intensities[:,i]
		eerr_i = errors[:,i]
		ept=np.average(e_i,weights=1.0/eerr_i)
		e_cut_guess[i]=ept 
		e_cut_guess_errs[i]=ept*np.mean(e_i/eerr_i)
	
	#q_vals_guess,q_cut_guess,q_cut_guess_errs = cut_MDHisto_powder(workspace_MDHisto,'|Q|',[np.nanmin(x)-q_res/2.0,np.nanmax(x)+q_res/2.0,q_res/10.0],e_lim)
	#e_vals_guess,e_cut_guess,e_cut_guess_errs = cut_MDHisto_powder(workspace_MDHisto,'DeltaE',[np.nanmin(y)-e_res/2.0,np.nanmax(y)+e_res/2.0,e_res/10.0],q_lim)

	e_vals_guess = y 
	q_vals_guess = x
	#Normalize the cuts to one in energy
	e_cut_guess[np.where(e_cut_guess<=0)[0]]=1e-2
	e_cut_integral = np.trapz(x=y,y=e_cut_guess)
	print('Gw integral')
	print(e_cut_integral)
	e_cut_guess/=e_cut_integral
	q_cut_guess*=e_cut_integral
	#Convert the actual cut into deltas used in the expoential definition of G(omega)
	#We do this by defining the delta at the first value in the cut to be zero. 
	g_omega_0 = e_cut_guess[0]
	delta_0 = 0.0
	Z = 1.0/g_omega_0 #This is used to solve for delta now. 
	delta_arr = np.zeros(len(e_cut_guess))
	delta_arr = np.log(1.0/e_cut_guess*Z)
	calc_ecut_guess = np.exp(-1.0*delta_arr)/Z

	m = len(y) # number of E-values
	n = len(x) #number of q_values
	#e_cut_guess/=e_cut_guess[0]
	#q_cut_guess*=e_cut_guess[0]
	Q_cut = q_cut_guess.reshape(1,n)
	E_cut = e_cut_guess.reshape(m,1)
	xy=Q_cut*E_cut
	arr_guess = np.append(q_cut_guess,delta_arr)
	arr_guess[np.isnan(arr_guess)]=0
	params= Parameters()
	for i in range(len(arr_guess)):
		val = arr_guess[i]
		if i>=n:
			#Spectral weight can't be negative physically
			#Need to fix the first energy value 
			if i==n:
				vary_val = False
				param_guess = 0.0
			else:
				if fix_Ecut==True:
					vary_val=False
				else:
					vary_val=True
				param_guess = arr_guess[i]
			#From of G(w) is e^(-delta), e^-15 is 1e-7
			params.add('param_'+str(i),vary=vary_val,value=param_guess,max=15.0)
		else:
			if fix_Qcut == True:
				vary_val=False
			else:
				vary_val=True
			params.add('param_'+str(i),value=val,vary=vary_val)
	if debug==True:
		plt.figure()
		plt.errorbar(x,q_cut_guess,q_cut_guess_errs,color='k')
		plt.show()
		plt.figure()
		plt.errorbar(y,calc_ecut_guess,e_cut_guess_errs,color='k')
		plt.errorbar(y,e_cut_guess,e_cut_guess_errs,color='r')
		plt.show()

	weights = np.ones(np.shape(intensities))
	weights = 1.0/(np.abs(errors))

	#Make note of q, e indices at which there exist no intensities. They will be masked.
	bad_q_i=[]
	bad_e_i=[]
	for i in range(np.shape(intensities)[0]):
		#Check Q-cuts
		q_cut = intensities[i]
		num_nan = np.sum(np.isnan(q_cut))
		num_zero = np.sum([q_cut==0])
		num_bad = num_nan+num_zero
		if num_bad==len(q_cut):
			bad_q_i.append(i)
		else:
			#Do nothing
			pass
	#Some high energies will also have no counts from detailed balance
	for i in range(np.shape(intensities)[1]):
		e_cut = intensities[:,i]
		num_nan = np.sum(np.isnan(e_cut))
		num_zero = np.sum([e_cut==0])
		num_bad =num_nan+num_zero
		if num_bad==len(e_cut):
			bad_e_i.append(i)
		else:
			#Do nothing
			pass
	weights=np.ravel(weights)
	meas_errs=1.0 / weights 

	z_fit = np.copy(intensities)
	Q,E = np.meshgrid(e_vals_guess,q_vals_guess)
	z_fit_orig = np.copy(z_fit)
	z_fit = z_fit.flatten()
	#weights[z_fit==0]=0
	meas_errs[np.isnan(z_fit)]=np.inf

	z_fit[np.isnan(z_fit)]=np.nan

	xy = np.arange(z_fit.size)
	num_points = len(z_fit) - np.sum(np.isnan(z_fit)) - np.sum([z_fit==0])
	vals = []
	for i in range(len(params)):
		vals.append(params['param_'+str(i)].value)
	vals = np.array(vals)
	Q_vals = vals[0:n].reshape(n,1) #Reconstruct y coordinates
	E_vals = vals[n:].reshape(1,m) # '' x coord
	slice2D = Q_vals * E_vals


	weights = 1.0/meas_errs
	data = z_fit 
	data[np.isnan(data)]=0
	weights[data==0]=0
	meas_errs=1.0/weights 
	meas_errs[weights==0]=np.nan
	model = Model(factorization_f,independent_vars=['n','m','delE'])
	eRes = np.abs(energies[1]-energies[0])
	#minimize this chisqr function. 
	result = model.fit(data,n=n,m=m,delE=eRes,params=params,method=method,weights=weights,nan_policy='omit')
	f_array=[]
	for i in range(len(result.params)):
		f_array.append(result.params['param_'+str(i)].value)


	num_operations=len(params)
	#Normalize s.t. energy spectra integrates to one
	q = q_values
	e = energies
	x_q = np.array(f_array[0:n])
	x_q[bad_q_i]=0
	deltas = np.array(f_array[n:])
	Z = np.nansum(np.exp(-1.0*deltas))
	g_e = np.exp(-1.0*deltas)/(eRes*Z) 
	if os.path.isfile(fname) or fast_mode==True:
		if fast_mode==True:
			err_array =0.3*np.array(f_array)
		else:
			err_dict = np.load(fname,allow_pickle=True).item()
			err_array=[]
			for i in range(len(err_dict.keys())):
				key = 'param_'+str(i)
				err_array.append(err_dict[key])
		x_q_errs = err_array[0:n]
		delta_errs = err_array[n:]
		#Need to normalize

		#Normalize s.t. energy spectra integrates to one
		q = q_values
		e = energies
		x_q = np.array(f_array[0:n])
		x_q[bad_q_i]=0
		deltas = np.array(f_array[n:])
		Z = np.nansum(np.exp(-1.0*deltas))
		g_e = np.exp(-1.0*deltas)/(eRes*Z) 

		xq_err = err_array[0:n]
		ge_err = err_array[n:]

		x_q,g_e,xq_err,ge_err = np.array(x_q),np.array(g_e),np.array(xq_err),np.array(ge_err)

		#Now convert X(Q) into S(Q)
		r0 = 0.5391
		g=g_factor
		q_FF,magFF = get_MANTID_magFF(q,mag_ion)
		magFFsqr = 1.0/np.array(magFF)
		s_q = (2.0*x_q)/(r0**2 * g**2 * magFFsqr)
		s_q_err = (2.0*xq_err)/(r0**2 * g**2 * magFFsqr)
		return q,s_q,s_q_err,e,g_e,ge_err
	#Get error bars using random sampling method
	#Create a parameter object mased on linearized form of the factorization for uncertainty

	err_dict = calculate_param_uncertainty(data,meas_errs,model,result.params,result,fast_calc=fast_mode,independent_vars={'n':n,'m':m,'delE':eRes},\
		extrapolate=True,show_plots=True,fname=fname,overwrite_prev=overwrite_prev,num_test_points = 20,debug=False,fit_method=method)
	err_array = []
	for i in range(len(result.params)):
		f_array.append(result.params['param_'+str(i)].value)
		err_array.append(err_dict['param_'+str(i)])
	err_array=np.array(err_array)
	num_operations=len(params)
	#Translate delta values to g_e
	#Z = np.nansum(np.exp(-1.0*np.array(f_array[n:])))
	#g_e = np.exp(-1.0*np.array(f_array[n:]))
	q = q_values
	e = energies
	x_q = np.array(f_array[0:n])
	x_q[bad_q_i]=0
	g_e[bad_e_i]=0
	g_e[np.isnan(g_e)]=0
	x_q[np.isnan(x_q)]=0
	ge_int = np.trapz(g_e,x=energies)
	xq_err = err_array[0:n]
	delta_err = np.array(err_array[n:])
	ge_err = g_e*np.array(deltas)*(delta_err/np.array(deltas))

	x_q = np.array(x_q)
	xq_err = np.array(xq_err)
	g_e = np.array(g_e)
	ge_err = np.array(ge_err)
	#Now convert X(Q) into S(Q)
	r0 = 0.5391
	g=g_factor
	q_FF,magFF = get_MANTID_magFF(q,mag_ion)
	magFFsqr = 1.0/np.array(magFF)
	s_q = (2.0*x_q)/(r0**2 * g**2 * magFFsqr)
	s_q_err = (2.0*xq_err)/(r0**2 * g**2 * magFFsqr)
	if plot_result==True:
		try:
			plt.figure()
			plt.errorbar(q,s_q*magFFsqr,yerr=s_q_err*magFFsqr,capsize=3,ls=' ',mfc='w',mec='k',color='k',marker='',label=r'S(Q)|M(Q)|$^2$')
			plt.errorbar(q,s_q,yerr=xq_err,capsize=3,ls=' ',mfc='w',mec='b',color='b',marker='',label=r'S(Q)')
			plt.legend()
			plt.title('S(Q) Factorization Result')
			plt.xlabel('Q($\AA^{-1}$))')
			plt.ylabel('S(Q) barn/mol/f.u.')
			plt.figure()
			plt.title('G($\omega$) Factorization Result')
			plt.xlabel('$\hbar$\omega (meV)')
			plt.ylabel('G($\omega$) (1/meV)')
			plt.errorbar(energies,g_e,yerr=ge_err,capsize=3,ls=' ',mfc='w',mec='k',color='k',marker='')
		except Exception as e:
			print('Error when trying to plot result:')
			print(e)
	#Finally, save result

	return q,s_q,s_q_err,e,g_e,ge_err



def calculate_param_uncertainty(obs_vals,obs_errs,model,params,result,independent_vars=False,fast_calc=False,extrapolate=False,show_plots=True,fname='test.jpg',overwrite_prev=False,num_test_points = 30,debug=False,fit_method='powell'):
	'''
	This is a function to calculate the uncertainties in the free parameters of any lmfit model.
	It assumes a parabolic form, and takes the following arguments:
		obs_vals- np array - experimental values of the function
		obs_errs- np array - experimental errors of the function
		model - lmfit model describing the fitting function
		params - lmfit parameter object
		result - lmfit results of the best fit for the function
		indpendent_vars- bool or dict - if the model requires indepenedent vars, they are included here in the form a dictionary (i.e. independent_vars={'x':x,'y',y} in the function call)
		fast_calc - bool - spits out values if just testing and don't need to evaluate perfectly.
		extrapolate -bool- assumes a parabolic form for chisqr and gets uncertainties based around that. 
		show_plots - bool - shows the parabolic fits or the raw calculations
		fname - string- filename to store results 
		overwrite_prev- bool - determines if previous results should be loaded from file or overwritten. 

	Returns a dictionary of param names and errors
	'''
	#First step is to check if the parameters already exist.
	if os.path.isfile(fname) and overwrite_prev==True:
		os.remove(fname)
	if os.path.isfile(fname) and overwrite_prev!=True:
		errors=np.load(fname,allow_pickle=True).item()
		return errors
	if fast_calc==True:
		errs={}
		for param in result.params:
			errs[param]=result.params[param].value*0.2
		return errs 
	#Get an initial value of chisqr
	obs_errs[obs_errs==0]=np.nan 
	obs_errs[np.isnan(obs_errs)]=np.inf
	obs_vals[obs_vals==0]=0
	obs_vals[np.isnan(obs_vals)]=0
	chisqr0=calc_chisqr_val(obs_vals,result.best_fit,obs_errs)
	#Get the number of free parameters in the fit:
	num_free_params=0
	for param in result.params:
		if result.params[param].vary==True:
			num_free_params+=1
	#Get the number of points being fit
	num_points = np.nansum([obs_errs<1e10])
	#Calculate the statistical min and max allowed values of chisqr
	chisqrmin=chisqr0/(1.0+1.0/(num_points-num_free_params))
	chisqrmax=chisqr0/(1.0-1.0/(num_points-num_free_params))
	if debug==True:
		print('Chisqr0='+str(chisqr0))
		print('Chisqrmax='+str(chisqrmax))
	#Calculated errors will be returned in this dictionary:
	err_out = {}
	#Evaluated points will be kept track of in a param val list and a chisqr list

	progress = ProgressBar(num_free_params, fmt=ProgressBar.FULL) #I like a progress bar
	#Now we run into the problem of which points to test. This needs to be an adaptive process. 
	'''
	General algorithm for which points to test is the following:
	1. Set a min/max param value based on a percent error, noting that near zero a pure percent error will fail. 
	2. Test each of these points. If they are above the max value, then reduce the percent error by a factor of two. 
			If they are below the max value, then increase the percent error by a factor of two. 
			This should result in wide adaptability for ranges of fit parameters. 
	3. Once the two border points have been found (the points directly above and below the max chisqr)
		see if the number of evaluated points is acceptable. If not, fill in the parameter space between the points 
		uniformly to find an acceptable number of points. 
	4. Once a suitable number of points have been evaluated, perform a parabolic fit to determine the error bar if desired.
	5. If the parabolic fit is not performed, then the error bar is taken as half the distance between the min/max points.

	'''
	#Iterate through each parameter to do this 
	weights = 1.0 / obs_errs
	prev_slope = False
	for param in result.params:
		affect_chisqr=True
		try:
			if result.params[param].vary==True:
				print('Evaluating Uncertainty for '+str(param))
				found_min = False 
				found_max = False
				min_i = 0.0
				max_i = 0.0
				#Evaluated points will be kept track of in a param val list and a chisqr list
				param_list=[]
				chisqr_list = []
				init_param_val = result.params[param].value
				if debug==True:
					print('Init param val')
					print(init_param_val)
				#param_list.append(init_param_val)
				#chisqr_list.append(chisqr0)
				opt_val = init_param_val
				new_min_chisqr_prev = 0.0
				while found_min==False and min_i<1e2:
					new_params_min=result.params.copy()
					min_param_val = init_param_val - np.abs(init_param_val*(2.0**min_i)*0.005) #Start with a small 0.5% error 
					new_params_min.add(param,vary=False,value=min_param_val)
					if type(independent_vars)==bool:
						new_result_min=model.fit(obs_vals,params=new_params_min,nan_policy='omit',method='powell',weights=weights)
					else:
						new_result_min=model.fit(obs_vals,params=new_params_min,nan_policy='omit',method='powell',weights=weights,**independent_vars)
					#Get the chisqr after fitting the new param
					new_min_chisqr = calc_chisqr_val(obs_vals,new_result_min.best_fit,obs_errs)
					if new_min_chisqr>chisqrmax:
						found_min=True #We are free
					if new_min_chisqr<new_min_chisqr_prev:
						#Jumped out of a local minima, need to break. Do not append these points to the array
						found_min=True
					else:
						param_list.append(min_param_val)
						chisqr_list.append(new_min_chisqr)

						min_i = min_i+1.0
						if debug==True: 
							print(min_i)
							print('Curr chisqr: '+str(new_min_chisqr))
							print('Max chisqr: '+str(chisqrmax))
						if new_min_chisqr==new_min_chisqr_prev and min_i>4:
							print('Param '+str(param)+' does not affect chisqr.')
							found_min=True
							affect_chisqr=False
						new_min_chisqr_prev=new_min_chisqr
				new_max_chisqr_prev=0.0
				while found_max==False and max_i<1e2:
					new_params_min=result.params.copy()
					max_param_val = init_param_val + np.abs(init_param_val*(2.0**max_i)*0.005)
					new_params_max=result.params.copy()
					new_params_max.add(param,vary=False,value=max_param_val)
					if type(independent_vars)==bool:
						new_result_max=model.fit(obs_vals,params=new_params_max,nan_policy='omit',method='powell',weights=weights)
					else:
						new_result_max=model.fit(obs_vals,params=new_params_max,nan_policy='omit',method='powell',weights=weights,**independent_vars)            
					#Get the chisqr after fitting the new param
					new_max_chisqr = calc_chisqr_val(obs_vals,new_result_max.best_fit,obs_errs)
					if new_max_chisqr>chisqrmax:
						found_max=True 
					if new_max_chisqr<new_max_chisqr_prev:
						#Jumped out of local minima, function is not well behaved.
						found_max=True
					else:
						param_list.append(max_param_val)
						chisqr_list.append(new_max_chisqr)

						if new_max_chisqr==new_max_chisqr_prev and max_i>4:
							print('Param '+str(param)+' does not affect chisqr.')
							affect_chisqr=False
							found_max=True
						max_i = max_i+1
						if debug==True: 
							print(max_i)
							print('Curr chisqr: '+str(new_max_chisqr))
							print('Max chisqr: '+str(chisqrmax))
						new_max_chisqr_prev=new_max_chisqr
				if found_min==False or found_max==False:
					print('WARNING- strange behavior in uncertainty calculation, enbable show_plots==True to assure correctness')
				#Supposedly they have both been found. Check the number of points. if the estimate was initially way too large, then we need to fill them in . 
				num_eval_points = np.sum([np.array(chisqr_list)<chisqrmax])
				max_point = np.max(param_list)
				min_point = np.min(param_list)
				if debug==True:
					print('Max Point: ')
					print(max_point)
					print('Min point: ')
					print(min_point)
					print('Num eval points:')
					print(num_eval_points)
				while num_eval_points<num_test_points:
					#Evaluate the remainder of points in an evenly spaced fashion
					num_new_points = int(1.5*(num_test_points-num_eval_points))
					#fill_points = np.linspace(min_point,max_point,num_new_points)
					fill_points = np.random.uniform(low=min_point,high=max_point,size=num_new_points)
					for param_val in fill_points:
						new_params=result.params.copy()
						new_params.add(param,vary=False,value=param_val)
						if type(independent_vars)==bool:
							new_result=model.fit(obs_vals,params=new_params,nan_policy='omit',method='powell',weights=weights)
						else:
							new_result=model.fit(obs_vals,params=new_params,nan_policy='omit',method='powell',weights=weights,**independent_vars)            
						#Get the chisqr after fitting the new param
						new_chisqr = calc_chisqr_val(obs_vals,new_result.best_fit,obs_errs)
						param_list.append(param_val)
						chisqr_list.append(new_chisqr)
					#number of ponts below chisqrmax is the num_eval points
					num_eval_points = np.sum([np.array(chisqr_list)<chisqrmax])
					#num_eval_points == 0 or 1 is a special case. 
					good_param_vals_i = np.where(np.array(chisqr_list)<chisqrmax)[0]
					good_param_vals = np.array(param_list)[good_param_vals_i]
					if num_eval_points>1:
						max_point = np.max(good_param_vals)
						min_point = np.min(good_param_vals)
					else:
						minus_points_i = [param_list<opt_val]
						minus_points = np.array(param_list)[minus_points_i]
						min_point = np.max(minus_points)
						plus_points = np.array(param_list)[param_list>opt_val]
						max_point = np.max(plus_points)
						if debug==True:
							print('new min point')
							print(min_point)
							print('new max point')
							print(max_point)
					if num_eval_points<num_test_points:
						print('Insufficient number of points under max chisqr. Recursively iterating.')
						print('Good points: '+str(num_eval_points)+'/'+str(num_test_points))
				#In theory should have all points needed to get uncertainties now.
				chisqr_list = np.array(chisqr_list)
				param_list = np.array(param_list) 

				#If there are new points that have a lower chisqr than the initial fit, 
				#   adjust the maxchisqr now or the error will be artificially large.
				min_eval_chisqr =  param_list[np.argmin(chisqr_list)]
				if min_eval_chisqr<chisqr0-np.abs(chisqr0-chisqrmax)/2.0:
					print('WARNING: Local minima found that is different from initial value.')
					print('Enable show_plots=True and check quality of initial fit.')
					opt_val = param_list[np.argmin(chisqr_list)]
					opt_chisqr  = np.mean([chisqr0,np.min(chisqr_list)])
				else:
					opt_chisqr = chisqr0 
					opt_val = init_param_val 
				temp_chisqrmin=opt_chisqr/(1.0+1.0/(num_points-num_free_params))
				temp_chisqrmax=opt_chisqr/(1.0-1.0/(num_points-num_free_params))  
				if extrapolate==True:
					def parabola(x,a,b,c):
						return a*((x-b)**2)+c
					para_model = Model(parabola)
					para_params = para_model.make_params()
					#very roughly assume it's linear between 0 and 1 points
					if prev_slope==False:
						guess_slope=np.abs(temp_chisqrmax-chisqr0)/(((np.nanmax(np.array(param_list))-opt_val)**2))
					else:
						guess_slope = prev_slope
					para_params.add('a',value=guess_slope,min=0,max=1e8)
					para_params.add('b',vary=False,value=opt_val) #Should just be hte optimum value
					para_params.add('c',vary=True,value=chisqr0,min=chisqr0-0.1,max=chisqrmax)
					para_weights = np.ones(len(chisqr_list))
					#weight by distance from optimum value?
					para_weights = np.exp(-1.0*(np.abs(param_list-opt_val))/np.abs(opt_val))
					para_fit = para_model.fit(chisqr_list,x=param_list,params=para_params,method='powell',weights=para_weights)
					a_fit = para_fit.params['a'].value
					b_fit = para_fit.params['b'].value
					c_fit = para_fit.params['c'].value
					prev_slope = a_fit
					max_param_val = np.sqrt(np.abs((temp_chisqrmax-c_fit)/a_fit))+b_fit
					#print('Max param_val='+str(max_param_val))
					error = np.abs(max_param_val-b_fit)
					opt_val=b_fit
					eval_range = np.linspace(opt_val-error*1.2,opt_val+error*1.2,3000)
					fit_eval = para_model.eval(x=eval_range,params=para_fit.params)
					if affect_chisqr==True:
						err_out[param]=error
					else:
						err_out[param]=np.nanmean(param_list)/np.sqrt(len(param_list))
				else:
					good_param_val_i = [np.array(chisqr_list)<temp_chisqrmax]
					good_param_vals = param_list[good_param_val_i]
					error = np.abs(np.max(good_param_vals)-np.min(good_param_vals))/2.0
					if affect_chisqr==True:
						err_out[param]=error
					else:
						err_out[param]=np.nanmean(param_list)/np.sqrt(len(param_list))            
				if show_plots==True:
					try:
						plt.figure()
						plt.plot(param_list,chisqr_list,color='k',marker='o',ls='')
						plt.xlabel('Param val')
						plt.ylabel('Chisqr')
						plt.title('Uncertainty '+str(param)+ ' Error ='+str(round(error,3))+' or '+str(round(100*error/opt_val,3))+'%')
						#plt.xlim(np.min(test_value_arr)-np.abs(np.min(test_value_arr))/10.0,np.max(test_value_arr)*1.1)
						plt.plot(np.linspace(np.min(param_list),np.max(param_list)+1e-9,10),np.ones(10)*np.abs(temp_chisqrmax),'r')
						if extrapolate==True:
							plt.plot(eval_range,fit_eval,'b--')
						plt.plot(opt_val+error,temp_chisqrmax,'g^')
						plt.plot(opt_val-error,temp_chisqrmax,'g^')
						plt.ylim(np.min(chisqr_list)-np.abs(np.abs(temp_chisqrmax)-np.min(chisqr_list))/3.0,np.abs(temp_chisqrmax)+np.abs(np.abs(temp_chisqrmax)-np.min(chisqr_list))/3.0)
						#plt.xlim(0.9*np.min(test_value_arr),np.max(test_value_arr)*1.1)
						plt.show()
					except Exception as e:
						print('Some error while plotting.')
						print(e)
					#plt.ylim(0.0,1.3*np.abs(chisqrmax))
				progress.current+=1
				progress()
			else:
				err_out[param]=0.0
		except Exception as e:
			err_out[param]=0
			print('Warning: Error when evaluating uncertainty. ')
			tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
			print(tb_str)
	#Save to a file
	np.save(fname,err_out)
	return err_out

def factorization_f(n,m,delE,**vals):
	#Returns flattened calculated spectra from factorization
	vals_arr = []
	for i in range(len(vals)):
		vals_arr.append(vals['param_'+str(i)])
	vals = np.array(vals_arr)
	Q_vals = vals[0:n].reshape(n,1) #Reconstruct y coordinates
	#Update to this function means that these are delta vals in e^(-delta_i)/Z
	deltas = vals[n:]
	Z = np.nansum(np.exp(-1.0*deltas))
	E_vals = np.exp(-1.0*deltas)/(delE*Z) 

	E_vals = E_vals.reshape(1,m) # '' x coord

	slice2D = Q_vals * E_vals

	calcI = slice2D.flatten()
	#chisqr = np.nansum((z_fit - calcI)**2 / (meas_errs**2))/num_points
	return calcI
def uncertainty_determination(fit_params,z_test,obs_errs,model,result,independent_var_arr=False,fast_mode=False,extrapolate=False,show_plots=False,fname='placeholder.jpg',overwrite_prev=False,num_chisqr_test_points=30):
	#Does a rocking curve foreach parameter to determine its uncertainty
	#Assumes that z is already flatted into a 1D array
	print('WARNING: this call is outdated. Use THfuncs.calculate_param_uncertainty instead.')
	errs = calculate_param_uncertainty(z_test,1.0 / obs_errs,model,fit_params,result,independent_vars=independent_var_arr,fast_calc=fast_mode,extrapolate=extrapolate,\
		show_plots=show_plots,fname=fname,overwrite_prev=overwrite_prev,num_test_points = num_chisqr_test_points)
	err_array=[]
	for key in errs.keys():
		err_array.append(errs[key])
	return err_array

def get_MANTID_magFF(q,mag_ion):
	#Given a str returns a simple array of the mantid defined form factor. basically a shortcut for the mantid version
	cw = CreateWorkspace(DataX = q,DataY = np.ones(len(q)))
	cw.getAxis(0).setUnit('MomentumTransfer')
	ws_corr = MagFormFactorCorrection(cw,IonName=mag_ion,FormFactorWorkspace='FF')
	FFcorrection = np.array(ws_corr[0].readY(0))
	return q,FFcorrection
def calc_chisqr_val(obs_arr,theory_arr,obs_err_arr,theory_err_arr=0.0):
	#For arbitrary arrays of theory, experiment, errors, returns the chisqr statistic.
	#Returns it for each point I guess. 
	N = np.nansum([obs_err_arr<1e9])
	chisqr = 0
	obs_arr = np.array(obs_arr)
	theory_arr = np.array(theory_arr)
	obs_err_arr = np.array(obs_err_arr)
	diffsqr = (obs_arr - theory_arr)**2
	chisqr_arr = diffsqr / obs_err_arr**2
	chisqr = np.nansum(chisqr_arr) / N
	return chisqr

class ProgressBar(object):
	DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
	FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

	def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
				 output=sys.stderr):
		assert len(symbol) == 1

		self.total = total
		self.width = width
		self.symbol = symbol
		self.output = output
		self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
			r'\g<name>%dd' % len(str(total)), fmt)

		self.current = 0

	def __call__(self):
		percent = self.current / float(self.total)
		size = int(self.width * percent)
		remaining = self.total - self.current
		bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

		args = {
			'total': self.total,
			'bar': bar,
			'current': self.current,
			'percent': percent * 100,
			'remaining': remaining}
		print('\r' + self.fmt % args,file=self.output, end='')

	def done(self):
		self.current = self.total
		self()
		print('', file=self.output)