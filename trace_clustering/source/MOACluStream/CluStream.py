"""
This code is adapdated from the source code of moa, that can be found in the following address
https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/clusterers/clustream/WithKmeans.java
"""
from .Kernel import Kernel
import numpy as np
import pandas as pd
import itertools
from sklearn.cluster import KMeans
from scipy.spatial import distance
import copy

class MOACluStream:
	"""
	CluStream data stream clustering algorithm implementation

	Args:
		h (int): 	   Range of the window
		m (int): 	   Maximum number of micro kernels to use
		t (int): 	   Multiplier for the kernel radius
		k_macro (int): Number of macro clusters
	Attributes:
		kernels (list of clustream.Kernel) : microclusters
		macro_clusters (list of clustream.Kernel) : macroclusters
		time_window (int) : h
		m (int): m
		t (int): t
	"""
	def __init__(self,h=1000,m=100,t=2,k_macro=3):
		self.kernels = []
		self.time_window = h
		self.m = m
		self.t = t
		self.k_macro = k_macro
		self.nb_created_clusters = 0 
		self.previous_centers = None

	def __get_pair_pdist(self, number_ref, size):
		cnt = 0
		for i in range(size):
			for j in range(i + 1, size):
				if cnt == number_ref:
					return i, j
				cnt += 1

	def offline_cluster(self,datapoint,timestamp):
		"""
		offline clustering of a datapoint

		Args:
			datapoint (ndarray): single point from the dataset
			timestamp (int): timestamp of the datapoint
		"""
		datapoint_ = np.copy(datapoint)

		if len(self.kernels)!=self.m:
			#0. Initialize
			self.kernels.append(Kernel(
				self.nb_created_clusters, datapoint_, timestamp,self.t,self.m
			))
			self.nb_created_clusters += 1
			# print([kernel.center for kernel in self.kernels])
			return

		centers = [kernel.center for kernel in self.kernels] #TODO :faster computing with caching
		#print(timestamp, np.all([x <= 1 for x in centers]))

		distances_to_centers = distance.cdist([datapoint_], centers)[0]
		distances_argsort = np.argsort(distances_to_centers)

		closest_kernel_index = distances_argsort[0]
		min_distance = distances_to_centers[closest_kernel_index]

		#1. Determine closest kernel
		
		# - - - - - - - OLD - -  - - - - - 
 		# closest_kernel_index, min_distance = min(
		# 	((i,np.linalg.norm(center-datapoint_)) for i,center in enumerate(centers)),
		# 	key=lambda t:t[1]
		# )
		# - - - - - - - OLD - -  - - - - - 
		closest_kernel = self.kernels[closest_kernel_index]
		closet_kernel_center = centers[closest_kernel_index]

		# 2. Check whether instance fits into closest_kernel
		if closest_kernel.n==1:
			# Special case: estimate radius by determining the distance to the
			# next closest cluster
			
			# - - - - - - - OLD - -  - - - - - 
			# radius=min(( #distance between the 1st closest center and the 2nd
			# 	np.linalg.norm(center-closet_kernel_center) for center in centers if not center is closet_kernel_center
			# ))
			# - - - - - - - OLD - -  - - - - - 

			radius = np.min(
				distance.cdist(
					[closet_kernel_center], 
					[centers[i] for i in range(len(centers)) if i != distances_argsort[0]]
				)[0]
			)
		else:
			radius = closest_kernel.get_radius()

		if min_distance < radius:
			# Date fits, put into kernel and be happy
			closest_kernel.insert(datapoint_, timestamp)
			# print(f"{timestamp} fits")
			return "FITS_EXISTING_MICROCLUSTER"

		# 3. Date does not fit, we need to free
		# some space to insert a new kernel

		threshold = timestamp - self.time_window # Kernels before this can be forgotten
		# 3.1 Try to forget old kernels
		oldest_kernel_index=next((
			i for i,kernel in enumerate(self.kernels) if kernel.get_relevance_stamp() < threshold
		), None)
		if oldest_kernel_index!=None:
			# print(f"{timestamp} forgot old kernel")
			self.kernels[oldest_kernel_index] = Kernel(
				self.nb_created_clusters, datapoint_, timestamp, self.t, self.m
			)
			self.nb_created_clusters += 1
			return "FORGET_OLD_MICROCLUSTER"

		# 3.2 Merge closest two kernels
		# print(f"{timestamp} merge closest kernel")
		
		# - - - - - - - OLD - -  - - - - - 
		# combination_indexes=itertools.combinations(range(len(centers)),2)
		# closest_a, closest_b, _=min(
		# 	((i,j,np.linalg.norm(centers[j]-centers[i])) for i,j in combination_indexes),
		# 	key=lambda t:t[-1]
		# )
		# - - - - - - - OLD - -  - - - - - 

		argmin_pdist = np.argmin(distance.pdist(centers))
		closest_a, closest_b = self.__get_pair_pdist(argmin_pdist, len(centers))
		
		self.kernels[closest_a].add(self.kernels[closest_b])		
		self.kernels[closest_b] = Kernel(
			self.nb_created_clusters, datapoint_, timestamp, self.t, self.m
		)
		self.nb_created_clusters += 1

		return "MERGED_MICROCLUSTER"

	def merge_kernels(self, k1, k2):
		result = Kernel(0, 0, self.t, self.m)
		result.add(k1)
		result.add(k2)
		return result


	# # # # # 
    # MICRO #
    # # # # #
	def get_micro_clusters_ids(self):
		return copy.deepcopy([
			x.identifier for x in self.kernels
		])

	def get_micro_clusters_centers(self):
		return copy.deepcopy([
			x.get_center() for x in self.kernels
		])

	def get_micro_clusters_radius(self):
		return copy.deepcopy([
			x.get_radius() for x in self.kernels
		])

	def get_micro_clusters_weights(self):
		#total_weight = np.sum([x.n for x in self.kernels])
		total_weight = 1
		return copy.deepcopy([
			x.n/total_weight for x in self.kernels
		])

	# # # # # 
    # MACRO #
    # # # # #
	def update_macro_clusters(self):
		micro_centers = np.array([x.get_center() for x in self.kernels])
		
		if len(micro_centers) >= self.k_macro:
			if self.previous_centers is not None and len(self.previous_centers) == self.k_macro:
				macro_kmeans = KMeans(
					n_clusters=self.k_macro, 
					#precompute_distances=True,
					copy_x=True,
					random_state=42,
					init=self.previous_centers,
					n_init=1
					#n_jobs=-1
				)
			else:
				macro_kmeans = KMeans(
					n_clusters=self.k_macro, 
					#precompute_distances=True,
					copy_x=True,
					random_state=42,
					#n_jobs=-1
				)
			macro_labels = macro_kmeans.fit_predict(
				X=micro_centers, 
				y=None,
				sample_weight=self.get_micro_clusters_weights()
			)

			self.macro_labels = {
				self.kernels[i].identifier : macro_labels[i] for i in range(len(macro_labels))
			}
			self.macro_labels_values = macro_labels
		else:
			self.macro_labels = {self.kernels[i].identifier: 0 for i in range(len(self.kernels))}
			self.macro_labels_values = np.array([0 for x in self.kernels])


	def get_macro_clusters_centers(self):
		self.update_macro_clusters()

		micro = np.array([x.get_center() for x in self.kernels])
		macro_centers = pd.DataFrame(micro).groupby(self.macro_labels_values).mean()

		self.previous_centers = macro_centers

		return macro_centers.values

	def get_macro_clusters_radius(self):
		# self.update_macro_clusters()

		micro = self.get_micro_clusters_radius()
		macro_radius = pd.DataFrame([micro]).transpose().groupby(self.macro_labels_values).sum()
		return macro_radius.values.reshape(1, -1)[0]
		

	def get_macro_clusters_weights(self):		
		# self.update_macro_clusters()

		micro = self.get_micro_clusters_weights()

		macro_radius = pd.DataFrame([micro]).transpose().groupby(self.macro_labels_values).sum()
		return macro_radius.values.reshape(1, -1)[0]