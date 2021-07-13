import argparse
from tensorflow.keras import backend as K
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from Utils.Preprocess import *
from Metrics.ImageMetrics import GramMatrix

def save_grp_samples(images, groups, scores, output_dir, nsamples=20, method="random", out_format=".jpg"):
	"""
	given images ,their group prediction and the score of their clustering
	saves nsamples of the best image per group in the specified output directory
	the method can be "best","worst",or "random" and defines the way we sample images
	the out_format can be ".jpg" or ".npy" and is
	"""
	assert len(images)==len(groups) and len(groups)==len(scores), "images, groups, and scores should all be the same length"

	#get list of group values and for each value sort image by best accuracy
	groups_id = np.unique(groups)
	for grp_id in groups_id:
		values = [(x,y) for x, y in zip(images[groups==grp_id], scores[groups==grp_id])]
		values.sort(key=lambda x:x[1], reverse=True) #sort best first
		#select images index
		sample_size = min(len(values),nsamples)	#if the group does not have enough images
		if method=="random":
			choices = np.random.choice(len(values), size=sample_size, replace=False)
		elif method=="best":
			choices = [i for i in range(sample_size)]
		elif method=="worst":
			choices = [i for i in range(len(values)-sample_size, len(values))]
		else:
			raise ValueError("invalid method {}".format(method))
		#save images
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)
		if out_format==".jpg":
			for i, idx in enumerate(choices):
				img, score = values[idx]
				plt.imsave(os.path.join(output_dir, 'grp{}_{}_{:.3f}.jpg'.format(grp_id, i, score)), img)
		elif out_format==".npy":
			filename = "grp{}_{}{}_images.npy".format(grp_id, nsamples, method)
			data = np.array([values[c][0] for c in choices])
			np.save(os.path.join(output_dir, filename), data)
		else:
			raise ValueError("invalid format {}".format(format))


def normalize(ldf):
	assert ldf.ndim==2, "latent given should be 2dims, here {}".format(ldf.ndim)
	#normalize and standardize per feature
	std = np.std(ldf)
	mean = np.mean(ldf)
	ldf = (ldf-mean)/std
	mini = np.min(ldf)
	maxi = np.max(ldf)
	ldf = (ldf-mini)/(maxi-mini)
	return ldf

def do_pca(data, n_pca):
	pca = PCA(n_components=n_pca)
	data = pca.fit_transform(data)
	return data

def do_MDS(data, n_components):
	embedding = MDS(n_components=n_components)
	fit_data = embedding.fit_transform(data)
	return fit_data

def show_3D(data, groups=None, title=""):
	x = data[:, 0]
	y = data[:, 1]
	z = data[:, 2]

	# axes instance
	fig = plt.figure(figsize=(6,6))
	fig.suptitle(title)
	ax = Axes3D(fig)

	# get colormap from seaborn
	cmap = ListedColormap(sns.color_palette("tab10", 256).as_hex())
	# plot
	if groups is not None:
		sc = ax.scatter(x, y, z, s=40, c=groups, marker='o', cmap=cmap, alpha=0.8)
	else:
		sc = ax.scatter(x, y, z, s=40, marker='o', cmap=cmap, alpha=0.8)
	ax.set_xlabel("C1")
	ax.set_ylabel("C2")
	ax.set_zlabel("C3")

	plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
	plt.show()
	plt.close()

def show_by_class(images, groups, scores, title="", output_dir=None, nrows=6):
	groups_id = np.unique(groups)
	f , axs = plt.subplots(nrows=nrows, ncols=len(groups_id), squeeze=False)
	f.suptitle(title)
	for j, grp_id in enumerate(groups_id):
		axs[0, j].set_title("grp {}".format(grp_id))
		values = [(x,y) for x, y in zip(images[groups==grp_id], scores[groups==grp_id])]
		values.sort(key=lambda x:x[1], reverse=True)
		for i, pair in enumerate([*values[:nrows-1], values[-1]]):
			axs[i, j].imshow(pair[0], cmap="gray")
			axs[i, j].axis('off')
			axs[i, j].annotate("{:.2f}".format(pair[1]), xy=(0, 0),  xycoords='data',
            xytext=(0, 0), textcoords='axes fraction', fontsize="x-small",
            horizontalalignment='left', verticalalignment='bottom')

	if output_dir:
		# plt.tight_layout()
		plt.savefig(os.path.join(output_dir, "grouping_{}".format(title)))
		# plt.show()
		plt.close()


def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:min(len(lst),i + n)]

def get_df(model, test, list_layers, output_dir):
	print(model.summary())

	#get deep features from specified layers
	deep_features=[]
	for layer in model.layers:
		if layer.name in list_layers:
			generator = chunks(test, 50)
			ldf = K.function([model.input], layer.output)([next(generator), 1])
			for batch in generator:
				bdf = K.function([model.input], layer.output)([batch, 1])
				ldf = np.concatenate((ldf, bdf), axis=0)
			if ldf.ndim!=2: #if layer is conv
				ldf = np.mean(ldf, axis=(1,2))
				# ldf = np.array([GramMatrix(img) for img in ldf])
			deep_features.append(np.reshape(ldf, (ldf.shape[0], -1)))
			#np.save(os.path.join(output_dir,"df_hab_{}.npy".format(layer.name)), ldf)
	return deep_features

def get_AE_df(model, test, list_layers, output_dir):
	print(model.summary())
	deep_features=[]
	deep_features += get_df(model.encoder, test, list_layers, output_dir)
	deep_features += get_df(model.decoder, test, list_layers, output_dir)
	return deep_features

def do_kmeans(data, img, color_img, k_grps, output_dir, title, palette_size, save_clusters, show3D=False):
	list_avg=[]

	for k in k_grps:
		print("Clustering k: ",k)

		clusterer = KMeans(n_clusters=k, random_state=10)
		cluster_labels = clusterer.fit_predict(data)
		np.save(os.path.join(output_dir,"kpreds_{}_{}".format(k, title)), cluster_labels)

		if show3D:
			show_3D(data, groups=cluster_labels, title="{} PCA 3dim with {} clusters".format(title, k))


		silhouette_avg = silhouette_score(data, cluster_labels)
		list_avg.append(silhouette_avg)
		print("for n_clusters= ", k, " The average silhouette_score is :", silhouette_avg)

		sample_silhouette_values = silhouette_samples(data, cluster_labels)
		show_by_class(color_img, groups=cluster_labels, scores=sample_silhouette_values, title="{}_{}clusters".format(title, k), output_dir=output_dir)

		# save a palette of the n_to_save best clusturised colored images
		palette_size=450-k*50 	#so 3k:300 , 6k:150
		if palette_size:
			save_grp_samples(color_img, cluster_labels, sample_silhouette_values,
							output_dir=os.path.join(output_dir, "Palette_{}_{}clusters".format(title, k)),
							nsamples=palette_size,
							method="best", out_format=".jpg")
		# save a numpy array of the n_to_save best clusterised colored images
		if save_clusters:
			save_grp_samples(img, cluster_labels, sample_silhouette_values,
							output_dir=os.path.join(output_dir, "Palette_{}_{}clusters".format(title, k)),
							nsamples=palette_size,
							method="best", out_format=".npy")

		#plot silhouettes
		y_lower = 10
		fig, ax1 = plt.subplots(1,1)
		ax1.set_xlim([-0.1, 1])
		ax1.set_ylim([0, len(data) + (k+1)*10])
		for i in range(k):
			print(i,"/",k)
			#aggregate silhouette scores for samples belonging to the cluster i and sort them
			ith_cluster_sil_val = sample_silhouette_values[cluster_labels==i]
			ith_cluster_sil_val.sort()
			size_cluster_i = ith_cluster_sil_val.shape[0]
			y_upper = y_lower + size_cluster_i

			ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_val, alpha=0.7)
			ax1.text(-0.05, y_lower+0.05*size_cluster_i, str(i))
			ax1.text(1.05, y_lower+0.05*size_cluster_i, len(ith_cluster_sil_val))

			y_lower=y_upper+10
		ax1.set_title("silhouette plot for various clusters, k {}, {}".format(k, title))
		ax1.set_xlabel("silhouette coefficient values")
		ax1.set_ylabel("Cluster Label")

		ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
		ax1.set_yticks([])
		ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


		plt.savefig(os.path.join(output_dir, "cluster {} silhouette {}".format(k, title)))
		# plt.show()
		plt.close()

	fig2, ax2 = plt.subplots(1,1)
	ax2.set_title("mean silhouette for each number of cluster")
	ax2.set_xlabel("kluster")
	ax2.set_ylabel("accuracy")
	ax2.plot(k_grps, list_avg)
	plt.savefig(os.path.join(output_dir, "silhouette per cluster group "+title))
	# plt.show()
	plt.close()

if __name__=="__main__":
	DEFAULT_MODEL="vgg16"
	DEFAULT_K=[2,3,4,5,6,7,8]
	#parsing parameters
	parser = argparse.ArgumentParser(description="Clusterize given numpy images using network deep features and k means")
	parser.add_argument("input", help="path of the file to open (.npy)")
	parser.add_argument("output_dir", help="path of the file to save")
	parser.add_argument("-m", "--model", choices=["vgg16", "places", "hybrid", "AEconvo"], default=DEFAULT_MODEL, help="network model to use")
	parser.add_argument("-r", "--reduction", choices=["pca", "mds"], default=None, help="dimension reduction method to use")
	parser.add_argument("-d", "--dim", type=int, default=None, help="number of dim for the PCA, default None performs no PCA")
	parser.add_argument("-o", "--optional", help="add the model path here")
	parser.add_argument("-k", "--klusters", nargs="+", type=int, default=DEFAULT_K, help="kernels to use for clusterisation, default {}".format(DEFAULT_K))
	parser.add_argument("-p", "--palette_size", type=int, default=0, help="number of images to save in a palette")
	parser.add_argument("--save_clusters", default=False, action="store_true", help="add this argument if you want to save a numpy of the predicted clusters")

	args = parser.parse_args()
	n_pca=args.dim

	#prepare images
	test = np.load(args.input)
	print("test shape: {}".format(test.shape))

	#select model and extract deep features
	layers_to_extract = ["block1_pool", "block2_pool", "block3_pool", "fc1", "fc2"]
	if args.model=="vgg16":
		from tensorflow.keras.applications.vgg16 import VGG16
		model = VGG16(weights='imagenet', include_top=True)
		deep_features = get_df(model, test, layers_to_extract, args.output_dir)
	elif args.model=="places":
		from AutoEncoders.VGG16Places.vgg16_places_365 import VGG16_Places365
		model = VGG16_Places365(weights='places', include_top=True)
		deep_features = get_df(model, test, layers_to_extract, args.output_dir)
	elif args.model=="hybrid":
		from AutoEncoders.VGG16Places.vgg16_hybrid_places_1365 import VGG16_Hybrid_1365
		model = VGG16_Hybrid_1365(weights='places', include_top=True)
		deep_features = get_df(model, test, layers_to_extract, args.output_dir)
	elif args.model=="AEconvo":
		layers_to_extract = ["max_pooling2d","max_pooling2d_1","max_pooling2d_2","last_pool_encoder"]#, "output_encoder"]
		model = keras.models.load_model(args.optional, compile=False)
		deep_features = get_AE_df(model, test, layers_to_extract, args.output_dir)

	#add a concatenation of features
	# layers_to_extract.append("01_pools")
	# deep_features.append(np.concatenate([x for x in deep_features[0:2]], axis=-1))

	# layers_to_extract.append("12_pools")
	# deep_features.append(np.concatenate([x for x in deep_features[1:3]], axis=-1))

	# layers_to_extract.append("012_pools")
	# deep_features.append(np.concatenate([x for x in deep_features[0:3]], axis=-1))

	# layers_to_extract.append("all_pools")
	# deep_features.append(np.concatenate([x for x in deep_features[0:4]], axis=-1))

	layers_to_extract.append("23_pools")
	deep_features.append(np.concatenate([x for x in deep_features[2:4]], axis=-1))

	#retrieve only all pools concatenated
	# deep_features=[np.concatenate([x for x in deep_features[2:4]], axis=-1)]
	# layers_to_extract = ["23_pools"]



	deep_features = [normalize(df) for df in deep_features]
	print([x.shape for x in deep_features])

	title_end = ["","_{}{}".format(args.reduction,n_pca)][(n_pca is not None and args.reduction is not None)]

	for name_layer, df in zip(layers_to_extract, deep_features):
		df = df.reshape((len(df), -1))
		#do dimension reduction
		if args.reduction=="pca":
			df = do_pca(df, n_pca)
		elif args.reduction=="mds":
			df = do_MDS(df, n_pca)

		#clusterize
		title = "{}{}".format(name_layer, title_end)
		color_img = np.load(os.path.join(os.path.dirname(args.input) , "color_reference.npy"))
		do_kmeans(df, test, color_img, args.klusters, args.output_dir, title=title, palette_size=args.palette_size, save_clusters=args.save_clusters ,show3D=n_pca==3)


	# df3 = np.concatenate((df1, df2), axis=-1)
