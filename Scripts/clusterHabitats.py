import argparse
from tensorflow.keras import backend as K
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

from Utils.Preprocess import *
from Metrics.ImageMetrics import GramMatrix


def do_pca(data, n_pca):
	pca = PCA(n_components=n_pca)
	data = pca.fit_transform(data)
	return data

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


def load_test(list_files, output=None):
	if len(list_files)==1 and list_files[0].endswith(".npy"):
		test = np.load(list_files[0])
	else:
		pr = Preprocess((224, 224), normalize=True, standardize=True, img_type=IMG.RGB, img_channel=CHANNEL.ALL)
		test = np.array([pr(path) for path in list_files])
		if output:
			np.save(os.path.join(output, "habitats.npy"), test)
		else:
			np.save("habitats.npy", test)
	return test


def show_by_class(data, groups, title="", output_dir=None, nrows=5):
	print(groups.shape, data.shape)
	groups_id = np.unique(groups)
	f , axs = plt.subplots(nrows=nrows, ncols=len(groups_id), squeeze=False)
	f.suptitle(title)
	for j, grp_id in enumerate(groups_id):
		grp = data[groups==grp_id]
		axs[0, j].set_title("group {}".format(grp_id))
		np.random.shuffle(grp)
		for i in range(min(nrows, len(grp))):
			axs[i, j].imshow(grp[i])
			axs[i, j].axis('off')
	if output_dir:
		plt.savefig(os.path.join(output_dir, "grouping_{}".format(title)))
		# plt.show()
		plt.close()
		# annot_data = np.array([ cv2.putText(img=np.copy(img), text=str(groups[idx_img]), org=(10,10),fontFace=3, fontScale=10, color=(0,0,255), thickness=5)
		# 						for idx_img, img in enumerate(data)])
		# np.save(os.path.join(output_dir, "annotedpreds_{}".format(title)), annot_data)

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
			#normalize and standardize per feature
			std = np.std(ldf, axis=-1)[...,None]
			mean = np.mean(ldf, axis=-1)[...,None]
			ldf = (ldf-mean)/std
			mini = np.min(ldf, axis=-1)[...,None]
			maxi = np.max(ldf, axis=-1)[...,None]
			ldf = (ldf-mini)/(maxi-mini)
			if ldf.ndim!=2: #conv
				ldf = np.array([GramMatrix(img.mean(axis=(0,1))[..., np.newaxis]) for img in ldf])
			deep_features.append(ldf)
			#np.save(os.path.join(output_dir,"df_hab_{}.npy".format(layer.name)), ldf)
	return deep_features


def do_kmeans(data, test, k_grps, output_dir, title, show=False):
	list_avg=[]

	for k in k_grps:
		print("Clustering k: ",k)

		clusterer = KMeans(n_clusters=k, random_state=10)
		cluster_labels = clusterer.fit_predict(data)
		np.save(os.path.join(output_dir,"kpreds_{}_{}".format(k, title)), cluster_labels)

		if show:
			show_3D(data, groups=cluster_labels, title="{} PCA 3dim with {} clusters".format(title, k))

		show_by_class(test, groups=cluster_labels, title="{}_{}clusters".format(title, k), output_dir=output_dir)

		silhouette_avg = silhouette_score(data, cluster_labels)
		list_avg.append(silhouette_avg)
		print("for n_clusters= ", k, " The average silhouette_score is :", silhouette_avg)

		sample_silhouette_values = silhouette_samples(data, cluster_labels)
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
	#parsing parameters
	parser = argparse.ArgumentParser(description="output the deep features of VGG16")
	parser.add_argument("glob_input", help="path of the file to open")
	parser.add_argument("output_dir", help="path of the file to save")
	parser.add_argument("-m", "--model", choices=["vgg16", "places", "hybrid"], default=DEFAULT_MODEL, help="network model to use")
	parser.add_argument("-d", "--dim", type=int, default=None, help="number of dim for the PCA, default None performs no PCA")
	args = parser.parse_args()
	n_pca=args.dim

	#prepare images
	test = load_test(glob(args.glob_input), output=args.output_dir)
	print("test shape: {}".format(test.shape))

	#select model
	if args.model=="vgg16":
		from tensorflow.keras.applications.vgg16 import VGG16
		model = VGG16(weights='imagenet', include_top=True)
	elif args.model=="places":
		from AutoEncoders.VGG16Places.vgg16_places_365 import VGG16_Places365
		model = VGG16_Places365(weights='places', include_top=True)
	elif args.model=="hybrid":
		from AutoEncoders.VGG16Places.vgg16_hybrid_places_1365 import VGG16_Hybrid_1365
		model = VGG16_Hybrid_1365(weights='places', include_top=True)

	#extract deep features
	layers_to_extract = ["block1_pool", "block2_pool", "block3_pool", "fc1", "fc2"]
	deep_features = get_df(model, test, layers_to_extract, args.output_dir)
	print([x.shape for x in deep_features])

	title_end = ["","_pca{}".format(n_pca)][n_pca is not None]

	for name_layer, df in zip(layers_to_extract, deep_features):
		df = df.reshape((len(df), -1))
		if n_pca:
			#do PCA
			df = do_pca(df, n_pca)
		#clusterize
		title = "{}{}".format(name_layer, title_end)
		do_kmeans(df, test, [2,3,4,5,6,7,8], args.output_dir, title=title, show=n_pca==3)


	# df3 = np.concatenate((df1, df2), axis=-1)
