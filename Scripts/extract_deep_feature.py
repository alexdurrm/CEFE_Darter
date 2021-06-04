import argparse
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


from Utils.Preprocess import *


def load_test(list_files, output=None):
	if len(list_files)==1 and list_files[0].endswith(".npy"):
		test = np.load(list_files[0])
	else:
		pr = Preprocess((224, 224), normalize=True, standardize=True, img_type=IMG.RGB, img_channel=CHANNEL.ALL)
		test = np.array([pr(path) for path in list_files])
	if output:
		np.save(os.path.join(output, "habitats.npy"), test)
	return test


def get_df(test, list_layers, output_dir):
	def chunks(lst, n):
	    """Yield successive n-sized chunks from lst."""
	    for i in range(0, len(lst), n):
	        yield lst[i:i + n]
	vgg = VGG16(weights='imagenet', include_top=True)
	print(vgg.summary())


	#get deep features from specified layers
	deep_features=[]
	for layer in vgg.layers:
		if layer.name in list_layers:
			ldf = K.function([vgg.input], layer.output)([next(chunks(test, 50)), 1])
			for batch in chunks(test, 50):
				bdf = K.function([vgg.input], layer.output)([batch, 1])
				ldf = np.concatenate((ldf, bdf), axis=0)
				print(ldf.shape)
			np.save(os.path.join(output_dir,"df_hab_{}.npy".format(layer.name)), ldf)
			deep_features.append(ldf)

	return deep_features


def do_kmeans(data, k_grps, output_dir, title):
	list_avg=[]
	data = data.reshape((len(data), -1))
	for k in k_grps:
		print("Clustering k: ",k)
		fig, ax1 = plt.subplots(1,1)
		ax1.set_xlim([-0.1, 1])
		ax1.set_ylim([0, len(data) + (k+1)*10])

		clusterer = KMeans(n_clusters=k, random_state=10)
		cluster_labels = clusterer.fit_predict(data)
		np.save("kpreds_{}_{}".format(k, title), cluster_labels)

		silhouette_avg = silhouette_score(data, cluster_labels)
		list_avg.append(silhouette_avg)
		print("for n_clusters= ", k, " The average silhouette_score is :", silhouette_avg)

		sample_silhouette_values = silhouette_samples(data, cluster_labels)
		y_lower = 10
		for i in range(k):
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
		plt.show()
		plt.close()

	fig2, ax2 = plt.subplots(1,1)
	ax2.set_title("mean accuracy for each number of cluster")
	ax2.set_xlabel("kluster")
	ax2.set_ylabel("accuracy")
	ax2.plot(k_grps, list_avg)
	plt.savefig(os.path.join(output_dir, "accuracy per cluster group "+title))
	plt.show()
	plt.close()

if __name__=="__main__":
	#parsing parameters
	parser = argparse.ArgumentParser(description="output the deep features of VGG16")
	parser.add_argument("glob_input", help="path of the file to open")
	parser.add_argument("output_dir", help="path of the file to save")
	args = parser.parse_args()

	#prepare images
	test = load_test(glob(args.glob_input), output=args.output_dir)
	res = get_df(test, ["block4_conv3","block5_conv3","fc1"], args.output_dir)
	print(res[0].shape, res[1].shape, res[2].shape)
	do_kmeans(res[0], [3,4,5,6,7,8], args.output_dir, title="block4_conv3")
	do_kmeans(res[1], [3,4,5,6,7,8], args.output_dir, title="block5_conv3")
	do_kmeans(res[2], [3,4,5,6,7,8], args.output_dir, title="fc1")
