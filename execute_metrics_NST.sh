#!/bin/bash

#parameters
DIR_RESULTS="Results/NST_metrics/"


#list images
python Scripts/Metrics/list_images.py both Images/StyleTransferImages/ -o $DIR_RESULTS -f .jpg .png -e True


#deep features vgg16
python Scripts/Metrics/DeepFeatureMetrics.py work ${DIR_RESULTS}image_list.csv -o $DIR_RESULTS -t RGB -c ALL -m vgg16


#deep features vgg19
python Scripts/Metrics/DeepFeatureMetrics.py work ${DIR_RESULTS}image_list.csv -o $DIR_RESULTS -t RGB -c ALL -m vgg19


	#fft_range=(10, 110) #110 pour des fenetres 200x200!!!
	# GLCM_DISTANCES=[1]
	#gabor_angles=[0, 45, 90, 135]
	#gabor_freq=[0.2, 0.4, 0.8]
	#resize = (1536, 512)

	#initialize preprocessor used
	process_darter_gray = Preprocess(resizeX=resize[0], resizeY=resize[1], normalize=False, standardize=False, img_type=IMG.DARTER, img_channel=CHANNEL.GRAY)
	process_darter_all = Preprocess(resizeX=resize[0], resizeY=resize[1], normalize=False, standardize=False, img_type=IMG.DARTER, img_channel=CHANNEL.ALL)
	process_RGB_all = Preprocess(resizeX=resize[0], resizeY=resize[1], normalize=True, standardize=False, img_type=IMG.RGB, img_channel=CHANNEL.ALL)

	fft_slope = FFTSlopes(fft_range, 512, process_darter_gray, os.path.join(output_dir, CSV_FFT_SLOPE))
	fft_slope.metric_from_csv(data_path)
	fft_slope.save()

	fft_bins = FFT_bins(fft_range, 512, process_darter_gray, os.path.join(output_dir, CSV_FFT_BINS))
	fft_bins.metric_from_csv(data_path)
	fft_bins.save()

	gabor_metric = GaborMetrics(gabor_angles, gabor_freq, process_darter_gray, os.path.join(output_dir, CSV_GABOR))
	gabor_metric.metric_from_csv(data_path)
	gabor_metric.save()

	glcm_metric = HaralickMetrics([2,4], gabor_angles, process_darter_gray, os.path.join(output_dir, CSV_HARALICK))
	glcm_metric.metric_from_csv(data_path)
	glcm_metric.save()

	phog_metric = PHOGMetrics(40, 2, process_darter_gray, os.path.join(output_dir, CSV_PHOG))
	phog_metric.metric_from_csv(data_path)
	phog_metric.save()

	lbp_metric = LBPHistMetrics([8, 16], [2,4], 100, process_darter_gray, os.path.join(output_dir, CSV_LBP))
	lbp_metric.metric_from_csv(data_path)
	lbp_metric.save()

	best_lbp_metric = BestLBPMetrics([8, 16], [2,4], 100, process_darter_gray, os.path.join(output_dir, CSV_BEST_LBP))
	best_lbp_metric.metric_from_csv(data_path)
	best_lbp_metric.save()

	stats_metric = StatMetrics(process_darter_gray, os.path.join(output_dir, CSV_STATS_METRICS))
	stats_metric.metric_from_csv(data_path)
	stats_metric.save()

	color_ratio = ColorRatioMetrics(process_darter_all, os.path.join(output_dir, CSV_COLOR_RATIO))
	color_ratio.metric_from_csv(data_path)
	color_ratio.save()

	print("DONE")
