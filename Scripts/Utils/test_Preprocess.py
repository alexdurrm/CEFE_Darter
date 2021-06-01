import Preprocess
import unittest

class test_Preprocess(unittest.TestCase):
	def test(self):
		pass

		
if __name__=='__main__':
	unittest.main()

"""
	image = imageio.imread(args.input_path)
	pr = Preprocess(resizeX=args.resizeX, resizeY=args.resizeY, normalize=args.normalize, standardize=args.standardize,
		img_type=args.type_img, img_channel=args.channel_img)
	#1st way to do
	img_pr1 = pr(args.input_path)
	#2nd way to do
	img_pr2 = pr.get_image()

	if args.verbose >=1:
		f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
		ax1.imshow(image, cmap='gray')
		ax2.imshow(img_pr1, cmap='gray')
		ax3.imshow(img_pr2, cmap='gray')
		plt.show()

"""