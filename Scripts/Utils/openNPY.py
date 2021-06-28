import matplotlib.pyplot as plt
import numpy as np
import argparse

# class Visu:
# 	def callback_left_button(self, event):
# 		''' this function gets called if we hit the left button'''
# 		print('Left button pressed')
# 		self.current_idx = np.max(0, self.current_idx-1)
#
# 	def callback_right_button(self, event):
# 		''' this function gets called if we hit the left button'''
# 		print('Right button pressed')
# 		self.current_idx = np.min(len(self.data), self.current_idx+1)
#
# 	def update_figure(self):
# 		self.ax.imshow(self.data[self.current_idx])
# 		self.ax.set_title(self.current_idx)
#
# 	def __init__(self, data):
# 		self.data = data
# 		self.current_idx = 0
# 		self.fig = plt.figure(figsize=(24, 18))
# 		self.ax = self.fig.add_subplot(111)
#
# 		toolbar_elements = self.fig.canvas.toolbar.children()
# 		left_button = toolbar_elements[6]
# 		right_button = toolbar_elements[8]
# 		left_button.click()
# 		left_button.toggle()
# 		self.fig.canvas.toolbar.set_history_buttons()
# 		# left_button.setCheckable(True)
# 		left_button.clicked.connect(self.callback_left_button)
# 		right_button.clicked.connect(self.callback_right_button)
#
# 		self.update_figure()
#
# 	def show(self):
# 		plt.show()

def f_open(path):
	data = np.load(path)
	print(data.shape)
	assert data.ndim==4 and data.shape[-1]<=3, "wrong data shape {}".format(data.shape)
	for idx, img in enumerate(data):
		plt.title(idx)
		if data.shape[-1]==2:
			plt.imshow((img[...,0]+img[...,1])/2, cmap='gray')
		else:
			plt.imshow(img, cmap='gray')
		plt.show()




if __name__=="__main__":
	parser = argparse.ArgumentParser(description="Script used to train, test and research optimal latend dim on different Autoencoders")
	parser.add_argument("path_npy", help="path of the numpy to open")
	args = parser.parse_args()
	f_open(args.path_npy)
	# a = Visu(np.load(args.path_npy))
	# a.show()
