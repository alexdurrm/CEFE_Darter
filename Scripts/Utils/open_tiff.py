import matplotlib.pyplot as plt
import imageio
import sys
if __name__ == '__main__':
    image = imageio.imread(sys.argv[1])
    ax=plt.imshow(image)
    plt.gcf().set_facecolor("black")
    plt.show()
