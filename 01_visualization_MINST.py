
# example of loading the fashion mnist dataset
import tensorflow as tf
from matplotlib import pyplot
# from keras.datasets import mnist
mnist = tf.keras.datasets.mnist
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()


fig = pyplot.figure()
for i in range(60):
    subplot = fig.add_subplot(4, 15, i + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % trainy[i])
    subplot.imshow(trainX[i], cmap=pyplot.cm.gray_r)

pyplot.show()

