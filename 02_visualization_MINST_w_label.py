
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

LABELS_LIST = [
    '0', 
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9'
]

num_row = 4
num_col = 12
num = num_row*num_col
images = trainX[:num]
labels = trainy[:num]
# plot images
fig, axes = pyplot.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray')
    ax.set_title('{}'.format(LABELS_LIST[trainy[i]]))
pyplot.tight_layout()
pyplot.show()

