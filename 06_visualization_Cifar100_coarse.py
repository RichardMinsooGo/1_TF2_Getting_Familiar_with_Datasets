
# example of loading the fashion mnist dataset
from matplotlib import pyplot
import tensorflow as tf

# Load the CIFAR-10 dataset
cifar100 = tf.keras.datasets.cifar100

# Get test and training data where x are the images and y are the labels
# load dataset
(trainX, trainy), (testX, testy) = cifar100.load_data(label_mode='coarse')

# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(trainX[i])
# show the figure
pyplot.show()

LABELS_LIST = [
    'aquatic mammals',
    'fish',
    'flowers',
    'food containers',
    'fruit and vegetables',
    'household electrical devices',
    'household furniture',
    'insects',
    'large carnivores',
    'large man-made outdoor things',
    'large natural outdoor scenes',
    'large omnivores and herbivores',
    'medium-sized mammals',
    'non-insect invertebrates',
    'people',
    'reptiles',
    'small mammals',
    'trees',
    'light vehicle',
    'heavy vehicle'
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
    ax.set_title('{}'.format(LABELS_LIST[trainy[i][0]]))
pyplot.tight_layout()
pyplot.show()




