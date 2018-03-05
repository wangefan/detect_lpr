import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from optparse import OptionParser
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pylab import rcParams
import tensorflow as tf
import tools.models as mo
from utils.process_config import process_config
from utils.text_dataset import TextDataSet
import random
import numpy as np

def show_image(image, labels):
    rect = Rectangle((labels[0], labels[1]), labels[2] - labels[0], labels[3] - labels[1], edgecolor='r', fill=False)
    plt.imshow(image)
    gca = plt.gca()
    gca.add_patch(rect)


def plot_images(images, labels):
    rcParams['figure.figsize'] = 14, 8
    plt.gray()
    fig = plt.figure()
    for i in range(min(9, images.shape[0])):
        fig.add_subplot(3, 3, i + 1)
        show_image(images[i], labels[i])
    plt.show(block=True)

sys.path.append('./')
parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure",
                  help="configure filename")
(options, args) = parser.parse_args()
if options.configure:
  conf_file = str(options.configure)
else:
  print('please sspecify --conf configure filename')
  exit(0)

_, data_set_params, _ = process_config(conf_file)

# 3.
dataSet = TextDataSet(data_set_params)
dataSet.loadTestData()
X_test, y_test = dataSet.getTestData()

ids = [random.randint(0, X_test.shape[0]-1) for _ in range(9)]
# predictions = model.rect_predict.eval(session=sess, feed_dict={model.img_in_placeholder: X_test[ids]})
plot_images(X_test[ids], y_test[ids])
