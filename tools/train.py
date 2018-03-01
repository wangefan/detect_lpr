import sys
import os.path
from optparse import OptionParser
import os
from utils.text_dataset import TextDataSet
from utils.process_config import process_config
import tools.models as mo
import tensorflow as tf
import time

# 讓import module可以找到，解決import error
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('./')

parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure", help="configure filename")
(options, args) = parser.parse_args() 
if options.configure:
  conf_file = str(options.configure)
else:
  print('please specify --conf configure filename')
  exit(0)

common_params, data_set_params, models_params = process_config(conf_file)
MODEL_PATH_KEY = 'model_path'
model_path_str = common_params[MODEL_PATH_KEY]

# create model path
if not os.path.exists(model_path_str):
    os.makedirs(model_path_str)

print('Step1: begin to load training data..')
dataSet = TextDataSet(data_set_params)
dataSet.loadTrainData()
print('Step1: load training data OK !!')

print('Step2: begin to build model..')
model = mo.build_model(models_params)
loss = tf.reduce_mean(tf.square(model.rect_label_placeholder - model.rect_predict))
train_step = tf.train.AdamOptimizer().minimize(loss)
print('Step2: begin to build model OK !!')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
with tf.Session(config=config) as sess:

    # 1. initialize variables
    sess.run(tf.global_variables_initializer())

    # 2. 建立saver物件，todo:從未train完的model restore後繼續train
    saver = tf.train.Saver()
    # saver.restore(session, os.path.join(MODEL_PATH, "model"))

    start_time = time.time()
    best_score = 1
    last_epoch = -1
    while dataSet.epoch_completed() < dataSet.epoch:
        (imgs, rect_labels) = dataSet.batch()
        loss_val, _ = sess.run([loss, train_step], feed_dict={model.img_in_placeholder: imgs, model.rect_label_placeholder: rect_labels})
        if dataSet.epoch_completed() > last_epoch:
            last_epoch = dataSet.epoch_completed()
            # score_test = loss.eval(feed_dict={model.img_in_placeholder: X2_test, model.rect_label_placeholder: Y2_test})
            # if score_test < best_score:
            #     best_score = score_test
            if dataSet.epoch_completed() % 10 == 0:
                saver.save(sess, os.path.join(model_path_str, "model.ckpt"), global_step=dataSet.epoch_completed())

            if dataSet.epoch_completed() % 1 == 0:
                epm = 60 * dataSet.epoch_completed() / (time.time() - start_time)
                print('Epoch: %d, loss = %d, Epoch per minute: %f' % (dataSet.epoch_completed(), loss_val, epm))
    print('Finished in %f seconds.' % (time.time() - start_time))

