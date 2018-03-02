import sys
import os.path
from optparse import OptionParser
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.text_dataset import TextDataSet
from utils.process_config import process_config
import tools.models as mo
import tensorflow as tf
import time

# 讓import module可以找到，解決import error

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

    # 2. 建立saver物件，若有check point，從未train完的model restore後繼續train
    saver = tf.train.Saver(max_to_keep=3)
    begin_epoch = 0
    path = tf.train.latest_checkpoint(model_path_str)
    if path != None:
        saver.restore(sess, path)  # search for checkpoint file
        last_itr_num = path[path.index('-') + 1:]
        begin_epoch = int(last_itr_num) + 1

    start_time = time.time()

    print('Step3: begin to train..')
    for epo_idx in range(begin_epoch, begin_epoch + dataSet.epoch):
        epoch_complete = False
        total_loss_val = 0.0
        num_batch = 0.0
        while True:
            (epoch_complete, imgs, rect_labels) = dataSet.batch()
            if epoch_complete == True:
                break
            else:
                loss_val, _ = sess.run([loss, train_step], feed_dict={model.img_in_placeholder: imgs, model.rect_label_placeholder: rect_labels})
                total_loss_val += loss_val
                num_batch += 1.0
        total_loss_val /= num_batch
        print('Epoch: %d, loss = %f' % (epo_idx, total_loss_val))
        if epoch_complete:
            # score_test = loss.eval(feed_dict={model.img_in_placeholder: X2_test, model.rect_label_placeholder: Y2_test})
            # if score_test < best_score:
            #     best_score = score_test
            if epo_idx % 500 == 0:
                saver.save(sess, os.path.join(model_path_str, "model.ckpt"), global_step=epo_idx)
                print('Epoch: %d, loss = %f, saved!' % (epo_idx, total_loss_val))
            dataSet.resetBatch()
    print('Step3: train ok !!')
    print('Finished in %f seconds.' % (time.time() - start_time))

