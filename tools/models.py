import tensorflow as tf


# Create placeholders for image data and expected point positions

class Model(object):
    xxx = 0

def weight_variable(name, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, initializer=initial)

def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Build neural network
def build_model(models_params):
    img_width = (int)(models_params['image_width'])
    img_height = (int)(models_params['image_height'])
    LABEL_COUNT = 4
    img_in_placeholder = tf.placeholder(tf.float32, shape=[None, img_height, img_width])
    x_image = tf.reshape(img_in_placeholder, shape=(-1, img_height, img_width, 1))
    # Convolution Layer 1
    nNumFilter1 = 32
    W_conv1 = weight_variable("w1", [3, 3, 1, nNumFilter1])
    b_conv1 = bias_variable("b1", [nNumFilter1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # [height, width, 1] => [height, width, 32], 32，注意寬高不變是因為padding='SAME'
    h_pool1 = max_pool_2x2(h_conv1)                           # [height/2, width/2, 1]

    # Convolution Layer 2
    nNumFilter2 = 64
    W_conv2 = weight_variable("w2", [2, 2, nNumFilter1, nNumFilter2])
    b_conv2 = bias_variable("b2", [nNumFilter2])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # [height/2, width/2, 32] => [height/2, width/2, 64]
    h_pool2 = max_pool_2x2(h_conv2)                           # [height/4, width/4, 64]


    # Convolution Layer 3
    nNumFilter3 = 128
    W_conv3 = weight_variable("w3", [2, 2, nNumFilter2, nNumFilter3])
    b_conv3 = bias_variable("b3", [nNumFilter3])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)  # [height/4, width/4, 64] => [height/4, width/4, 128]
    h_pool3 = max_pool_2x2(h_conv3)                           # [height/8, width/8, 128]
    # 除數，若再加pooling層數，加幾層就多除幾次2
    nDividFac = 8

    # Dense layer 1
    nNumAfterFlat = int(img_height / nDividFac * img_width / nDividFac * nNumFilter3)
    nNumFullConn = 500
    h_pool3_flat = tf.reshape(h_pool3, [-1, nNumAfterFlat])
    W_fc1 = weight_variable("w4", [nNumAfterFlat, nNumFullConn])
    b_fc1 = bias_variable("b4", [nNumFullConn])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Dense layer 2
    W_fc2 = weight_variable("w5", [nNumFullConn, nNumFullConn])
    b_fc2 = bias_variable("b5", [nNumFullConn])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    # Output layer
    W_out = weight_variable("w6", [nNumFullConn, LABEL_COUNT])
    b_out = bias_variable("b6", [LABEL_COUNT])

    rect_predict = tf.matmul(h_fc2, W_out) + b_out

    rect_label_placeholder = tf.placeholder(tf.float32, shape=[None, LABEL_COUNT])

    model = Model()
    model.img_in_placeholder = img_in_placeholder
    model.rect_label_placeholder = rect_label_placeholder
    model.rect_predict = rect_predict

    return model