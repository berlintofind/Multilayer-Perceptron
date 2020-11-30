# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:01:05 2020

@author: To find Berlin
"""


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import math
import h5py
import os

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X_train = X_train_flatten/255
X_test = X_test_flatten/255

Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)
# (Y_train.shape) -- (6, 1080)


# Step 1: Create a graph containing Tensors (Variables, Placeholders ...) and Operations (tf.matmul, tf.add, ...)
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x,None],name="X")
    Y = tf.placeholder(tf.float32, shape=[n_y,None],name="Y")
    
    return X, Y

def initialize_parameters(layers_dims):
    tf.set_random_seed(1)                   
    parameters = {}
    L = len(layers_dims)          # number of layers in the network

    for l in range(1,L):
        parameters['W' + str(l)] = tf.get_variable("W"+str(l), [layers_dims[l], layers_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters['b' + str(l)] = tf.get_variable("b"+str(l), [layers_dims[l],1], initializer = tf.zeros_initializer())

    return parameters

def L2_regular(parameters,weight_decay=0.00004):
    L = int(len(parameters)/2)

    if weight_decay > 0:
        for l in range(1,L):
            weight_loss= tf.nn.l2_loss(parameters["W" + str(l)]) * weight_decay         # L2, weight_loss
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value = weight_loss)
    else:
        pass


def compute_cost(y_hat, Y):
    logits = tf.transpose(y_hat)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    weight_loss_op = tf.losses.get_regularization_losses()
    weight_loss_op = tf.add_n(weight_loss_op)
    total_loss_op = cost + weight_loss_op

    tf.summary.scalar("loss",total_loss_op)
    global merged_summary_op
    merged_summary_op = tf.summary.merge_all()

    return total_loss_op

def fully_connected(input_op, scope,  parameters, l,num_outputs, weight_decay=0.00004, is_activation=True, fineturn=True):
    L2_regular(parameters,weight_decay=0.00004)

    with tf.compat.v1.variable_scope(scope):
        weights = parameters['W'+str(l)]
        biases = parameters['b'+str(l)]

        if is_activation:
            Z = tf.add(tf.matmul(weights,input_op),biases)
            axis = list(range(len(Z.get_shape()) - 1))
            Z = tf.layers.batch_normalization(Z,axis=axis,training=fineturn)
            return tf.nn.relu(Z)
        else:
            Z = tf.add(tf.matmul(weights,input_op),biases)
            axis = list(range(len(Z.get_shape()) - 1))
            Z = tf.layers.batch_normalization(Z,axis=axis,training=fineturn)
            return Z

def model(X_train, Y_train, X_test, Y_test, layers_dims,  weight_decay=0.00004,learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(layers_dims)


    L = len(layers_dims)
    net = X
    for l in range(1,L-1):
        net = fully_connected(net, 'fc'+str(l), parameters, l, layers_dims[l], weight_decay=weight_decay )

    y_hat = fully_connected(net, 'logits',parameters, L-1, layers_dims[L-1], is_activation=False, weight_decay=weight_decay ) # 中间好像差一步
    total_loss_op = compute_cost(y_hat, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss_op) #这里该变，变为二者相加之和     

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Step 2: Start the session to compute the tensorflow graph
    # Step 3: Initialize the session
    config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True,device_count = {'GPU': 1,'CPU':1})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1

    # with  tf.device('/gpu:0'):
    sess = tf.InteractiveSession(config=config)
    sess.run(init)

    checkpoint = tf.train.get_checkpoint_state("MLP_Softmax\\Softmax--820")
    # print(checkpoint) MLP_Softmax\\Softmax--820
    saver = tf.train.Saver() 

    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print ("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print ("Could not find old network weights")

    global summary_writer
    summary_writer = tf.summary.FileWriter('Summaryfile',graph=sess.graph)
    global merged_summary_op
    merged_summary_op = tf.summary.merge_all()

    for epoch in range(num_epochs):

        epoch_cost = 0.                           # Defines a cost related to an epoch
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            _ , minibatch_cost = sess.run([optimizer, total_loss_op], feed_dict = {X:minibatch_X, Y:minibatch_Y}) # 这里改变

            summary_str = sess.run(merged_summary_op,feed_dict={X:minibatch_X, Y:minibatch_Y})
            summary_writer.add_summary(summary_str,epoch)


            if epoch % 10 == 0:
                saver.save(sess, 'MLP_Softmax/'+'Softmax-', global_step = epoch)

            epoch_cost += minibatch_cost / minibatch_size

        if print_cost == True and epoch % 100 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 5 == 0:
            costs.append(epoch_cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.savefig("Mlp_Softmax")
    # plt.show()

    # lets save the parameters in a variable
    parameters = sess.run(parameters)
    print ("Parameters have been trained!")

    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(y_hat), tf.argmax(Y))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
    sess.close()
    return parameters


Number_Classes = 6
layers_dims = [X_train.shape[0], 25, 20,14, Number_Classes]
parameters = model(X_train, Y_train, X_test, Y_test,layers_dims)

import scipy
from PIL import Image
from scipy import ndimage

my_image = "thumbs_up.jpg"
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
image = image/255.
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T

my_image_prediction = predict(my_image, parameters)
plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
