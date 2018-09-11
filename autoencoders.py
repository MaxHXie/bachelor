import functions as func
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import sys
import os

def next_batch(batch_size, data):
    '''
    Randomly sample a batch of size batch_size, from the variable data.

    Arguments:
        batch_size    -- An integer on how big the batch should be
        data          -- 2D list containing data. (often training data)

    Returns:
        batch    -- 2D list of the new batch of data
    '''
    np.random.shuffle(data)
    batch = data[:batch_size]
    return batch

def kl_divergence(p, q):
    kl_div = p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))
    return kl_div

def sparse(one_hot_matrix, index_to_symbol_map, contents_to_row_number_map, test_size, layer_size_divide_list, learning_rate, sparsity_target, sparsity_weight, l2_reg, n_epochs=1, batch_size=500, output_progress=True, output_training_loss=True, plot_difference=False, results_df_rows=100):
    '''
    Train and use a Sparse autoencoder on the inputted data (one_hot_matrix)

    Arguments:
        one_hot_matrix                -- A 2D list, one hot matrix containing all rows in the dataframe
                                         in a one hot representation
        index_to_symbol_map           -- A dictionary mapping an index to a unique symbol
        contents_to_row_number_map    -- A dictionary that maps some string content to an integer (its row number)
        test_size                     -- A float between 0.0 and 1.0
        layer_size_divide_list        -- A list containing integers of how much each layer shuold be divided with.
                                         eg. [4,2] with 10 input units will create the network architecture: 10-3-2-3-10
        learning_rate                 -- A float deciding the learning rate of the model
        sparsity_target               -- A float deciding the sparsity target of the model
        sparsity_weight               -- A float deciding the sparsity weight of the model
        l2_reg                        -- A float deciding the l2 regularization of the model
        n_epochs                      -- An int deciding how many epochs to be trained
        batch_size                    -- An int deciding the size of each batch when training
        output_progress               -- A boolean deciding whether or not to output training progress
        output_training_loss          -- A boolean deciding whether or not to output training loss
        plot_difference               -- A boolean deciding whether or not to graphically scatter plot reconstruction error of datapoints

    Returns:
        df_results    -- A dataframe sorted by rows with highest error, in descending order.
					     Contains columns 'error', 'content' & row_number
    '''
    if test_size != 1.0:
        X_train, X_test = train_test_split(one_hot_matrix, test_size=test_size, random_state=42)
    else:
        print("[TRAINING AND TESTING ON THE SAME DATA] all data")
        np.random.shuffle(one_hot_matrix)
        X_train = one_hot_matrix
        np.random.shuffle(one_hot_matrix)
        X_test = one_hot_matrix

    #Define architecture here
    n_inputs = len(X_train[0])
    n_hidden1 = math.ceil(n_inputs / layer_size_divide_list[0])
    n_hidden2 = math.ceil(n_hidden1 / layer_size_divide_list[1])
    n_hidden3 = n_hidden1
    n_outputs = n_inputs
    print("Architecture: " + str(n_inputs) + "-" + str(n_hidden1) + "-" + str(n_hidden2) + "-" + str(n_hidden3) + "-" + str(n_outputs))

    regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
    initializer = tf.contrib.layers.variance_scaling_initializer()

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])

    #Initialize variables
    print("[INITIALIZING AUTOENCODER]")
    weights1_init = initializer([n_inputs, n_hidden1])
    weights2_init = initializer([n_hidden1, n_hidden2])

    weights1 = tf.Variable(weights1_init, dtype=tf.float32, name='weights1')
    weights2 = tf.Variable(weights2_init, dtype=tf.float32, name='weights2')
    weights3 = tf.transpose(weights2, name='weights3') #tied weights
    weights4 = tf.transpose(weights1, name='weights4') #ties weights

    biases1 = tf.Variable(tf.zeros(n_hidden1), name='biases1')
    biases2 = tf.Variable(tf.zeros(n_hidden2), name='biases2')
    biases3 = tf.Variable(tf.zeros(n_hidden3), name='biases3')
    biases4 = tf.Variable(tf.zeros(n_outputs), name='biases4')

    hidden1 = tf.nn.elu(tf.matmul(X, weights1) + biases1)
    hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, weights2) + biases2) # Use sigmoid so we can use kl_divergence on sparsity_loss
    hidden3 = tf.nn.elu(tf.matmul(hidden2, weights3) + biases3)
    outputs = tf.matmul(hidden3, weights4) + biases4

    hidden2_mean = tf.reduce_mean(hidden2, axis=0) # We only want the coding layer to be sparse
    sparsity_loss = tf.reduce_sum(kl_divergence(sparsity_target, hidden2_mean))
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE
    loss = reconstruction_loss + sparsity_weight * sparsity_loss

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Set iterations
    print("[STARTING TRAINING]")
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = len(X_train) // batch_size
            for iteration in range(n_batches):
                if output_progress:
                    print("\r{}%".format(100 * iteration // n_batches), end='')
                sys.stdout.flush()
                X_batch = next_batch(batch_size, X_train)
                sess.run(training_op, feed_dict={X: X_batch})
            reconstruction_loss_val, sparsity_loss_val, loss_val = sess.run([reconstruction_loss, sparsity_loss, loss], feed_dict={X: X_batch})
            if output_training_loss:
                print("\r{}".format(epoch), "Train MSE:", reconstruction_loss_val, "\tSparsity loss:", sparsity_loss_val, "\tTotal loss:", loss_val)
            saver.save(sess, "./trained_sparse_model.ckpt")

        print("FINAL Train MSE: " + str(reconstruction_loss_val))
        print("FINAL Sparsity loss " + str(sparsity_loss_val))
        print("FINAL Total loss " + str(loss_val))

        outputs_val = outputs.eval(feed_dict={X: X_test})
        if plot_difference:
            func.plot_difference(X_test, outputs_val)
        results_df = func.get_results_dataframe(X_test, outputs_val, index_to_symbol_map, contents_to_row_number_map, results_df_rows)
        return results_df
