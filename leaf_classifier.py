import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from cnn_utils import load_dataset, convert_to_one_hot, random_mini_batches
from alexnet import AlexNet , make_image_dataset, create_dataset

def train_model(image_path, learning_rate=0.001,num_epoch=100, batch_size = 5):
    X_train = np.load(image_path+".npy")
    Y_train = np.load(image_path+".npy")

    X_train = X_train / 255
    #Creating one-hot for label
    print(Y_train.shape)
    print(Y_train[0][1])
    Y_train = convert_to_one_hot(Y_train,10).T
    n_Y = Y_train.shape[1]

    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    Y = tf.placeholder(tf.float32 ,[None ,n_Y])
    keep_prob = tf.placeholder(tf.float32)
    model = AlexNet(x, keep_prob, 10, [])

    score = model.fc8
    cost = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=Y)))
    softmax = tf.nn.softmax(score)

    costs = []
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost) 
    init = tf.global_variables_initializer()

    [m ,nH, nW, nCH ] = X_train.shape
    seed =0
    #tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(model.load_initial_weights(sess))
        for i in range(num_epoch):
            minibatch_cost = 0
            num_minibatch = int(m / batch_size)# number of batches
            seed = seed + 1 
            minibatches = random_mini_batches(X_train, Y_train, batch_size , seed)
            for minibatch in minibatches:
                X_batches ,Y_batches = minibatch
                #Use cost and optimizer to run 
                _,c = sess.run([optimizer,cost], {x:X_batches , Y:Y_batches, keep_prob:1})

                minibatch_cost += c

            minibatch_cost = minibatch_cost / num_minibatch
            if i%5 == 0:
                print("Cost===>"+str(c))

        costs.append(c)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.title('learning_rate'+str(learning_rate))
        plt.show()

