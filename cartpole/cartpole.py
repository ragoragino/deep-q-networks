"""
Cartpole task solved with Double Deep Q-Network
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import gym

cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)
rseed = 123
np.random.seed(rseed)
plt.style.use('ggplot')

# Obtaining cart-pole environment
env = gym.make('CartPole-v0')
ACTION_SET = np.array([0, 1])  # available actions

"""
NN SETUP
"""
# Parameters of the learning
LEARNING_RATE = 0.001
HIDDEN_NAMES = ['h1', 'h2',  'out']
SIZE_HIDDEN = {}
SIZE_HIDDEN['h1'] = 32
SIZE_HIDDEN['h2'] = 64
SIZE_HIDDEN['out'] = len(ACTION_SET)
SIZE_INPUT = 4
STD_HIDDEN = {}
STD_HIDDEN['h1'] = np.sqrt(1 / SIZE_INPUT)
STD_HIDDEN['h2'] = np.sqrt(1 / SIZE_HIDDEN['h1'])
STD_HIDDEN['out'] = np.sqrt(1 / SIZE_HIDDEN['h2'])
REGULARIZER = 0.0001
ALPHA = 0.1
BUFFER_SIZE = 256
BATCH_SIZE = 196
UPDATE_TARGET = 10
MAX_ITER = 10000
EPSILON = 0.0
GAMMA = 1
EXPLORATION_LIMIT = 200

with tf.Graph().as_default():
    X = tf.placeholder("float32", [None, SIZE_INPUT])
    Y = tf.placeholder("float32", [None, SIZE_HIDDEN['out']])

    with tf.name_scope("layer1"):
        W1 = tf.Variable(tf.random_normal([SIZE_INPUT, SIZE_HIDDEN['h1']], 0, STD_HIDDEN['h1'],
                                             dtype=tf.float32), name='W1')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.multiply(tf.nn.l2_loss(W1), REGULARIZER))
        b1 = tf.Variable(tf.zeros([SIZE_HIDDEN['h1']], dtype=tf.float32), name='b1')
        layer1 = tf.matmul(X, W1) + b1
        layer1_act = tf.nn.relu(layer1) - ALPHA * tf.nn.relu(-layer1)

    with tf.name_scope("layer2"):
        W2 = tf.Variable(tf.random_normal([SIZE_HIDDEN['h1'], SIZE_HIDDEN['h2']], 0, STD_HIDDEN['h2'],
                                             dtype=tf.float32), name='W2')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.multiply(tf.nn.l2_loss(W2), REGULARIZER))
        b2 = tf.Variable(tf.zeros([SIZE_HIDDEN['h2']], dtype=tf.float32), name='b2')
        layer2 = tf.matmul(layer1_act, W2) + b2
        layer2_act = tf.nn.relu(layer2) - ALPHA * tf.nn.relu(-layer2)

    with tf.name_scope("layer3"):
        W3 = tf.Variable(tf.random_normal([SIZE_HIDDEN['h2'], SIZE_HIDDEN['out']], 0, STD_HIDDEN['out'],
                                             dtype=tf.float32), name='W3')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.multiply(tf.nn.l2_loss(W3), REGULARIZER))
        b3 = tf.Variable(tf.zeros([SIZE_HIDDEN['out']], dtype=tf.float32), name='b3')
        pred = tf.matmul(layer2_act, W3) + b3

    with tf.name_scope("layer1_back"):
        W1_back = tf.Variable(tf.random_normal([SIZE_INPUT, SIZE_HIDDEN['h1']], 0, STD_HIDDEN['h1'],
                                             dtype=tf.float32), name='W1_back')
        b1_back = tf.Variable(tf.zeros([SIZE_HIDDEN['h1']], dtype=tf.float32), name='b1_back')
        layer1_back = tf.matmul(X, W1_back) + b1_back
        layer1_act_back = tf.nn.relu(layer1_back) - ALPHA * tf.nn.relu(-layer1_back)

    with tf.name_scope("layer2_back"):
        W2_back = tf.Variable(tf.random_normal([SIZE_HIDDEN['h1'], SIZE_HIDDEN['h2']], 0, STD_HIDDEN['h2'],
                                             dtype=tf.float32), name='W2_back')
        b2_back = tf.Variable(tf.zeros([SIZE_HIDDEN['h2']], dtype=tf.float32), name='b2_back')
        layer2_back = tf.matmul(layer1_act_back, W2_back) + b2_back
        layer2_act_back = tf.nn.relu(layer2_back) - ALPHA * tf.nn.relu(-layer2_back)

    with tf.name_scope("layer3_back"):
        W3_back = tf.Variable(tf.random_normal([SIZE_HIDDEN['h2'], SIZE_HIDDEN['out']], 0, STD_HIDDEN['out'],
                                             dtype=tf.float32), name='W3_back')
        b3_back = tf.Variable(tf.zeros([SIZE_HIDDEN['out']], dtype=tf.float32), name='b3_back')
        pred_target = tf.matmul(layer2_act_back, W3_back) + b3_back

    # Adjusting the target network
    assign = [W1_back.assign(W1), W2_back.assign(W2), W3_back.assign(W3),
              b1_back.assign(b1), b2_back.assign(b2), b3_back.assign(b3)]

    # Regularizer
    regul_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # Loss node
    loss = tf.reduce_mean(tf.squared_difference(pred, Y)) + regul_loss

    # Estimator node
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # Build the graph
    sess = tf.Session()

    # Gathering of nodes
    features = [optimizer]
    prediction = [pred]
    prediction_target = [pred_target]

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Assign initialized action-value function to target action-value function
    sess.run(assign)

input_buffer = np.zeros((BUFFER_SIZE, SIZE_INPUT))
output_buffer = np.zeros((BUFFER_SIZE, SIZE_HIDDEN['out']))
input_old = np.zeros((1, SIZE_INPUT), dtype=np.float32)
input_new = np.zeros((1, SIZE_INPUT), dtype=np.float32)
average_reward = np.zeros(100, dtype=np.float32)
aggregate_reward = []
buffer_ready = False
h = 0
trials = 0

for it in range(MAX_ITER):
    step = 0
    old_state = env.reset()
    input_old.fill(0)
    input_new.fill(0)

    # env.render()

    while True:
        step += 1
        trials += 1
        h %= BUFFER_SIZE

        input_buffer[h, :] = old_state
        input_old[:] = old_state
        output_buffer[h, :] = sess.run(prediction, feed_dict={X: input_old})[0][0]

        # Epsilon-greedy action
        explore = 1 - min(it, EXPLORATION_LIMIT) * (1 - EPSILON) / EXPLORATION_LIMIT
        if np.random.binomial(1, explore) == 1:
            new_index = np.random.choice(np.arange(0, len(ACTION_SET)))
            new_action = ACTION_SET[new_index]
        else:
            high_action = - np.inf
            for i in range(len(ACTION_SET)):
                pick_action = output_buffer[h, i]
                if pick_action > high_action:
                    high_action = pick_action
                    new_index = i
            new_action = ACTION_SET[new_index]

        new_state, reward, done, info = env.step(new_action)

        high_action = - np.inf
        input_new[:] = new_state
        action_pred_new = sess.run(prediction_target, feed_dict={X: input_new})[0][0]
        for i in range(len(ACTION_SET)):
            pick_action = action_pred_new[i]
            if pick_action > high_action:
                high_action = pick_action

        if not done:
            delta = reward + GAMMA * high_action
        else:
            delta = reward

        output_buffer[h, new_index] = delta
        h += 1

        # Experience replay
        h_replay = min(BATCH_SIZE, trials)
        random_list = np.random.choice(np.arange(0, min(BUFFER_SIZE, trials), 1),
                                       size=h_replay, replace=False)
        _ = sess.run(features, feed_dict={X: input_buffer[random_list, :],
                                          Y: output_buffer[random_list, :]})

        # Update target action-value function
        if (trials % UPDATE_TARGET) == 0:
            sess.run(assign)

        old_state = new_state

        if done:
            print("Episode %d finished after %f time steps" % (it, step))
            aggregate_reward.append(step)
            average_reward[(it % 100)] = step
            if it >= 100 and average_reward.mean() >= 195.0:
                print("SUCCESS AFTER {} EPISODES".format(it))

                plt.figure(figsize=(16, 12), dpi=100)
                plt.plot(aggregate_reward, linewidth=1.0, linestyle="-")
                plt.title("Number of steps required to finish an episode")
                plt.savefig('cartpole.pdf', bbox_inches='tight',
                            dpi=100, format='pdf')

                sys.exit(0)

            break


