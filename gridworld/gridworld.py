"""
Gridworld, 4x4, with Q-learning neural-network

4x4 environment
(1, 1) - WALL (0)
(2, 2) - FINISH (+10)
(2, 1) - PIT (-10)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

cur_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(cur_dir)
plt.style.use('ggplot')

"""
PARAMETERS OF THE MODEL
"""
height = 4  # height of the track
width = 4  # width of the track

action_set = [(-1, 0),  (1, 0), (0, -1), (0, 1)]  # available actions
states = [(i, j) for i in range(width) for j in range(height)]

states_vec = {}
for i in range(width):
    for j in range(height):
        states_vec[(i, j)] = np.zeros((1, width * height), dtype=np.float32)
        states_vec[(i, j)][0, j * width + i] = 1

"""
NN SETUP
"""
# Parameters of the learning
LEARNING_RATE = 0.01
HIDDEN_NAMES = ['h1', 'h2',  'out']
SIZE_HIDDEN = {}
SIZE_HIDDEN['h1'] = 128
SIZE_HIDDEN['h2'] = 128
SIZE_HIDDEN['out'] = len(action_set)
SIZE_INPUT = width * height
STD_HIDDEN = {}
STD_HIDDEN['h1'] = np.sqrt(1 / SIZE_INPUT)
STD_HIDDEN['h2'] = np.sqrt(1 / SIZE_HIDDEN['h1'])
STD_HIDDEN['out'] = np.sqrt(1 / SIZE_HIDDEN['h2'])
REGULARIZER = 0.001
ALPHA = 0.1
BUFFER_SIZE = 512
BATCH_SIZE = 64
UPDATE_TARGET = 100
GAMMA = 1
EPSILON = 0.01
MAX_ITER = 1000
EXPLORATION_LIMIT = 500

with tf.Graph().as_default():
    X = tf.placeholder("float32", [None, SIZE_INPUT])
    Y = tf.placeholder("float32", [None, SIZE_HIDDEN['out']])
    inv_dropout = tf.placeholder("float32")

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

    # Adjusting the backup network
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

    # summary_features = [optimizer, merged]
    prediction = [pred]
    prediction_target = [pred_target]

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Assign initialized action-value function to target action-value function
    sess.run(assign)

input_buffer = np.zeros((BUFFER_SIZE, SIZE_INPUT))
output_buffer = np.zeros((BUFFER_SIZE, SIZE_HIDDEN['out']))
input_old = np.zeros((1, width * height), dtype=np.float32)
input_new = np.zeros((1, width * height), dtype=np.float32)
buffer_ready = False
h = 0
learning_step = 0
history = np.zeros(MAX_ITER)

for it in range(MAX_ITER):
    step = 0
    start = (0, 0)
    old_state = start
    input_old.fill(0)
    input_new.fill(0)

    while True:
        step -= 1
        delta = 0
        h %= BUFFER_SIZE

        input_buffer[h, :] = states_vec[(old_state[0], old_state[1])]
        input_old[:] = states_vec[(old_state[0], old_state[1])]
        output_buffer[h, :] = sess.run(prediction, feed_dict={X: input_old})[0][0]

        # Epsilon-greedy action
        explore = 1 - min(it, EXPLORATION_LIMIT) * (1 - EPSILON) / EXPLORATION_LIMIT
        if np.random.binomial(1, explore) == 1:
            new_index = np.random.choice(np.arange(0, len(action_set)))
            new_action = action_set[new_index]
        else:
            high_action = - np.inf
            for i in range(len(action_set)):
                pick_action = output_buffer[h, i]
                if pick_action > high_action:
                    high_action = pick_action
                    new_index = i
            new_action = action_set[new_index]

        new_state = tuple(map(sum, zip(old_state, new_action)))

        # Not viable -> outside the grid or hit the wall
        if new_state not in states or new_state in [(1, 1), (2, 1)]:
            new_state = old_state

            if new_state == (2, 1):
                delta = -10  # Custom negative reward for falling into the pit

        high_action = - np.inf
        input_new[:] = states_vec[(new_state[0], new_state[1])]
        action_pred_new = sess.run(prediction_target,
                                   feed_dict={X: input_new})[0][0]
        for i in range(len(action_set)):
            pick_action = action_pred_new[i]
            if pick_action > high_action:
                high_action = pick_action

        if new_state != (2, 2):
            delta += -1 + GAMMA * high_action
        else:
            delta += 10  # Custom positive reward for achieving the goal

        output_buffer[h, new_index] = delta
        h += 1
        if h == BUFFER_SIZE:
            buffer_ready = True

        # Experience replay
        if buffer_ready:
            random_list = np.random.choice(np.arange(0, BUFFER_SIZE, 1),
                                           size=BATCH_SIZE, replace=False)
            _ = sess.run(features, feed_dict={X: input_buffer[random_list, :],
                                              Y: output_buffer[random_list, :]})
            learning_step += 1

            # Update target action-value function
            if (learning_step % UPDATE_TARGET) == 0:
               sess.run(assign)

        old_state = new_state

        if old_state == (2, 2):
            break

    print("{}: {}".format(it, -step))
    history[it] = -step

window = 40
ma_history = np.zeros(MAX_ITER - window)
for i in range(0, MAX_ITER - window):
    ma_history[i] = np.sum(history[i: i + window]) / window

plt.figure(figsize=(16, 12))
plt.plot(ma_history, color="brown", linewidth=1.0, linestyle="-")
plt.title("Number of steps required to finish an episode, "
          "moving average by {} steps".format(window))
plt.savefig('gridworld_learning.pdf', bbox_inches='tight',
            dpi=100, format='pdf')

track_x = []
track_y = []

for i in range(width):
    for j in range(height):
        track_x.append(i)
        track_y.append(j)

new_state = start
traj_x = []
traj_y = []
traj_x.append(new_state[0])
traj_y.append(new_state[1])
# k there just to make sure the loop ends,
# if there would be no optimal path
k = 0
while k < 25:
    k += 1

    input_old[:] = states_vec[(new_state[0], new_state[1])]
    output_buffer[h, :] = sess.run(prediction,
                                   feed_dict={X: input_old})[0][0]

    high_action = - np.inf
    for i in range(len(action_set)):
        pick_action = output_buffer[h, i]
        if pick_action > high_action:
            high_action = pick_action
            new_index = i
    new_action = action_set[new_index]

    new_state = tuple(map(sum, zip(new_state, new_action)))

    traj_x.append(new_state[0])
    traj_y.append(new_state[1])

    if new_state == (2, 2):
        break

# Plot of the optimal path
plt.figure(figsize=(16, 12))
plt.scatter(track_x, track_y, color="blue", label="Track")
plt.plot(traj_x, traj_y, color="red", linewidth=1.0, linestyle="-")
plt.text(2.1, 2.1, r'END', fontsize=10)
plt.text(2.1, 1.1, r'PIT', fontsize=10)
plt.text(1.1, 1.1, r'WALL', fontsize=10)
plt.savefig('gridworld_track.pdf', bbox_inches='tight',
            dpi=100, format='pdf')