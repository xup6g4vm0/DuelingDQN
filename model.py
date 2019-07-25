import numpy as np
import tensorflow as tf

class DuelingDQN:
  def __init__(
          self,
          state_dims,
          action_dims,
          learning_rate=0.01,
          reward_decay=0.9,
          e_greedy=0.9,
          replace_target_iter=300,
          memory_size=500,
          batch_size=32,
          output_graph=False
  ):
    self.state_dims = state_dims
    self.action_dims = action_dims
    self.lr = learning_rate
    self.gamma = reward_decay
    self.epsilon = e_greedy
    self.replace_target_iter = replace_target_iter
    self.memory_size = memory_size
    self.batch_size = batch_size
    
    self.learn_step_counter = 0

    self.memory = []

    self._build_net()
    q_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    self.replace_target_op = [tf.assign(t, q) for t, q in zip(t_params, q_params)]

    self.sess = tf.Session()

    if output_graph:
      tf.summary.FileWriter('logs/', self.sess.graph)

    self.sess.run(tf.global_variables_initializer())

  def _build_net(self):
    self.s = tf.placeholder(tf.float32, [None, self.state_dims], name='state')
    self.q_target = tf.placeholder(tf.float32, [None, self.action_dims], name='Q_target')
    with tf.variable_scope('eval_net'):
      h1 = tf.layers.dense(self.s, 10, activation=tf.nn.relu)
      V = tf.layers.dense(h1, 1, activation=None) # value function
      A = tf.layers.dense(h1, self.action_dims, activation=None) #Advantage function

      self.q_eval = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))

    with tf.variable_scope('loss'):
      self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval, self.q_target))
    with tf.variable_scope('train'):
      self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    self.s_ = tf.placeholder(tf.float32, [None, self.state_dims], name='state_')
    with tf.variable_scope('target_net'):
      h1 = tf.layers.dense(self.s_, 10, activation=tf.nn.relu)
      V = tf.layers.dense(h1, 1, activation=None) # value function
      A = tf.layers.dense(h1, self.action_dims, activation=None) #Advantage function

      self.q_next = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))

  def store_transition(self, s, a, r, ns, d):
    self.memory.append( (s, a, r, ns, d) )

    if len(self.memory) > self.memory_size:
      self.memory = self.memory[-self.memory_size:]

  def choose_action(self, state, _eval=False):
    state = state[np.newaxis, :]

    if _eval or np.random.uniform() < self.epsilon:
      action_value = self.sess.run(self.q_eval, feed_dict={self.s: state})
      action = np.argmax(action_value)
    else:
      action = np.random.randint(self.action_dims)

    return action

  def learn(self):
    if self.learn_step_counter % self.replace_target_iter == 0:
      self.sess.run(self.replace_target_op)
      print('\ntarget_params_replaced')

    sample_index = np.random.choice(len(self.memory), size=self.batch_size)
    s  = [ self.memory[idx][0] for idx in sample_index ]
    a  = [ self.memory[idx][1] for idx in sample_index ]
    r  = [ self.memory[idx][2] for idx in sample_index ]
    ns = [ self.memory[idx][3] for idx in sample_index ]
    d  = [ self.memory[idx][4] for idx in sample_index ]
    
    q_next, q_eval = self.sess.run([self.q_next, self.q_eval], feed_dict={
                        self.s_: ns,
                        self.s: s })

    q_target = q_eval.copy()

    for idx in range(self.batch_size):
      if d[idx]:
        q_target[ idx, a[idx] ] = r[idx]
      else:
        q_target[ idx, a[idx] ] = r[idx] + self.gamma * np.max(q_next[idx])

    _, cost = self.sess.run([self.train_op, self.loss], feed_dict={
                        self.s: s,
                        self.q_target: q_target })

    self.learn_step_counter += 1
