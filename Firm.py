import numpy as np
import tensorflow as tf
import random, math
from collections import deque
import Globals
from collections import namedtuple
from Neural_Network import NeuralNetModel, ReplayMemory, EpsilonGreedyStrategy

# Named tuple to store experience (states, actions, rewards, next_states)
Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'next_states'])

# --- DQN-based Firms class ---
class Firms:
    def __init__(
        self, space, firm_id, position, wage_offer, 
        productivity, wage_list, memory_size, mini_batch_size
        ):
        # Initialize variables
        self.space = space
        self.firm_id = firm_id
        self.filled_jobs = 0
        self.position = position
        self.wage_offer = wage_offer
        self.action = 0
        self.greedy_wage = 0
        self.productivity = productivity
        self.base_productivity = productivity
        self.profits = 0
        self.current_iteration = 0
        self.av_wage_competitors = 0
        self.av_greedy_wage_for_analysis = 0

        self.memory_size = memory_size
        self.mini_batch_size = mini_batch_size
                
        self.wage_list = wage_list
        self.num_actions = len(wage_list) 

        # Experience replay and state variables
        self.memory = ReplayMemory(self.memory_size)
        self.mini_batch = []
        
        

        self.num_inputs = Globals.num_firms if Globals.model_type == 0 else 1

        # Neural NETWORKS
        initializer = tf.keras.initializers.GlorotNormal()

        self.strategy = EpsilonGreedyStrategy()

        if Globals.num_hidden_layers == 1:
            hid_layers = [25]
        else:
            sum = self.num_inputs + self.num_actions
            hid_layers = [math.ceil(Globals.factor_num_nodes_hidden_layers[0] *sum) , math.ceil(Globals.factor_num_nodes_hidden_layers[1]*sum)]

        # Create policy and target networks
        self.policy_net = NeuralNetModel(self.num_inputs, hid_layers, self.num_actions, initializer, Globals.activation_function)
        self.target_net = NeuralNetModel(self.num_inputs, hid_layers, self.num_actions, initializer, Globals.activation_function)
        
        if Globals.optimizer == 'SGD':
            self.optimizer = tf.optimizers.SGD(Globals.LEARNING_RATE)
        else:
            self.optimizer = tf.optimizers.Adam(Globals.LEARNING_RATE)
            

        # Define the train step
        self._run_train_step = self._run_train_step
        self.run_train_step = tf.function(self._run_train_step)


        self.employment_list = []
        self.application_list = []


    def update_iteration(self, iteration):

        self.current_iteration = iteration

    # Firing of workers
    def firing(self):
        for worker in self.employment_list:
            worker.employment_status = 0
            worker.where_work = -1
            worker.pay_off = 0
        self.application_list.clear()
        self.employment_list.clear()
        self.filled_jobs = 0

    # Set productivity (possibly randomize)
    def set_productivity(self):
        if Globals.random_productivity == 1:
            random_double = random.uniform(-Globals.delta_productivity,Globals.delta_productivity)
            self.productivity = self.base_productivity * (1 + random_double)

    # Set current state representation, as needed for DQN input, model_type specific
    def set_current_state(self):

        self.current_state = np.zeros(Globals.num_firms if Globals.model_type == 0 else 1)
        all_firms = sorted(self.space.get_objects(Firms), key=lambda firm: firm.firm_id)
        
        if Globals.model_type == 0:
            # Sort all_firms by firm_id in ascending order
            for i, firm in enumerate(all_firms):
                self.current_state[i] = firm.wage_offer
        else:
            # Single firm wage only
            self.current_state[0] = self.wage_offer

        competitor_wages = [firm.wage_offer for firm in all_firms if firm.firm_id != self.firm_id]

        if competitor_wages:
            self.av_wage_competitors = sum(competitor_wages) / len(competitor_wages)
        else:
            self.av_wage_competitors = 0
        
        self.current_state = tf.convert_to_tensor(self.current_state, dtype=tf.float32)


    # Determine current wage offer using epsilon-greedy policy
    def wage_offer_method(self):
        epsilon = self.strategy.get_exploration_rate(self.current_iteration)
        if epsilon > random.random():
            self.action = random.randrange(self.num_actions)
        else:
            q_values = self.policy_net(self.current_state)
            self.action = np.argmax(q_values)

        self.wage_offer = self.wage_list[self.action]
        self.greedy_wage = self.wage_offer if epsilon <= random.random() else None

    # Hiring of workers
    def hiring(self):
        if Globals.model_type == 0:
            for worker in self.application_list:
                self.employment_list.append(worker)
                worker.where_work = self.firm_id
                worker.employment_status = 1
        self.filled_jobs = len(self.employment_list)

    # Profit calculation
    def calculate_profits(self):
        self.profits = (
            self.productivity * (len(self.employment_list) ** Globals.alpha) -
            (self.wage_offer + self.wage_offer * Globals.fee * Globals.share_fee) * len(self.employment_list)
        )
        # Normalize reward if specified
        if Globals.reward_normalization == 1:
            self.profits /= 1000.0


    # Define next state and store experience in replay memory
    def set_next_state(self):

        self.next_state = np.zeros(Globals.num_firms if Globals.model_type == 0 else 1)
        if Globals.model_type == 0:
            # Sort all_firms by firm_id in ascending order
            all_firms = sorted(self.space.get_objects(Firms), key=lambda firm: firm.firm_id)

            for i, firm in enumerate(all_firms):
                self.next_state[i] = firm.wage_offer
        else:
            # Single firm wage only
            self.next_state[0] = self.wage_offer

        self.next_state = tf.convert_to_tensor(self.next_state, dtype=tf.float32)
        self.memory.push(Experience(self.current_state, self.action, self.profits, self.next_state))


    # Training step
    def training(self):

        # Sample mini-batch from replay memory
        experiences = self.memory.sample(min(self.mini_batch_size, len(self.memory)))

        # Prepare batches
        states_batch = tf.convert_to_tensor([exp.states for exp in experiences], dtype=tf.float32)
        actions_batch = tf.convert_to_tensor([exp.actions for exp in experiences], dtype=tf.int32)
        rewards_batch = tf.convert_to_tensor([exp.rewards for exp in experiences], dtype=tf.float32)
        next_states_batch = tf.convert_to_tensor([exp.next_states for exp in experiences], dtype=tf.float32)    
        
        # Call the train step
        self.run_train_step(states_batch, actions_batch, rewards_batch, next_states_batch)
        

    # Single training step implementation
    def _run_train_step(self, states, actions, rewards, next_states):
        # Compute target Q-values
        q_s_a_prime = tf.reduce_max(self.target_net(next_states), axis=1)
        q_s_a_target = tf.cast(rewards, tf.float32) + Globals.delta * q_s_a_prime

        # Gradient descent step
        with tf.GradientTape() as tape:
            q_s_a = tf.reduce_sum(self.policy_net(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.reduce_mean(tf.square(q_s_a_target - q_s_a))
        
        #Apply gradients
        gradients = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_net.trainable_variables))

    # Compute average greedy wage for analysis purposes
    def compute_av_greedy_wage_for_analysis(self):

        test_states = []

        if Globals.model_type == 0:
            for w1 in self.wage_list:
                for w2 in self.wage_list:
                    test_states.append([w1, w2])
        elif Globals.model_type == 1:
            for w1 in self.wage_list:
                test_states.append([w1])

        # Batch process all states at once
        test_states = tf.convert_to_tensor(test_states, dtype=tf.float32)
        q_values_batch = self.policy_net(test_states)

        # Select actions and calculate average greedy wage
        actions = np.argmax(q_values_batch, axis=1)
        self.av_greedy_wage_for_analysis = np.mean([self.wage_list[action] for action in actions])



# --- Enable tabular Firms when requested ---
try:
    import Globals
    if getattr(Globals, "USE_QTABLE", 0):
        # Replace exported Firms symbol with tabular version
        from QTable_Agent import QTableFirms as Firms
except Exception:
    # If anything goes wrong, keep the DQN Firms as-is
    pass
