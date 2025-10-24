
# QTable_Agent.py
# A drop-in replacement for the DQN-based Firms agent in Firm.py,
# implementing tabular Q-learning. Works for both model types:
#   - model_type = 0 : "take-it-or-leave-it" (state = all firms' wages)
#   - model_type = 1 : "bidding"            (state = own wage only)
#
#
#
from __future__ import annotations
import numpy as np
import random, math
from collections import defaultdict, namedtuple, deque
import tensorflow as tf
import Globals

# Experience tuple used when a replay-like update is desired
Experience = namedtuple("Experience", ["s_key", "action", "reward", "s_next_key"])

def _nearest_index(value, grid):
    """
    Map a (float) wage to the nearest index in wage_list (grid).
    Assumes grid is sorted ascending.
    """
    # Quick path if exact
    try:
        return grid.index(value)
    except ValueError:
        # find nearest
        arr = np.asarray(grid, dtype=float)
        idx = int(np.argmin(np.abs(arr - float(value))))
        return idx


class _PolicyShim:
    """
    Shim to mimic a tf.keras.Model:
      - __call__(inputs) -> tf.Tensor of Q-values
      - attributes: built, input_shape
      - methods: get_weights/set_weights (no-ops)
    """
    def __init__(self, q_func, num_inputs, num_actions):
        self._q_func = q_func
        self.built = False
        self.input_shape = (None, int(num_inputs))
        self._num_actions = int(num_actions)

    def __call__(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if len(x.shape) == 1:
            q = self._q_func(x.numpy())
            out = tf.convert_to_tensor(q, dtype=tf.float32)
            self.built = True
            return out
        elif len(x.shape) == 2:
            qs = []
            for i in range(x.shape[0]):
                q = self._q_func(x[i].numpy())
                qs.append(tf.convert_to_tensor(q, dtype=tf.float32))
            out = tf.stack(qs, axis=0)
            self.built = True
            return out
        else:
            raise ValueError("Unsupported input rank for policy shim: expected 1D or 2D.")

    # Keras compatibility used by copy_weights
    def get_weights(self):
        return []

    def set_weights(self, weights):
        # ignore; tabular agent has no weights
        self.built = True
class QTableFirms:
    def __init__(self, space, firm_id, position, wage_offer, productivity, wage_list, 
                 memory_size=None, mini_batch_size=None, **kwargs):
        # --- core attributes mirrored from DQN version ---
        self.space = space
        self.firm_id = firm_id
        self.position = position
        self.wage_offer = wage_offer
        self.action = 0
        self.greedy_wage = 0
        self.productivity = productivity
        self.base_productivity = productivity
        self.profits = 0.0
        self.current_iteration = 0
        self.av_wage_competitors = 0.0
        self.av_greedy_wage_for_analysis = 0.0

        self.wage_list = list(wage_list)
        self.num_actions = len(self.wage_list)

        # Worker interaction lists (Workers.* expects these)
        self.employment_list = []
        self.application_list = []
        self.filled_jobs = 0

        # --- RL parameters ---
        self.gamma = getattr(Globals, "delta", 0.95)               # discount
        self.alpha = getattr(Globals, "LEARNING_RATE", 0.1)        # step size
        self.beta  = getattr(Globals, "beta", 6e-5)                # epsilon decay
        self.learning_start = getattr(Globals, "learning_start", 0)

        # epsilon schedule: same as EpsilonGreedyStrategy
        self._epsilon = 1.0  # will be computed each step via beta & learning_start

        # For "experience replay"-like behavior (optional)
        self._mem_capacity = getattr(Globals, "MEMORY_SIZE", 10000)
        self._memory = deque(maxlen=self._mem_capacity)

        # State encoding helpers
        self._num_firms = getattr(Globals, "num_firms", 2)
        self._model_type = getattr(Globals, "model_type", 0)

        # Tabular Q: defaultdict(state_key -> np.array[num_actions])
        self.Q = defaultdict(lambda: np.zeros(self.num_actions, dtype=float))

        # Build a policy shim for DataHandling.save_q_values_over_time
        self.num_inputs = (self._num_firms if self._model_type == 0 else 1)
        self.policy_net = _PolicyShim(self._q_values_for_statevec, self.num_inputs, self.num_actions)
        self.target_net = _PolicyShim(self._q_values_for_statevec, self.num_inputs, self.num_actions)

    

    # ------------- lifecycle hooks used by Model.py -------------
    def update_iteration(self, iteration):
        self.current_iteration = iteration

    def firing(self):
        for worker in self.employment_list:
            worker.employment_status = 0
            worker.where_work = -1
            worker.pay_off = 0
        self.application_list.clear()
        self.employment_list.clear()
        self.filled_jobs = 0

    def set_productivity(self):
        if getattr(Globals, "random_productivity", False):
            dp = getattr(Globals, "delta_productivity", 0.0)
            rnd = random.uniform(-dp, dp)
            self.productivity = self.base_productivity * (1.0 + rnd)

    # ---------------- state handling ----------------
    def _state_key_from_space(self):
        """
        Build a *discrete* key for the current state.
        For model_type 0: tuple of indices of all firms' wages (sorted by firm_id).
        For model_type 1: tuple with only this firm's wage index.
        """
        all_firms = sorted(self.space.get_objects(QTableFirms if self._is_qtable_class() else type(self)), key=lambda f: f.firm_id)
        # Fallback: include firms of any class that expose wage_offer & firm_id
        if len(all_firms) == 0:
            all_firms = sorted([f for f in self.space if hasattr(f, "wage_offer") and hasattr(f, "firm_id")], key=lambda f: f.firm_id)

        if self._model_type == 0:
            idxs = tuple(_nearest_index(f.wage_offer, self.wage_list) for f in all_firms[:self._num_firms])
            # update competitor average (for stats only)
            comp = [f.wage_offer for f in all_firms if f.firm_id != self.firm_id]
            self.av_wage_competitors = sum(comp)/len(comp) if comp else 0.0
            return idxs
        else:
            # state is just own wage index
            comp = [f.wage_offer for f in all_firms if f.firm_id != self.firm_id]
            self.av_wage_competitors = sum(comp)/len(comp) if comp else 0.0
            idx = _nearest_index(self.wage_offer, self.wage_list)
            return (idx,)

    
    def set_current_state(self):
            import numpy as _np
            # Build numpy state vector as in DQN for compatibility
            if self._model_type == 0:
                all_firms = sorted(self.space.get_objects(QTableFirms if self._is_qtable_class() else type(self)), key=lambda f: f.firm_id)
                state = _np.zeros(self._num_firms, dtype=float)
                for i, f in enumerate(all_firms[:self._num_firms]):
                    state[i] = f.wage_offer
                comp = [f.wage_offer for f in all_firms if f.firm_id != self.firm_id]
                self.av_wage_competitors = sum(comp)/len(comp) if comp else 0.0
            else:
                state = _np.zeros(1, dtype=float)
                state[0] = self.wage_offer
                all_firms = sorted(self.space.get_objects(QTableFirms if self._is_qtable_class() else type(self)), key=lambda f: f.firm_id)
                comp = [f.wage_offer for f in all_firms if f.firm_id != self.firm_id]
                self.av_wage_competitors = sum(comp)/len(comp) if comp else 0.0

            self.current_state = tf.convert_to_tensor(state, dtype=tf.float32)
            self._s_key = self._state_key_from_space()

    def wage_offer_method(self):
        # epsilon schedule
        step = max(0, self.current_iteration - self.learning_start)
        self._epsilon = math.exp(-self.beta * step)

        if random.random() < self._epsilon:
            self.action = random.randrange(self.num_actions)
            self.greedy_wage = None
        else:
            self.action = int(np.argmax(self.Q[self._s_key]))
            self.greedy_wage = self.wage_list[self.action]

        self.wage_offer = self.wage_list[self.action]

    def hiring(self):
        if self._model_type == 0:
            for worker in self.application_list:
                self.employment_list.append(worker)
                worker.where_work = self.firm_id
                worker.employment_status = 1
        self.filled_jobs = len(self.employment_list)

    def calculate_profits(self):
        fee = getattr(Globals, "fee", 0.0)
        share_fee = getattr(Globals, "share_fee", 0.0)
        alpha = getattr(Globals, "alpha", 1.0)
        reward_norm = getattr(Globals, "reward_normalization", 1)

        self.profits = (
            self.productivity * (len(self.employment_list) ** alpha)
            - (self.wage_offer + self.wage_offer * fee * share_fee) * len(self.employment_list)
        )
        if reward_norm == 1:
            self.profits /= 1000.0

    def set_next_state(self):
        s_next_key = self._state_key_from_space()
        # Store experience for update (compatible with "learning_start" & optional replay)
        self._memory.append(Experience(self._s_key, self.action, self.profits, s_next_key))
        # overwrite current to next; Model will call training() next
        self._s_key = s_next_key

    # ---------------- learning ----------------
    def training(self):
        # If you want to mimic minibatch updates, sample a batch; else update last exp
        if len(self._memory) == 0:
            return

        # Determine batch size from Globals.MINI_BATCH_SIZE but do not exceed memory length
        mbs = max(1, min(getattr(Globals, "MINI_BATCH_SIZE", 1), len(self._memory)))
        batch = [self._memory[-1]] if mbs == 1 else random.sample(self._memory, mbs)

        for exp in batch:
            q_sa = self.Q[exp.s_key]
            q_next = self.Q[exp.s_next_key]
            target = float(exp.reward) + self.gamma * float(np.max(q_next))
            # Q-learning update
            a = int(exp.action)
            q_sa[a] = (1.0 - self.alpha) * q_sa[a] + self.alpha * target
            self.Q[exp.s_key] = q_sa

    # ---------------- analysis helpers ----------------
    def _q_values_for_statevec(self, state_vec_like):
        """
        For compatibility with DataHandling.save_q_values_over_time:
        Convert a given continuous/float "state" vector to our discrete key, then return Q-values.
        """
        x = np.asarray(tf.convert_to_tensor(state_vec_like, dtype=tf.float32).numpy(), dtype=float).ravel()
        if self._model_type == 0:
            # Expect num_firms entries
            idxs = tuple(_nearest_index(val, self.wage_list) for val in x[:self._num_firms])
        else:
            # Expect 1 entry: own wage
            idxs = ( _nearest_index(x[0], self.wage_list), )
        return np.copy(self.Q[idxs])

    def compute_av_greedy_wage_for_analysis(self):
        test_states = []
        if self._model_type == 0:
            for w1 in self.wage_list:
                for w2 in self.wage_list:
                    test_states.append([w1, w2])
        else:
            for w in self.wage_list:
                test_states.append([w])

        # Evaluate greedy action at each state (current Q-table)
        greedy_wages = []
        for s in test_states:
            qvals = self._q_values_for_statevec(s)
            a = int(np.argmax(qvals))
            greedy_wages.append(self.wage_list[a])
        self.av_greedy_wage_for_analysis = float(np.mean(greedy_wages)) if greedy_wages else 0.0

    # Utility to help get_objects(...) find our class regardless of import path
    @staticmethod
    def _is_qtable_class():
        return True
