#Population Size
num_firms =2
num_workers = 104


#DQN or Q Table
USE_QTABLE = 0

OUT_DIR = "./"


#Hyperparameters
MINI_BATCH_SIZE = 32
MEMORY_SIZE = 100000
FREQ_UPDATE_TARGETNET =10000
LEARNING_RATE = 0.001
factor_num_nodes_hidden_layers = [2/3.0, 3/2.0]

num_hidden_layers = 2
optimizer = "SGD"
activation_function = "sigmoid"
input_normalization = 1
reward_normalization = 1 
delta_productivity = 0.0
alpha = 1.0
beta = 6e-5
delta = 0.95
effort = 0.2
fee = 0
share_fee = 0

track_data_for_table_1 = False

model_type = 0 # 0 take it 1 bidding
random_productivity = False
asymetric_productivities = False
max_iterations = 500000
learning_start = 5000
data_store_freq = 100

set_simulation_scenario = 2

save_special_data_for_single_run_analysis = False