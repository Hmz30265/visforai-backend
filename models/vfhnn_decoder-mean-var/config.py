# Config for fhnn
eps = 1e-10
weighted_loss = False
multiple_linear_head = False
refinetuning = False # Almost always want this true for fhnn


# Train epoch information
pt_num_epochs = 50
ft_num_epochs = 350
pt_batch_size = 1024
ft_batch_size = 1024
pt_optim_lr = 5e-3
ft_optim_lr = 1e-3
pt_beta = 1.0
ft_beta = 1.0

# Sample information
sequence_length = 360
target_length = 8 # 1 for AR, prediction_length otherwise
prediction_length = 8

# Forecast information
n_mc_samples = 1
n_z_samples = 100

# Model information
output_size = 1
hidden_size = 24
dropout = 0.0
num_layers = 1 # only used for LSTMModel
enc_dim_list = [7, 2] 

# Site and train type information
model_name = "vfhnn" # "fhnn_add_enc", "LSTMModel", "vfhnn"
hyperparameter_search = False
experiment_name = "70_sites"

# Model ensemble variable
model_num = 1

output = f"../res/11_07_2025/target-{target_length}/prediction-{prediction_length}" 

# Config name for moving config file to result folder
config_name = f"config_{model_name}_target-len-{target_length}"