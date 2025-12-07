from model import vfhnn, decoder_prediction
import importlib.util
import os
import numpy as np
import torch

def load_config_from_py(config_path):
    """Dynamically import a config.py file as a module."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def load_model(model_dir, device): # TODO: load encoder as well this time
    '''
        Loads model and config file from model_dir 
    '''
    config_path = os.path.join(model_dir, "config.py")
    config = load_config_from_py(config_path)

    target_length = getattr(config, "target_length", 1)
    hidden_size = getattr(config, "hidden_size", 6)
    output_size = getattr(config, "output_size", 8)
    dropout = getattr(config, "dropout", 0)
    enc_dim_list = getattr(config, "enc_dim_list", [7, 2])
    input_shape = 76

    encoder = vfhnn(
        input_channels=input_shape,
        hidden_size=hidden_size,
        output_channels=output_size,
        target_length=target_length,
        dropout=dropout,
        enc_dim_list=enc_dim_list,
        device=device).to(device)

    decoder = decoder_prediction(
        input_channels=input_shape,
        hidden_size=hidden_size,
        output_channels=output_size,
        target_length=target_length,
        dropout=dropout,
        device=device).to(device)
    
    ckpt_path = os.path.join(model_dir, "all_finetuned_vfhnn")
    encoder.load_state_dict(torch.load(ckpt_path, map_location=device)[0][0])
    decoder.load_state_dict(torch.load(ckpt_path, map_location=device)[0][1])
    
    encoder.to(device).eval()
    decoder.to(device).eval()

    return encoder, decoder, config, device

def get_model_latent(site_ids, base_path, num_layers=3):
    '''
        Function to extract mu and var of encoded layers, similar to analyze_latent_activity
    ''' 
    all_layer_mu = [[] for _ in range(num_layers)]  # track mu and logvar over all layers for all sites at all steps
    all_layer_logvar = [[] for _ in range(num_layers)]
    forecast_steps_list = []

    for site_id in site_ids:
        file_path = os.path.join(base_path, f"{site_id}_latent_space_logs.npz")
        if not os.path.exists(file_path):
            print(f"Warning: File not found for site {site_id}")
            continue
        mu = np.load(file_path)["mu"]  # shape: (T, S, L, H)
        mu = mu.squeeze()
        forecast_steps_list.append(mu.shape[0])

    min_steps = min(forecast_steps_list)
    print(f"Using minimum forecast steps: {min_steps}")

    # load values
    for site_id in site_ids:
        file_path = os.path.join(base_path, f"{site_id}_latent_space_logs.npz")
        if not os.path.exists(file_path):
            print(f"Warning: File not found for site {site_id}")
            continue

        mu = np.load(file_path)["mu"]  # shape: (layers, site, forecast_steps, hidden_size)
        mu = mu.squeeze()[:min_steps]

        logvar = np.load(file_path)["logvar"]
        logvar = logvar.squeeze()[:min_steps]
        
        for layer in range(num_layers):
            all_layer_mu[layer].append(mu[:, layer, :])
            all_layer_logvar[layer].append(logvar[:, layer, :])

    all_layer_mu = np.stack(all_layer_mu)
    all_layer_logvar = np.stack(all_layer_logvar)
    print(all_layer_mu.shape, all_layer_logvar.shape)
    
    return all_layer_mu, all_layer_logvar, min_steps

def get_activity_latent(all_layer_mu):
    return  np.var(all_layer_mu, axis=(1, 2))

def sample_latent_dim_per_req(mu_list, logvar_list, layer_idx, dim_idx, offset):    
    std_dev = np.exp(logvar_list[layer_idx][:, :, dim_idx] / 2)    
    shifted_mu = mu_list[layer_idx, :, :, dim_idx] + offset * std_dev 
    print("Original μ:", mu_list)
    print("Shifted μ:", shifted_mu)
    print("Difference:", (shifted_mu - mu_list[layer_idx, :, :, dim_idx]).mean())
    modified_mu = mu_list.copy() 
    modified_mu[layer_idx, :, :, dim_idx] = shifted_mu
    
    return modified_mu, logvar_list

