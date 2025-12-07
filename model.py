# Decoder Prediction
import torch
import torch.nn as nn
import numpy as np

class vfhnn(torch.nn.Module):
    def __init__(self, input_channels, hidden_size, output_channels, target_length, dropout, enc_dim_list, device):
        super(vfhnn,self).__init__()
        # PARAMETERS
        self.input_channels = input_channels
        self.latent_code_dim = hidden_size
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.forecast = target_length
        self.enc_dim_list = enc_dim_list
        self.device = device
        
        # Encoder LAYERS
        self.encoder_daily = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
        self.inner_encoders = torch.nn.ModuleList()  
        for enc_dim in self.enc_dim_list:
            encoder_layer = torch.nn.LSTM(input_size=self.latent_code_dim, hidden_size=self.latent_code_dim, bidirectional=True, batch_first=True)
            self.inner_encoders.append(encoder_layer)
        self.out_encoder = torch.nn.Linear(in_features=self.hidden_size, out_features=self.input_channels)

        self.z_proj_linear = torch.nn.Linear(in_features=(len(enc_dim_list)+1)*self.hidden_size, out_features=self.hidden_size)

        # dropout layers
        self.dropout = torch.nn.Dropout(p=dropout)

        # linear layers used to infer mu and variance (in ea-cvi they use MLP but lets start with linear)
        # using different linear layers at each encoder would give better results
        self.mu_linears = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(1 + len(enc_dim_list))])
        self.logvar_linears = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size) for _ in range(1 + len(enc_dim_list))])

        self._init_weights()

    def _init_weights(self):
        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # GET SHAPES
        batch, window, num_channels = x.shape
        
        # ENCODER OPERATIONS
        x_encoder = x[:, :-self.forecast] # could move the indexing to be outside the class in the training epoch loop
        x_encoder, (h_daily, c_daily) = self.encoder_daily(x_encoder)
        x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
        h_daily, c_daily = torch.sum(h_daily, axis=0), torch.sum(c_daily, axis=0)

        mu_list, logvar_list, h_list, c_list = [], [], [h_daily], [c_daily]

        # Projection to get mu and var from daily layer
        mu = self.mu_linears[0](c_daily)
        logvar = self.logvar_linears[0](c_daily)
        mu_list.append(mu)
        logvar_list.append(logvar)
        
        for i, enc_dim in enumerate(self.enc_dim_list):
            x_encoder, (h_enc_dim, c_enc_dim) = self.inner_encoders[i](x_encoder.flip(1)[:, ::enc_dim].flip(1))
            x_encoder = x_encoder[:, :, :self.latent_code_dim] + x_encoder[:, :, self.latent_code_dim:]
            h_enc_dim, c_enc_dim = torch.sum(h_enc_dim, axis=0), torch.sum(c_enc_dim, axis=0)

            # projection to get mu, var from inner encoders
            mu = self.mu_linears[i+1](c_enc_dim)
            logvar = self.logvar_linears[i+1](c_enc_dim)
            mu_list.append(mu)
            logvar_list.append(logvar)

            # h and c for inner layers
            h_list.append(h_enc_dim)
            c_list.append(c_enc_dim)

        # stacking and summing the h and c states for each encoder layer
        h = torch.stack(h_list, dim=0).sum(dim=0) 
        c = torch.stack(c_list, dim=0).sum(dim=0)

        z, z_list = self.sampling(mu_list, logvar_list)

        return z, z_list, (mu_list, logvar_list), (h, c)

    def sampling(self, mu_list, logvar_list, mode="proj"):
        """
        Given list of mu and log variances, calculate std and reparameterize z for each temporal encoder.
        Returns:
            z: aggregated latent (sum or mean across encoders)
            z_list: tensor of z_i for each encoder [num_encoders, batch, latent_dim]
        """
        z_list = []
        for mu, logvar in zip(mu_list, logvar_list):
            z = self._reparameterize(mu, logvar)
            z_list.append(z)
        
        z_list = torch.stack(z_list, dim=0)  # Shape: [num_encoders, batch, latent_dim]
    
        if mode == "sum":
            z = z_list.sum(dim=0) 
        elif mode == "mean":
            z = z_list.mean(dim=0) 
        elif mode == "proj":
            # Permute z_list to shape [batch, num_encoders, latent_dim] for projection
            z_concat = z_list.permute(1, 0, 2) 
            z_concat = z_concat.reshape(z_concat.size(0), -1)  
            z = self.z_proj_linear(z_concat) 
    
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'sum', 'mean', or 'proj'.")
    
        return z, z_list
        
    # sampling function, to obtain z from mu and var
    # in testing it would be nice to optimize such that encoder is called once, sampling is done multiple times, and then decoder is used multiple times
    def _reparameterize(self, mu, logvar):
        """Return z = mu + sigma*eps"""
        std = torch.exp(logvar/2)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z
    
class decoder_prediction(torch.nn.Module):
    def __init__(self, input_channels, hidden_size, output_channels, 
                 target_length, dropout, device):
        super(decoder_prediction, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.forecast = target_length
        self.device = device
        
        self.decoder = torch.nn.LSTM(input_size=self.input_channels-self.output_channels, hidden_size=self.hidden_size, batch_first=True)
        
        # Separate output heads for mean and log-variance
        self.mu_decoders = torch.nn.ModuleList()
        self.logvar_decoders = torch.nn.ModuleList()
        
        for _ in range(target_length):
            self.mu_decoders.append(torch.nn.Linear(self.hidden_size, self.output_channels))
            self.logvar_decoders.append(torch.nn.Linear(self.hidden_size, self.output_channels))
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, h, c):
        batch, window, num_channels = x.shape
        
        x_decoder = x[:, -self.forecast:, :num_channels-1]
        x_decoder, _ = self.decoder(x_decoder, (torch.unsqueeze(h, axis=0), torch.unsqueeze(c, axis=0)))
        x_decoder = self.dropout(x_decoder)
        
        # Get both mu and logvar for each timestep
        mu_outputs = []
        logvar_outputs = []
        
        for t in range(self.forecast):
            mu_t = self.mu_decoders[t](x_decoder[:, t:t+1])
            logvar_t = self.logvar_decoders[t](x_decoder[:, t:t+1])
            mu_outputs.append(mu_t)
            logvar_outputs.append(logvar_t)
        
        mu = torch.cat(mu_outputs, dim=1).squeeze(-1)
        logvar = torch.cat(logvar_outputs, dim=1).squeeze(-1)
        
        return mu, logvar

# Old decoder class
class old_decoder_prediction(torch.nn.Module):
    def __init__(self, input_channels, hidden_size, output_channels, target_length, dropout, device):
        super(old_decoder_prediction, self).__init__()
        # PARAMETERS
        self.input_channels = input_channels
        self.latent_code_dim = hidden_size
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.forecast = target_length
        self.device = device

        # Decoder LSTM for prediction outputs
        self.decoder = torch.nn.LSTM(input_size=self.input_channels-self.output_channels, hidden_size=self.hidden_size, batch_first=True)
        
        # Create target_length output linear layers instead of manual creation
        self.out_decoders = torch.nn.ModuleList()
        for _ in range(target_length):
            decoder_layer = torch.nn.Linear(in_features=self.hidden_size, out_features=self.output_channels)
            self.out_decoders.append(decoder_layer)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, h, c): # h = z or c = z 
        '''
        Given weather drivers for forecast and latent space, decoder function creates predictions
        Hidden state (h) or Cell state (c) is replaced by the parameterized version respectively
        '''
        # DECODER OPERATIONS
        batch, window, num_channels = x.shape        
        # can attach z to the dynamic to feed with x
        # z_concat = z.repeat(1, self.forecast, 1).reshape(batch, self.forecast, -1)
        x_decoder = x[:, -self.forecast:, :num_channels-1]
        #x_decoder = torch.cat((x_decoder, z), axis=-1)
        x_decoder, _ = self.decoder(x[:, -self.forecast:, :num_channels-1], 
                                   (torch.unsqueeze(h, axis=0), torch.unsqueeze(c, axis=0)))
        x_decoder = self.dropout(x_decoder)
        
        # Apply each output decoder to its corresponding timestep
        outputs = []
        for t in range(self.forecast):
            out_t = self.out_decoders[t](x_decoder[:, t:t+1])
            outputs.append(out_t)
        
        # Concatenate all outputs along the timestep dimension
        out_decoder = torch.cat(outputs, dim=1)
        out_decoder = out_decoder.view(batch, self.forecast, self.output_channels)
        
        return out_decoder.squeeze(-1)
    

def VFHNN_ensemble_forecast(encoder_model, decoder_model, forecast_dataset, norms_Y, mu_array, logvar_array, min_steps, n_z_samples=10, prediction_length=8, device='cpu', eps=1e-10):
    """
    mu_array: numpy array of shape (layers, forecast_steps, hidden_dim)
    logvar_array: numpy array of shape (layers, forecast_steps, hidden_dim)
    """
    n_samples = len(forecast_dataset)
    
    pred_c_degree = {
        'pred': np.zeros((n_samples, n_z_samples, prediction_length)),
        'pred_var': np.zeros((n_samples, n_z_samples, prediction_length)),
        'pred_mask': [],
    }

    y_mean, y_std = norms_Y
    n_layers = mu_array.shape[0]

    for i in range(min_steps):
        batch_X, batch_Y = forecast_dataset[i]
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        
        if batch_X.isnan().any() or batch_Y.isnan().any():
            pred_c_degree['pred_mask'].append(False)
            continue

        pred_c_degree['pred_mask'].append(True)
        
        # Get encoder hidden states (we still need these for the decoder)
        _, _, distribution_lists, lstm_hidden_states = encoder_model(batch_X[np.newaxis,:,:])
        
        outputs_mu = []
        outputs_var = []

        # For this forecast step i, extract mu and logvar for all layers
        # The encoder's forward pass produces mu_list where each element has shape (batch_size, hidden_dim)
        # Since batch_size=1 for inference, we need (1, hidden_dim) for each layer
        mu_list = []
        logvar_list = []
        
        for layer in range(n_layers):
            # Extract for this forecast step and layer: shape (hidden_dim,)
            mu_np = mu_array[layer, i, :]
            logvar_np = logvar_array[layer, i, :]
            
            # Convert to tensor and add batch dimension: shape (1, hidden_dim)
            mu_tensor = torch.from_numpy(mu_np).float().unsqueeze(0).to(device)
            logvar_tensor = torch.from_numpy(logvar_np).float().unsqueeze(0).to(device)
            
            mu_list.append(mu_tensor)
            logvar_list.append(logvar_tensor)

        # Now mu_list is a list of n_layers tensors, each with shape (1, hidden_dim)
        # This matches what encoder.forward() produces

        for z_i in range(n_z_samples):
            # Pass as separate arguments, not as tuple
            z, z_list = encoder_model.sampling(mu_list, logvar_list, mode="proj")
            print(z)
            z, z_list = encoder_model.sampling(*distribution_lists, mode="proj")
            z_random = torch.randn(1, z.shape[-1]).to(device)
            mu_pred, logvar_pred = decoder_model(batch_X[np.newaxis,:,:], lstm_hidden_states[0], z)
            
            mu_pred = mu_pred.squeeze(0)
            var_pred = torch.exp(logvar_pred.squeeze(0))
            
            c_pred_mu = mu_pred.detach().cpu().numpy() * y_std + y_mean
            c_pred_var = var_pred.detach().cpu().numpy() * (y_std ** 2)
            
            outputs_mu.append(c_pred_mu)
            outputs_var.append(c_pred_var)
    
        pred_c_degree['pred'][i] = outputs_mu
        pred_c_degree['pred_var'][i] = outputs_var

    return pred_c_degree