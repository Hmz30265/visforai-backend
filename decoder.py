# Decoder Prediction
import torch
import torch.nn as nn

class decoder_prediction(torch.nn.Module):
    def __init__(self, input_channels, hidden_size, output_channels, target_length, dropout, device):
        super(decoder_prediction, self).__init__()
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