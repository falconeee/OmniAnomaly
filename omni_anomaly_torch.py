
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class PlanarFlow(nn.Module):
    """
    Planar Normalizing Flow.
    z_f = z + u * h(w^T * z + b)
    """
    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.w = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1))
        self.u = nn.Parameter(torch.randn(1, dim))

    def h(self, x):
        return torch.tanh(x)

    def h_prime(self, x):
        return 1 - torch.tanh(x) ** 2

    def forward(self, z):
        # z: [batch, seq_len, dim] or [batch, dim]
        # We want to apply this to the last dimension.
        
        # Safe constraint for u to ensure invertibility
        # u^T * w > -1
        w_dot_u = torch.sum(self.w * self.u)
        m_w_dot_u = -1 + torch.log(1 + torch.exp(w_dot_u))
        u_hat = self.u + (m_w_dot_u - w_dot_u) * self.w / (torch.sum(self.w ** 2) + 1e-7)
        
        # Linear projection
        # Expand w, b, u for broadcasting
        # w: [1, dim]
        # z: [..., dim]
        
        # w^T * z + b
        # shape: [..., 1]
        lin = F.linear(z, self.w, self.b) 
        
        # f(z) = z + u * h(lin)
        f_z = z + u_hat * self.h(lin)
        
        # Log determinant of Jacobian
        # psi = h'(lin) * w
        psi = self.h_prime(lin) * self.w # [..., dim]
        
        # det = |1 + u^T * psi|
        det = 1 + torch.sum(u_hat * psi, dim=-1, keepdim=True)
        log_det = torch.log(torch.abs(det) + 1e-7)
        
        return f_z, log_det

class FlowSequential(nn.Sequential):
    def forward(self, z):
        log_det_sum = 0
        for modules in self:
            z, log_det = modules(z)
            log_det_sum = log_det_sum + log_det
        return z, log_det_sum

class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, rnn_num_layers=1):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        
        # h_for_q_z: RNN over x
        self.gru = nn.GRU(x_dim, hidden_dim, rnn_num_layers, batch_first=True)
        
        # Processing RNN output
        self.post_rnn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim), # To match hidden_dense=2 logic roughly 
            nn.LeakyReLU(0.1)
        )
        
        # q_z_given_x parameters: predicts mu and sigma from (rnn_out, z_prev)
        self.z_mean = nn.Linear(hidden_dim + z_dim, z_dim)
        self.z_std = nn.Linear(hidden_dim + z_dim, z_dim)
        
    def forward(self, x, n_samples=1):
        # x: [batch, seq_len, x_dim]
        batch_size, seq_len, _ = x.size()
        
        # Pass x through RNN
        # h_seq: [batch, seq_len, hidden_dim]
        h_seq, _ = self.gru(x)
        
        # Post-process hidden states
        h_seq = self.post_rnn(h_seq)
        
        # Autoregressive generation of z
        # We need to sample z_t sequentially
        
        z_samples = []
        mu_list = []
        std_list = []
        
        # Initial z_prev (zeros or learned parameter)
        z_prev = torch.zeros(batch_size, self.z_dim).to(x.device)
        
        for t in range(seq_len):
            # context from x at time t
            h_t = h_seq[:, t, :] # [batch, hidden_dim]
            
            # Input to dist networks
            inp = torch.cat([h_t, z_prev], dim=1) # [batch, hidden_dim + z_dim]
            
            mu = self.z_mean(inp)
            std = F.softplus(self.z_std(inp)) + 1e-4
            
            # Sample z_t
            # epsilon ~ N(0, 1)
            eps = torch.randn_like(mu)
            z_t = mu + std * eps
            
            z_samples.append(z_t)
            mu_list.append(mu)
            std_list.append(std)
            
            z_prev = z_t
            
        # Stack
        z_out = torch.stack(z_samples, dim=1) # [batch, seq_len, z_dim]
        mu_out = torch.stack(mu_list, dim=1)
        std_out = torch.stack(std_list, dim=1)
        
        return z_out, mu_out, std_out

class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim, rnn_num_layers=1):
        super(Decoder, self).__init__()
        
        # h_for_p_x: RNN over z
        self.gru = nn.GRU(z_dim, hidden_dim, rnn_num_layers, batch_first=True)
        
        self.post_rnn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # p_x_given_z parameters
        self.x_mean = nn.Linear(hidden_dim, x_dim)
        self.x_std = nn.Linear(hidden_dim, x_dim)
        
    def forward(self, z):
        # z: [batch, seq_len, z_dim]
        
        h_seq, _ = self.gru(z)
        h_seq = self.post_rnn(h_seq)
        
        mu = self.x_mean(h_seq)
        std = F.softplus(self.x_std(h_seq)) + 1e-4
        
        return mu, std

class OmniAnomaly(nn.Module):
    def __init__(self, x_dim, z_dim=3, hidden_dim=500, window_length=100, nf_layers=20):
        super(OmniAnomaly, self).__init__()
        
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.window_length = window_length
        
        self.encoder = Encoder(x_dim, z_dim, hidden_dim)
        self.decoder = Decoder(x_dim, z_dim, hidden_dim)
        
        if nf_layers > 0:
            flows = [PlanarFlow(z_dim) for _ in range(nf_layers)]
            self.flow = FlowSequential(*flows)
        else:
            self.flow = None
            
    def forward(self, x):
        # x: [batch, seq_len, x_dim]
        
        # Encode -> q(z|x)
        z_gen, z_mu, z_std = self.encoder(x)
        
        # Normalizing Flow
        log_det_jac = 0
        if self.flow is not None:
            # Flatten for flow? Flow works on last dim, so [batch, seq, z_dim] is fine logic-wise 
            # if we consider each timestep independent for the flow parameters.
            # But the PlanarFlow implementation above supports arbitrary leading dims.
            z_fin, log_det_jac = self.flow(z_gen)
        else:
            z_fin = z_gen
            
        # Decode -> p(x|z)
        # Reconstruct from the Transformed z
        x_rec_mu, x_rec_std = self.decoder(z_fin)
        
        return {
            'x_rec_mu': x_rec_mu,
            'x_rec_std': x_rec_std,
            'z_gen': z_gen, # pre-flow
            'z_fin': z_fin, # post-flow
            'z_mu': z_mu,
            'z_std': z_std,
            'log_det_jac': log_det_jac
        }

    def loss_function(self, x, output):
        # ELBO = E_q [ log p(x|z) + log p(z) - log q(z|x) ]
        
        x_rec_mu = output['x_rec_mu']
        x_rec_std = output['x_rec_std']
        z_gen = output['z_gen']
        z_fin = output['z_fin']
        z_mu = output['z_mu']
        z_std = output['z_std']
        log_det_jac = output['log_det_jac'] # log |det dz_fin/dz_gen|
        
        # 1. Reconstruction Loss: log p(x|z)
        # p(x|z) ~ N(x_rec_mu, x_rec_std)
        # We assume independent across dims and time
        # log_prob = -0.5 * log(2pi) - log(std) - 0.5 * ((x - mu)/std)^2
        # We can use Gaussian negative log likelihood (MSE if std=1)
        recon_dist = torch.distributions.Normal(x_rec_mu, x_rec_std)
        log_p_x_given_z = recon_dist.log_prob(x).sum(dim=-1).sum(dim=-1) # Sum over dims and time
        
        # 2. Log q(z|x)
        # q(z_gen|x) ~ N(z_mu, z_std)
        # log q(z_fin|x) = log q(z_gen|x) - log |det Jacobian|
        # Auto-regressive structure handled by sampling logic
        q_dist = torch.distributions.Normal(z_mu, z_std)
        log_q_z_gen = q_dist.log_prob(z_gen).sum(dim=-1).sum(dim=-1)
        log_q_z_fin = log_q_z_gen - log_det_jac.sum(dim=-1).sum(dim=-1) if isinstance(log_det_jac, torch.Tensor) else log_q_z_gen
        
        # 3. Log p(z)
        # Prior is Random Walk: z_t = z_{t-1} + N(0, I)
        # p(z_{1:T}) = p(z_1) * prod_{t=2}^T p(z_t | z_{t-1})
        # p(z_1) ~ N(0, I)
        # p(z_t | z_{t-1}) ~ N(z_{t-1}, I)
        
        # Shift z_fin to get z_{t-1}
        # z_fin: [batch, seq, dim]
        z_t = z_fin
        z_t_minus_1 = torch.cat([torch.zeros(z_t.size(0), 1, z_t.size(2)).to(z_t.device), z_t[:, :-1, :]], dim=1)
        
        # log p(z_t | z_{t-1})
        # mean = z_{t-1}, std = 1
        prior_dist = torch.distributions.Normal(z_t_minus_1, torch.ones_like(z_t))
        log_p_z = prior_dist.log_prob(z_t).sum(dim=-1).sum(dim=-1)
        
        # ELBO (maximize) -> Loss (minimize -ELBO)
        elbo = log_p_x_given_z + log_p_z - log_q_z_fin
        loss = -elbo.mean()
        
        return loss

# --- Helper Functions ---

def preprocess_dataframe(df_train, df_test, window_size=100):
    """
    Convert dataframes to sliding windows.
    Assumes numerical columns only.
    Normalize data based on TRAIN stats.
    """
    # 1. Normalize
    mean = df_train.mean()
    std = df_train.std()
    
    # Avoid div by zero
    std[std == 0] = 1.0
    
    train_norm = (df_train - mean) / std
    test_norm = (df_test - mean) / std
    
    # 2. Sliding Window
    def create_windows(data, length):
        windows = []
        # data is numpy array: [N, dims]
        for i in range(len(data) - length + 1):
            windows.append(data[i : i+length])
        return np.array(windows)
        
    x_train = create_windows(train_norm.values, window_size)
    x_test = create_windows(test_norm.values, window_size)
    
    return x_train, x_test, mean, std

class TimeSeriesDataset(Dataset):
    def __init__(self, x):
        self.x = torch.FloatTensor(x)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx]

def train_model(model, train_loader, optimizer, epochs=10, device='cpu'):
    model.to(device)
    model.train()
    
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            output = model(batch_x)
            loss = model.loss_function(batch_x, output)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
    return loss_history

def get_anomaly_scores(model, test_loader, device='cpu'):
    """
    Get anomaly scores using reconstruction probability (or simpler MSE for now, 
    but OmniAnomaly uses reconstruction prob).
    """
    model.eval()
    scores = []
    
    with torch.no_grad():
        for batch_x in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            
            # Reconstruction Probability Score
            # log p(x|z)
            x_rec_mu = output['x_rec_mu']
            x_rec_std = output['x_rec_std']
            
            # Log prob of the LAST point in the window (as per typical anomaly detection settings)
            # or average over window? 
            # Original code: "if last_point_only: r_prob = r_prob[:, -1]"
            
            dist = torch.distributions.Normal(x_rec_mu, x_rec_std)
            log_prob = dist.log_prob(batch_x) # [batch, seq, dims]
            
            # Sum log probs over dimensions
            log_prob_dim = log_prob.sum(dim=-1) # [batch, seq]
            
            # Take last point
            score = -log_prob_dim[:, -1] # Negative log prob as anomaly score (higher = more anomalous)
            
            scores.append(score.cpu().numpy())
            
    return np.concatenate(scores)

