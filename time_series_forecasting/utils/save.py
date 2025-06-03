def freq_add(self, x, y, rate=0.6):
        """
        Perturb a single randomly selected low-frequency component per channel by setting its magnitude
        to half of the maximum magnitude per channel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, time_steps, channels)
            y (torch.Tensor): Label tensor of shape (batch_size, time_steps, channels)
            
        Returns:
            torch.Tensor: Augmented concatenated tensor of shape (batch_size, total_time_steps, channels)
        """
        # Input validation
        if x.dim() != 3 or y.dim() != 3:
            raise ValueError("Input tensors must be 3-dimensional (batch_size, time_steps, channels)")
        if x.size(0) != y.size(0) or x.size(2) != y.size(2):
            raise ValueError("Input tensors must have matching batch size and channel dimensions")
        
        # Concatenate data and labels along the time dimension
        xy = torch.cat([x, y], dim=1)  # Shape: (B, T_total, D)
        device = xy.device
        
        # Convert to frequency domain
        X = torch.fft.rfft(xy, dim=1)  # Shape: (B, num_freqs, D)
        
        # Compute magnitude and maximum magnitude per channel
        mag = torch.abs(X)  # Shape: (B, num_freqs, D)
        max_mag = mag.amax(dim=1, keepdim=True)  # Shape: (B, 1, D)
        
        # Get dimensions
        batch_size, num_freqs, num_channels = X.shape
        
        # Define low-frequency region (first half of spectrum, excluding DC)
        N_low = num_freqs // 2
        low_freq_indices = torch.arange(1, N_low, device=device)
        num_low_freqs = len(low_freq_indices)
        
        # Randomly select one low frequency per channel per batch
        selected_indices = torch.randint(0, num_low_freqs, 
                                       (batch_size, num_channels), 
                                       device=device)  # Shape: (B, D)
        freq_indices = low_freq_indices[selected_indices]  # Shape: (B, D)
        
        # Create index tensors for batch and channel dimensions
        batch_indices = torch.arange(batch_size, device=device)\
                            .unsqueeze(1)\
                            .expand(-1, num_channels)  # Shape: (B, D)
        channel_indices = torch.arange(num_channels, device=device)\
                              .unsqueeze(0)\
                              .expand(batch_size, -1)  # Shape: (B, D)
        
        # Get original phase at selected frequencies
        orig_phase = torch.angle(X[batch_indices, freq_indices, channel_indices])
        
        # Calculate new magnitude (half of maximum) per channel
        new_mag = max_mag.squeeze(1) * 0.5  # Shape: (B, D)
        
        # Reconstruct complex coefficients using new magnitude and original phase
        X[batch_indices, freq_indices, channel_indices] = new_mag * torch.exp(1j * orig_phase)
        
        # Convert back to time domain
        xy = torch.fft.irfft(X, n=xy.size(1), dim=1)
        
        return xy
        