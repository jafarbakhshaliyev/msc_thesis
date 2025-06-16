# Adapted from dominant-shuffle (https://github.com/zuojie2024/dominant-shuffle) and FrAug implementations
# Original works: Dominant Shuffle by Kai Zhao et al., FrAug by Muxi Chen et al.
# Modified and extended by Jafar Bakhshaliyev (2025)

import torch
import torch.nn as nn
import numpy as np
import random
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import torch.nn.functional as F
from typing import List, Tuple
from joblib import Parallel, delayed
from dtaidistance import dtw
from statsmodels.tsa.seasonal import STL
from arch.bootstrap import MovingBlockBootstrap

def augmentation(augment_time):
    if augment_time == 'batch':
        return BatchAugmentation()
    elif augment_time == 'dataset':
        return DatasetAugmentation()


class BatchAugmentation():
    def __init__(self):
        pass

    def flipping(self,x,y,rate=0):
        xy = torch.cat([x,y],dim=1)
        idxs = np.arange(xy.shape[1])
        idxs = list(idxs)[::-1]
        xy = xy[:,idxs,:]
        return xy

    def warping(self,x,y,rate=0):
        xy = torch.cat([x,y],dim=1)
        new_xy = torch.zeros_like(xy)
        for i in range(new_xy.shape[1]//2):
            new_xy[:,i*2,:] = xy[:,i + xy.shape[1]//2,:]
            new_xy[:,i*2 + 1,:] = xy[:,i + xy.shape[1]//2,:]
        return xy

    def noise(self,x,y,rate=0.05):
        xy = torch.cat([x,y],dim=1)
        noise_xy = (torch.rand(xy.shape)-0.5) * 0.1
        xy = xy + noise_xy.cuda()
        return xy

    def noise_input(self,x,y,rate=0.05):
        noise = (torch.rand(x.shape)-0.5) * 0.1
        x = x + noise.cuda()
        xy = torch.cat([x,y],dim=1)
        return xy

    def masking(self,x,y,rate=0.5):
        xy = torch.cat([x,y],dim=1)
        b_idx = np.arange(xy.shape[1])
        np.random.shuffle(b_idx)
        crop_num = int(xy.shape[1]*rate) # rate was 0.5
        xy[:,b_idx[:crop_num],:] = 0
        return xy

    def masking_seg(self,x,y,rate=0.5):
        xy = torch.cat([x,y],dim=1)
        b_idx = int(np.random.rand(1)*xy.shape[1]//2) # can change to rate
        xy[:,b_idx:b_idx+xy.shape[1]//2,:] = 0
        return xy

    def freq_mask(self,x, y, rate=0.5, dim=1):
        xy = torch.cat([x,y],dim=1)
        xy_f = torch.fft.rfft(xy,dim=dim)
        m = torch.cuda.FloatTensor(xy_f.shape).uniform_() < rate
        amp = abs(xy_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 5
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=dim)
        return xy

    def freq_mix(self, x, y, rate=0.5, dim=1):
        xy = torch.cat([x,y],dim=dim)
        xy_f = torch.fft.rfft(xy,dim=dim)
        
        m = torch.cuda.FloatTensor(xy_f.shape).uniform_() < rate
        amp = abs(xy_f)
        _,index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 2
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        
        b_idx = np.arange(x.shape[0])
        np.random.shuffle(b_idx)
        x2, y2 = x[b_idx], y[b_idx]
        xy2 = torch.cat([x2,y2],dim=dim)
        xy2_f = torch.fft.rfft(xy2,dim=dim)

        m = torch.bitwise_not(m)
        freal2 = xy2_f.real.masked_fill(m,0)
        fimag2 = xy2_f.imag.masked_fill(m,0)

        freal += freal2
        fimag += fimag2

        xy_f = torch.complex(freal,fimag)
        
        xy = torch.fft.irfft(xy_f,dim=dim)
        return xy
    
    def dom_shuffle(self, x, y, rate=4, dim=1):
        xy = torch.cat([x,y],dim=1)
        xy_f = torch.fft.rfft(xy,dim=dim)

        magnitude = abs(xy_f)
        
        topk_indices = torch.argsort(magnitude, dim=1, descending=True)[:, 1:int(rate+1)]        
        #minor_indices = torch.argsort(magnitude, dim=1, descending=True)[:, 10:]  
              
        new_xy_f = xy_f
        for i in range(topk_indices.size(2)):
            for j in range(xy.size(0)):
                    
                random_indices = torch.randperm(topk_indices.size(1))                
                shuffled_tensor1 = topk_indices[:,:,i][j][random_indices]   
                new_xy_f[:,:,i][j][topk_indices[:,:,i][j]] = new_xy_f[:,:,i][j][shuffled_tensor1]
                            
        xy = torch.fft.irfft(new_xy_f,dim=dim)
        
        return xy

    def wave_mask(self, x, y, rates, wavelet='db1', level=2, uniform = False, sampling_rate=0.5, dim=1):

        xy = torch.cat([x, y], dim=1)  # (B, T, C)

        if uniform:
            rates = [rates[0]] * (level + 1)

        rate_tensor = torch.tensor(rates, device=x.device) 
        xy = xy.permute(0, 2, 1).to(x.device) # (B, C, T)

        dwt = DWT1DForward(J=level, wave=wavelet, mode='symmetric').to(x.device)
        cA, cDs = dwt(xy)

        mask = torch.rand_like(cA).to(cA.device) < rate_tensor[0]
        cA = cA.masked_fill(mask, 0)
        masked_cDs = []
        for i, cD in enumerate(cDs):
            mask_cD = torch.rand_like(cD).to(cD.device) < rate_tensor[i+1]  # Create mask
            cD = cD.masked_fill(mask_cD, 0)
            masked_cDs.append(cD)

        idwt = DWT1DInverse(wave=wavelet, mode='symmetric').to(x.device)
        xy = idwt((cA, masked_cDs))
        xy = xy.permute(0, 2, 1) # (B, T, C)

        # Sample a subset of batches
        batch_size = xy.shape[0]
        sampling_steps = int(batch_size * sampling_rate)
        indices = torch.randperm(batch_size)[:sampling_steps]

        xy = xy[indices]

        return xy, indices

    def wave_mix(self, x, y, rates, wavelet='db1', level=2, uniform = False, sampling_rate=0.5, dim=1):
        xy = torch.cat([x, y], dim=1)  # (B, T, C)
        batch_size, _, _ = xy.shape

        if uniform: 
            rates = [rates[0]] * (level + 1)
        rate_tensor = torch.tensor(rates, device=x.device) 

        xy = xy.permute(0, 2, 1).to(x.device) # (B, C, T)

        # Shuffle the batch for mixing
        b_idx = torch.randperm(batch_size)
        xy2 = xy[b_idx]

        dwt = DWT1DForward(J=level, wave=wavelet, mode='symmetric').to(x.device)
        cA1, cDs1 = dwt(xy)
        cA2, cDs2 = dwt(xy2)

        mask = torch.rand_like(cA1).to(cA1.device) < rate_tensor[0] # Create mask
        cA_mixed = cA1.masked_fill(mask, 0) + cA2.masked_fill(~mask, 0)
        mixed_cDs = []
        for i, (cD1, cD2) in enumerate(zip(cDs1, cDs2)):
            mask = torch.rand_like(cD1).to(cD1.device) < rate_tensor[i+1] # Create mask
            cD_mixed = cD1.masked_fill(mask, 0) + cD2.masked_fill(~mask, 0)
            mixed_cDs.append(cD_mixed)

        idwt = DWT1DInverse(wave=wavelet, mode='symmetric').to(x.device)
        xy = idwt((cA_mixed, mixed_cDs))
        xy = xy.permute(0, 2, 1) # (B, T, C)

        # Sample a subset of batches
        batch_size = xy.shape[0]
        sampling_steps = int(batch_size * sampling_rate)
        indices = torch.randperm(batch_size)[:sampling_steps]

        xy = xy[indices]

        return xy, indices



    def freq_add(self, x, y, rate = 0.5):
        """
        Perturb a single randomly selected low-frequency component per channel by setting its magnitude
        to half of the maximum magnitude per channel.
        """
        xy = torch.cat([x, y], dim=1)  # Shape: (B, T_total, D)
        device = xy.device
        X = torch.fft.rfft(xy, dim=1)  # Shape: (B, num_freqs, D)

        # Compute the magnitude and maximum magnitude per channel
        mag = torch.abs(X)  # Shape: (B, num_freqs, D)
        max_mag = mag.amax(dim=1, keepdim=True)  # Shape: (B, 1, D)
        num_freqs = X.size(1)

        # low-frequency indices (excluding DC component at index 0)
        N_low = num_freqs // 2  
        low_freq_indices = torch.arange(1, N_low, device=device)  # Indices from 1 to N_low - 1
        num_low_freqs = len(low_freq_indices)

        # Randomly select a low-frequency index per sample and per channel
        batch_size = x.size(0)
        num_channels = x.size(2)
        selected_indices = torch.randint(0, num_low_freqs, (batch_size, num_channels), device=device)  # Shape: (B, D)
        freq_indices = low_freq_indices[selected_indices]  # Shape: (B, D)
        
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_channels)  # Shape: (B, D)
        channel_indices = torch.arange(num_channels, device=device).unsqueeze(0).expand(batch_size, -1)  # Shape: (B, D)
        
        orig_phase = torch.angle(X[batch_indices, freq_indices, channel_indices])  # Shape: (B, D)
        
        # Set the magnitude to half of the maximum magnitude per channel
        new_mag = max_mag.squeeze(1) * rate  # Shape: (B, D)
        
        # Reconstruction
        X[batch_indices, freq_indices, channel_indices] = new_mag * torch.exp(1j * orig_phase)
        xy = torch.fft.irfft(X, n=xy.size(1), dim=1)  # Shape: (B, T_total, D)
        
        return xy
    

    def freq_pool(self, x, y, rate=0.5):
        """
        Implements FreqPool augmentation from Chen et al. (2023b).
        Applies maximum pooling to the entire frequency spectrum with given pool size.

        """
        pool_size=int(rate)
        xy = torch.cat([x, y], dim=1)  # Shape: (B, T_total, D)
        xy_f = torch.fft.rfft(xy, dim=1)  # Shape: (B, num_freqs, D)
        
        # Get real and imaginary components
        real = xy_f.real
        imag = xy_f.imag
        B, F, D = real.shape

        pad_size = (pool_size - (F % pool_size)) % pool_size
        if pad_size > 0:
            real = torch.nn.functional.pad(real, (0, 0, 0, pad_size))
            imag = torch.nn.functional.pad(imag, (0, 0, 0, pad_size))
        
        real = real.unsqueeze(1)  # Shape: (B, 1, F, D)
        imag = imag.unsqueeze(1)  # Shape: (B, 1, F, D)
        
        # max pooling layer
        max_pool = torch.nn.MaxPool2d(kernel_size=(pool_size, 1), stride=(pool_size, 1))
        
        # max pooling separately to real and imaginary parts
        real_pooled = max_pool(real)
        imag_pooled = max_pool(imag)
        
        real_pooled = real_pooled.squeeze(1)[:, :F, :]
        imag_pooled = imag_pooled.squeeze(1)[:, :F, :]
        
        # Reconstruction
        xy_f_pooled = torch.complex(real_pooled, imag_pooled)
        xy = torch.fft.irfft(xy_f_pooled, n=xy.size(1), dim=1)
        
        return xy


    def temporal_patch_shuffle(self, x, y, patch_len=16, stride=5, rate=0.5):
        """
        Divides the time series into patches using `unfold` and shuffles a proportion of them.
        Prioritizes shuffling patches with less critical information to preserve essential temporal dependencies.
        """
        xy = torch.cat([x, y], dim=1)  # (B, T, C)
        B, T, C = xy.shape

        xy = xy.permute(0, 2, 1)
        patches = xy.unfold(dimension=2, size=patch_len, step=stride)
        B, C, num_patches, _ = patches.shape

        patches = patches.permute(0, 2, 1, 3).contiguous() # (B, num_patches, C, patch_len)

        # importance scores based on variance
        importance_scores = patches.var(dim=(2, 3))  # (B, num_patches)

        # Decide which patches to shuffle based on shuffle rate
        num_to_shuffle = int(num_patches * rate)
        if num_to_shuffle > 0:
            for b in range(B):
                scores = importance_scores[b]
                sorted_indices = torch.argsort(scores)
                shuffle_indices = sorted_indices[:num_to_shuffle]

                # Shuffle selected patches
                patches_to_shuffle = patches[b, shuffle_indices]
                shuffled_order = torch.randperm(num_to_shuffle)
                patches[b, shuffle_indices] = patches_to_shuffle[shuffled_order]

        # Reconstruction
        patches = patches.permute(0, 2, 1, 3)
        reconstructed = torch.zeros(B, C, T, device=xy.device)
        counts = torch.zeros(B, C, T, device=xy.device)

        for i in range(num_patches):
            start = i * stride
            end = start + patch_len
            reconstructed[:, :, start:end] += patches[:, :, i]
            counts[:, :, start:end] += 1

        counts[counts == 0] = 1  # Prevent division by zero
        xy = reconstructed / counts

        # Permute back to (B, T, C)
        xy = xy.permute(0, 2, 1)

        return xy

    def robusttad_m(self, x, y, rate=0.5, m_K=3, segment_ratio=0.2):
        """
        RobustTAD magnitude augmentation with multiple segments.
        Args:
            rate: qA parameter controlling perturbation
            m_K: number of segments to perturb
        """        

        xy = torch.cat([x, y], dim=1)
        xy_f = torch.fft.rfft(xy, dim=1)  # N' = N//2 + 1 frequencies for real signal

        magnitude = torch.abs(xy_f)
        phase = torch.angle(xy_f)

        # Calculate segment length K = rN'
        B, N_prime, D = xy_f.shape  
        K = int(N_prime * segment_ratio)  # segment length

        # Ensure |ki - ki+1| >= K/2 for overlapping constraint
        valid_starts = []
        for _ in range(m_K):
            if not valid_starts:
                start = torch.randint(0, N_prime - K, (1,)).item()
            else:
                # Create mask for valid start positions
                mask = torch.ones(N_prime - K, dtype=torch.bool)
                for prev_start in valid_starts:
                    # minimum distance K/2 from previous segments
                    invalid_range_start = max(0, prev_start - K//2)
                    invalid_range_end = min(N_prime - K, prev_start + K//2)
                    mask[invalid_range_start:invalid_range_end] = False
                
                # valid positions
                valid_positions = torch.where(mask)[0]
                if len(valid_positions) == 0:
                    break  
                
                # Randomly select from valid positions
                start = valid_positions[torch.randint(0, len(valid_positions), (1,))].item()
            
            valid_starts.append(start)
        

        for start_idx in valid_starts:
            end_idx = start_idx + K
        
            magnitude_segment = magnitude[:, start_idx:end_idx, :]

            # segment statistics
            mu_bar_A = magnitude_segment.mean(dim=(1, 2), keepdim=True)
            delta_bar_A_sq = magnitude_segment.var(dim=(1, 2), unbiased=False, keepdim=True)

            
            mu = torch.zeros_like(mu_bar_A)

            # Sampling
            std = (rate * delta_bar_A_sq).sqrt()
            v = torch.normal(mean=mu, std=std)
            v = v.expand(B, K, D)
            
            # Replacing magnitude values in segment
            magnitude[:, start_idx:end_idx, :] = v

        # Reconstruction
        xy_f_aug = torch.polar(magnitude, phase)
        xy = torch.fft.irfft(xy_f_aug, n=xy.size(1), dim=1)

        return xy

    def robusttad_p(self, x, y, rate=0.5, m_K=3, segment_ratio=0.2):
        """
        RobustTAD phase augmentation with multiple segments.
        Args:
            rate: q_theta parameter controlling perturbation degree
            m_K: number of segments to perturb
        """
        
        xy = torch.cat([x, y], dim=1)
        xy_f = torch.fft.rfft(xy, dim=1)  # N' = N//2 + 1 frequencies for real signal


        magnitude = torch.abs(xy_f)
        phase = torch.angle(xy_f)

        # Calculate segment length K = rN'
        B, N_prime, D = xy_f.shape 
        K = int(N_prime * segment_ratio)  # segment length
        
  
        # Ensure |ki - ki+1| >= K/2 for overlapping constraint
        valid_starts = []
        for _ in range(m_K):
            if not valid_starts:
                start = torch.randint(0, N_prime - K, (1,)).item()
            else:
                mask = torch.ones(N_prime - K, dtype=torch.bool)
                for prev_start in valid_starts:
                    #  minimum distance K/2 from previous segments
                    invalid_range_start = max(0, prev_start - K//2)
                    invalid_range_end = min(N_prime - K, prev_start + K//2)
                    mask[invalid_range_start:invalid_range_end] = False
                # valid positions
                valid_positions = torch.where(mask)[0]
                if len(valid_positions) == 0:
                    break  
                
                # Randomly select from valid positions
                start = valid_positions[torch.randint(0, len(valid_positions), (1,))].item()
            
            valid_starts.append(start)
        

        for start_idx in valid_starts:
            end_idx = start_idx + K
            

            phase_segment = phase[:, start_idx:end_idx, :]
            delta_theta_sq = phase_segment.var(dim=(1, 2), unbiased=False, keepdim=True)

            # Sampling
            std_theta = (rate * delta_theta_sq).sqrt()
            theta = torch.normal(mean=torch.zeros_like(std_theta), std=std_theta)
            theta = theta.expand(B, K, D)
            
            # Add perturbation
            phase[:, start_idx:end_idx, :] += theta

        # Reconstruction
        xy_f_aug = torch.polar(magnitude, phase)
        xy = torch.fft.irfft(xy_f_aug, n=xy.size(1), dim=1)

        return xy

    def upsample(self, x, y, rate=0.1):
        """
        Upsample method based on Semenoglou et al. (2023).

        Parameters:
        - x: Input data tensor of shape (B, T_x, D)
        - y: Labels tensor of shape (B, T_y, D)
        - K: Number of points to select (must be less than T_total)
        - rate: If K is not provided, K = int(T_total * rate)
        
        Returns:
        - Augmented data tensors x_aug and y_aug of shapes matching x and y
        """
        xy = torch.cat([x, y], dim=1)  # Shape: (B, T_total, D)
        B, T_total, D = xy.shape
        K = int(T_total * rate)

        # randomly select start indices for each sample
        max_start = T_total - K
        start_indices = torch.randint(0, max_start + 1, (B,), device=xy.device)

        xy_aug_list = []


        for b in range(B):
            s = start_indices[b].item()
            # Extract the subsequence of length K
            subseq = xy[b, s:s+K, :]  # Shape: (K, D)
            subseq = subseq.unsqueeze(0).permute(0, 2, 1)  # Shape: (1, D, K)

            # linear interpolation
            upsampled = F.interpolate(subseq, size=T_total, mode='linear', align_corners=False)
            upsampled = upsampled.squeeze(0).permute(1, 0)  # Shape: (T_total, D)
            xy_aug_list.append(upsampled)

        xy = torch.stack(xy_aug_list, dim=0)  # Shape: (B, T_total, D)

        return xy

    def asd(self, x, y, rate = 0.5, top_k = 5, dtw_window_ratio = 0.1):
        """
        Perform ASD-based data augmentation on a batch of time series.

        x: (B, T_x, D)  # look-back window
        y: (B, T_y, D)  # horizon
          - both on the same device (CPU or GPU)

        Returns
        -------
        xy_aug : torch.Tensor
            Shape: (B, T_x+T_y, D)
            The newly synthesized samples for each of the B original samples.
            You can then concatenate these to your original data if desired.
        """
        device = x.device
        xy = torch.cat([x, y], dim=1) 
        B, T_total, D = xy.shape

        # Move to CPU for dtw calculation (dtaidistance uses NumPy/C)
        xy_np = xy.detach().cpu().numpy()  # shape: (B, T_total, D)

        # pairwise DTW distances
        distances = np.zeros((B, B), dtype=np.float64)

        # a simple approach: for each dimension, compute distance_matrix_fast, sum results.
        for d_idx in range(D):
            series_d = xy_np[:, :, d_idx]
            series_d = np.ascontiguousarray(series_d, dtype=np.double)

            window = None
            if dtw_window_ratio is not None:
                window = int(dtw_window_ratio * T_total)

            try:
                dist_d = dtw.distance_matrix_fast(series_d,
                                                  window=window,
                                                  parallel=True)
            except Exception as e:
                print(f"[ASD] Fallback (dimension={d_idx}): {e}, using distance_matrix()")
                dist_d = dtw.distance_matrix(series_d, window=window)

            distances += dist_d  

        if D > 1:
            distances /= D

        distances_t = torch.from_numpy(distances).float().to(device)  # shape (B, B)

        # for each sample i, get its top_k neighbors, build an exponentially weighted sum
        xy_syn_list = []

        for i in range(B):

            dist_i = distances_t[i].clone()
            dist_i[i] = float('inf')

            # top_k neighbors
            vals, idxs = torch.topk(dist_i, k=top_k, largest=False)  # shape: (top_k,)

            # Compute weights
            d_nn = vals[0].clamp(min=1e-8)
            factor = vals / d_nn  # ratio
            weights = torch.exp(torch.log(torch.tensor(rate, device=device)) * factor)
            weights = weights / weights.sum()  # normalize

            # Weighted sum of neighbors
            neighbors = xy[idxs]
            weighted_sum = (neighbors * weights.view(top_k, 1, 1)).sum(dim=0)  # (T_total, D)

            new_sample = weighted_sum

            xy_syn_list.append(new_sample)

        xy = torch.stack(xy_syn_list, dim=0)
        xy = xy.to(device, dtype=torch.float32)

        return xy

    def process_one_sample_cpu(self, sample_i_cpu, block_size=24, stl_period=7):
        """
        Process a single sample using arch.bootstrap.MovingBlockBootstrap properly.
        Args:
            sample_i_cpu: numpy array of shape (T, D)
            block_size: size of blocks for MBB
            stl_period: period for STL decomposition
        """
        
        T, D = sample_i_cpu.shape
        
        trends_np = np.zeros_like(sample_i_cpu)
        seasonals_np = np.zeros_like(sample_i_cpu)
        remainders_np = np.zeros_like(sample_i_cpu)

        # STL decomposition
        for d in range(D):
            series_d = sample_i_cpu[:, d]
            stl = STL(series_d, period=stl_period, robust=True)
            res = stl.fit()
            trends_np[:, d] = res.trend
            seasonals_np[:, d] = res.seasonal
            remainders_np[:, d] = res.resid

        # MBB to remainder series
        new_remainders = np.zeros_like(remainders_np)
        for d in range(D):
            bs = MovingBlockBootstrap(block_size, remainders_np[:, d], seed=None)
        
            for data in bs.bootstrap(1):
                new_remainders[:, d] = data[0][0]  # First bootstrap replicate
                break  # We only need one sample

        # Reconstruction
        new_sample = trends_np + seasonals_np + new_remainders
        return new_sample

    def mbb(self, x, y, block_size=24, stl_period=7):
        """
        Parallel MBB-based STL augmentation using arch.bootstrap.
        Args:
            x: input tensor (B, T_x, D)
            y: target tensor (B, T_y, D)
            block_size: size of moving blocks
            stl_period: period for STL decomposition
        """
        xy = torch.cat([x, y], dim=1) 
        device = xy.device
        dtype = xy.dtype
        B, T, D = xy.shape

        x_cpu = xy.detach().cpu().numpy()
        results = Parallel(n_jobs=-1)(
            delayed(self.process_one_sample_cpu)(
                x_cpu[i], block_size, stl_period
            )
            for i in range(B)
        )

        results_np = np.stack(results, axis=0)
        xy = torch.from_numpy(results_np).to(device=device, dtype=dtype)
        
        return xy
    

class DatasetAugmentation():
    def __init__(self):
        pass

    def freq_dropout(self, x, y, dropout_rate=0.2, dim=0, keep_dominant=True):
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        xy = torch.cat([x,y],dim=0)
        print('AAAA', x.shape, y.shape)
        xy_f = torch.fft.rfft(xy,dim=0)

        m = torch.FloatTensor(xy_f.shape).uniform_() < dropout_rate

        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=dim)

        x, y = xy[:x.shape[0],:].numpy(), xy[-y.shape[0]:,:].numpy()
        return x, y

    def freq_mix(self, x, y, x2, y2, dropout_rate=0.2):
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        xy = torch.cat([x,y],dim=0)
        xy_f = torch.fft.rfft(xy,dim=0)
        m = torch.FloatTensor(xy_f.shape).uniform_() < dropout_rate
        amp = abs(xy_f)
        _,index = amp.sort(dim=0, descending=True)
        dominant_mask = index > 2
        m = torch.bitwise_and(m,dominant_mask)
        freal = xy_f.real.masked_fill(m,0)
        fimag = xy_f.imag.masked_fill(m,0)
        

        x2, y2 = torch.from_numpy(x2), torch.from_numpy(y2)
        xy2 = torch.cat([x2,y2],dim=0)
        xy2_f = torch.fft.rfft(xy2,dim=0)

        m = torch.bitwise_not(m)
        freal2 = xy2_f.real.masked_fill(m,0)
        fimag2 = xy2_f.imag.masked_fill(m,0)

        freal += freal2
        fimag += fimag2

        xy_f = torch.complex(freal,fimag)
        xy = torch.fft.irfft(xy_f,dim=0)
        x, y = xy[:x.shape[0],:].numpy(), xy[-y.shape[0]:,:].numpy()
        return x, y
