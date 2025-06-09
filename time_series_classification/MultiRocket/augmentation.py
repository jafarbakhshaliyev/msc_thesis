import numpy as np
from tqdm import tqdm


def tips(x, y, patch_len=0, stride=0, shuffle_rate=0.0):
    """
    Temporal Index Patch Shuffle (TIPS) augmentation for time series classification.

    Parameters:
    -----------
    x : numpy.ndarray
        Input time series data of shape (n_samples, timesteps, n_features)
    y : numpy.ndarray
        Labels of shape (n_samples,) or (n_samples, n_classes) for one-hot encoded
    patch_len : int
        Length of each patch
    stride : int
        Stride between patches
    shuffle_rate : float
        Proportion of patches to shuffle (between 0 and 1)
        
    Returns:
    --------
    numpy.ndarray
        Augmented time series data with same shape as input
    """
    n_samples, T, n_features = x.shape
    
    # Convert one-hot encoded labels to class indices if necessary
    if len(y.shape) > 1:
        labels = np.argmax(y, axis=1)
    else:
        labels = y
        
    # Initialize output array
    ret = np.zeros_like(x)
    
    # Calculate required padding to avoid zeros at the end
    total_patches = (T - patch_len + stride - 1) // stride + 1
    total_len = (total_patches - 1) * stride + patch_len
    padding_needed = total_len - T
    
    # Process each sample
    for i in range(n_samples):
        # current sample and its class
        sample = x[i]  # shape: (timesteps, n_features)
        current_class = labels[i]
        
        # Find indices of all samples from the same class (excluding current sample)
        same_class_indices = np.where(labels == current_class)[0]
        same_class_indices = same_class_indices[same_class_indices != i]
        
        if len(same_class_indices) == 0:
            # If no other samples of same class, keep original
            ret[i] = sample
            continue
            
        # Apply padding if needed
        if padding_needed > 0:
            padded_sample = np.pad(sample, ((0, padding_needed), (0, 0)), mode='edge')
            T_padded = T + padding_needed
        else:
            padded_sample = sample
            T_padded = T
            
        num_patches = ((T_padded - patch_len) // stride) + 1
        
        # Create patches for current sample
        patches = np.zeros((num_patches, patch_len, n_features))
        for j in range(num_patches):
            start = j * stride
            patches[j] = padded_sample[start:start + patch_len]
            
        # Compute importance of each patch
        importance_scores = np.var(patches, axis=(1, 2))
        
        # number of patches to shuffle
        num_to_shuffle = int(num_patches * shuffle_rate)
        
        if num_to_shuffle > 0:
            # indices of least important patches
            shuffle_indices = np.argsort(importance_scores)[:num_to_shuffle]
            
            for idx, p_idx in enumerate(shuffle_indices):
                # Randomly select a sample from same class
                random_sample_idx = np.random.choice(same_class_indices)
                random_sample = x[random_sample_idx]
                
                # Apply padding to the random sample 
                if padding_needed > 0:
                    random_sample = np.pad(random_sample, ((0, padding_needed), (0, 0)), mode='edge')
                
                # Extract corresponding patch from random sample
                start = p_idx * stride
                random_patch = random_sample[start:start + patch_len]
                patches[p_idx] = random_patch
            
        # Reconstruct the time series
        reconstructed = np.zeros((T_padded, n_features))
        counts = np.zeros((T_padded, n_features))
        
        for j in range(num_patches):
            start = j * stride
            end = start + patch_len
            reconstructed[start:end] += patches[j]
            counts[start:end] += 1
            
        # Average overlapping patches and handle potential zeros
        # Use a mask to identify zero counts
        mask = counts == 0
        if np.any(mask):
            # Fill in zeros with nearest non-zero values
            for feat in range(n_features):
                feat_mask = mask[:, feat]
                if np.any(feat_mask):
                    # Get indices of zero and non-zero values
                    zero_indices = np.where(feat_mask)[0]
                    nonzero_indices = np.where(~feat_mask)[0]
                    
                    if len(nonzero_indices) > 0:
                        # Find nearest non-zero index for each zero index
                        for zero_idx in zero_indices:
                            nearest_idx = nonzero_indices[np.argmin(np.abs(nonzero_indices - zero_idx))]
                            reconstructed[zero_idx, feat] = reconstructed[nearest_idx, feat]
                            counts[zero_idx, feat] = 1
        
        # Avoid division by zero 
        counts[counts == 0] = 1
        reconstructed = reconstructed / counts
        
        # Remove padding
        ret[i] = reconstructed[:T]
        
    return ret

def tps(x, y, patch_len=0, stride=0, shuffle_rate=0.0):
    """
    Temporal Patch Shuffle (TPS) augmentation for time series classification.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input time series data of shape (n_samples, timesteps, n_features)
    patch_len : int
        Length of each patch
    stride : int
        Stride between patches
    shuffle_rate : float
        Proportion of patches to shuffle (between 0 and 1)
        
    Returns:
    --------
    numpy.ndarray
        Augmented time series data with same shape as input
    """
    n_samples, T, n_features = x.shape
        
    ret = np.zeros_like(x)
    
    # Calculate required padding to avoid zeros at the end
    total_patches = (T - patch_len + stride - 1) // stride + 1
    total_len = (total_patches - 1) * stride + patch_len
    padding_needed = total_len - T
    
    # Process each sample
    for i in range(n_samples):
        # current sample
        sample = x[i]  # shape: (timesteps, n_features)
        
        # Apply padding if needed
        if padding_needed > 0:
            padded_sample = np.pad(sample, ((0, padding_needed), (0, 0)), mode='edge')
            T_padded = T + padding_needed
        else:
            padded_sample = sample
            T_padded = T
            
        num_patches = ((T_padded - patch_len) // stride) + 1
        
        # Create patches for current sample
        patches = np.zeros((num_patches, patch_len, n_features))
        for j in range(num_patches):
            start = j * stride
            patches[j] = padded_sample[start:start + patch_len]
            
        # importance of each patch 
        importance_scores = np.var(patches, axis=(1, 2))
        
        # number of patches to shuffle
        num_to_shuffle = int(num_patches * shuffle_rate)
        
        if num_to_shuffle > 0:
            # indices of least important patches
            shuffle_indices = np.argsort(importance_scores)[:num_to_shuffle]
            
            # Shuffle these patches among themselves
            patches_to_shuffle = patches[shuffle_indices].copy()
            shuffled_order = np.random.permutation(num_to_shuffle)
            
            for idx, new_idx in enumerate(shuffled_order):
                patch_idx = shuffle_indices[idx]
                new_patch = patches_to_shuffle[new_idx]
                patches[patch_idx] = new_patch
            
        # Reconstruct the time series
        reconstructed = np.zeros((T_padded, n_features))
        counts = np.zeros((T_padded, n_features))
        
        for j in range(num_patches):
            start = j * stride
            end = start + patch_len
            reconstructed[start:end] += patches[j]
            counts[start:end] += 1
            
        # Average overlapping patches and handle potential zeros
        # Use a mask to identify zero counts
        mask = counts == 0
        if np.any(mask):
            # Fill in zeros with nearest non-zero values
            for feat in range(n_features):
                feat_mask = mask[:, feat]
                if np.any(feat_mask):
                    # Get indices of zero and non-zero values
                    zero_indices = np.where(feat_mask)[0]
                    nonzero_indices = np.where(~feat_mask)[0]
                    
                    if len(nonzero_indices) > 0:
                        # Find nearest non-zero index for each zero index
                        for zero_idx in zero_indices:
                            nearest_idx = nonzero_indices[np.argmin(np.abs(nonzero_indices - zero_idx))]
                            reconstructed[zero_idx, feat] = reconstructed[nearest_idx, feat]
                            counts[zero_idx, feat] = 1
        
        # Avoid division by zero
        counts[counts == 0] = 1
        reconstructed = reconstructed / counts
        
        # Remove padding 
        ret[i] = reconstructed[:T]
        
    return ret

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])

def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)    
    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]



def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret

def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:,dim]).T
    return ret

def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i,:,dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
    return ret

def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                # Fix: Check if we have enough points to sample from
                available_points = x.shape[1] - 2
                needed_points = num_segs[i] - 1
                
                # Ensure we have enough points to sample and adjust if necessary
                if available_points <= 0:
                    # Not enough points for random splitting, fallback to equal segments
                    splits = np.array_split(orig_steps, num_segs[i])
                elif needed_points > available_points:
                    # Too many segments requested, adjust number of segments
                    actual_segs = min(available_points + 1, num_segs[i])
                    splits = np.array_split(orig_steps, actual_segs)
                else:
                    # Original logic can work
                    split_points = np.random.choice(available_points, needed_points, replace=False)
                    split_points.sort()
                    splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])

            # Only permute if we have more than one segment
            if len(splits) > 1:
                perm = np.random.permutation(len(splits))
                warp = np.concatenate([splits[j] for j in perm]).ravel()
                ret[i] = pat[warp]
            else:
                ret[i] = pat
        else:
            ret[i] = pat
    return ret

# Fixed window_warp function
def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
    
    # Handle edge cases: ensure warp_size is at least 1
    warp_size = max(1, warp_size)
    window_steps = np.arange(warp_size)
        
    ret = np.zeros_like(x)
    
    for i, pat in enumerate(x):
        # Check if we have enough room for warping
        if x.shape[1] <= warp_size + 2:
            # Not enough space for warping, return original pattern
            ret[i] = pat
            continue
        
        # Safely generate window start position
        try:
            window_start = np.random.randint(low=1, high=x.shape[1]-warp_size-1)
        except ValueError:
            # Fallback if random range is invalid
            window_start = 1
            
        window_end = window_start + warp_size
            
        for dim in range(x.shape[2]):
            start_seg = pat[:window_start, dim]
            window_seg = np.interp(
                np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), 
                window_steps, 
                pat[window_start:window_end, dim]
            )
            end_seg = pat[window_end:, dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i, :, dim] = np.interp(
                np.arange(x.shape[1]), 
                np.linspace(0, x.shape[1]-1., num=warped.size), 
                warped
            ).T
    
    return ret

def spawner(x, labels, sigma=0.05, verbose=0):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
    # use verbose=-1 to turn off warnings
    # use verbose=1 to print out figures
    
    import dtw as dtw
    
    # Fix for the random_points generation
    if x.shape[1] <= 2:  # Check if there are enough time points
        if verbose > -1:
            print("Warning: Time series too short for spawner augmentation")
        return x  # Return the original data if too short
    
    # Generate random points safely for each time series
    random_points = np.zeros(x.shape[0], dtype=int)
    for i in range(x.shape[0]):
        try:
            random_points[i] = np.random.randint(low=1, high=x.shape[1]-1)
        except ValueError:
            # Fallback if random range is invalid
            random_points[i] = 1
    
    window = np.ceil(x.shape[1] / 10.).astype(int)
    window = max(1, window)  # Ensure window is at least 1
    
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x) if 'tqdm' in globals() else x):
        # guarantees that same one isn't selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:
            random_sample = x[np.random.choice(choices)]
            
            # SPAWNER splits the path into two randomly
            try:
                # Handle potential edge cases with very small sequences
                random_point = random_points[i]
                if random_point <= 0:
                    random_point = 1
                if random_point >= x.shape[1]:
                    random_point = x.shape[1] - 1
                
                # Check if window size is appropriate
                if window >= min(random_point, pat.shape[0] - random_point):
                    # Adjust window if it's too large
                    adjusted_window = max(1, min(random_point, pat.shape[0] - random_point) - 1)
                    if verbose > -1:
                        print(f"Warning: Adjusting window from {window} to {adjusted_window}")
                    window = adjusted_window
                
                path1 = dtw.dtw(pat[:random_point], random_sample[:random_point], 
                               dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
                               
                path2 = dtw.dtw(pat[random_point:], random_sample[random_point:], 
                               dtw.RETURN_PATH, slope_constraint="symmetric", window=window)
                
                combined = np.concatenate((np.vstack(path1), np.vstack(path2+random_point)), axis=1)
                
                if verbose:
                    print(random_point)
                    dtw_value, cost, DTW_map, path = dtw.dtw(pat, random_sample, 
                                                          return_flag=dtw.RETURN_ALL, 
                                                          slope_constraint="symmetric", 
                                                          window=window)
                    dtw.draw_graph1d(cost, DTW_map, path, pat, random_sample)
                    dtw.draw_graph1d(cost, DTW_map, combined, pat, random_sample)
                
                mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
                
                # Handle potential size mismatch
                if mean.shape[0] > 0:
                    for dim in range(x.shape[2]):
                        ret[i,:,dim] = np.interp(orig_steps, 
                                               np.linspace(0, x.shape[1]-1., num=mean.shape[0]), 
                                               mean[:,dim]).T
                else:
                    if verbose > -1:
                        print("Warning: DTW produced empty path, skipping augmentation")
                    ret[i,:] = pat
                    
            except Exception as e:
                if verbose > -1:
                    print(f"Error in DTW computation: {e}")
                ret[i,:] = pat
        else:
            if verbose > -1:
                print(f"There is only one pattern of class {l[i]}, skipping pattern average")
            ret[i,:] = pat
    
    # Assuming jitter is defined elsewhere
    try:
        return jitter(ret, sigma=sigma)
    except:
        if verbose > -1:
            print("Warning: jitter function failed or not found, returning unjittered data")
        return ret

def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, verbose=0):
    # https://ieeexplore.ieee.org/document/8215569
    # use verbose = -1 to turn off warnings    
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    
    import dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
        
    ret = np.zeros_like(x)
    for i in tqdm(range(ret.shape[0])):
        # get the same class as i
        choices = np.where(l == l[i])[0]
        if choices.size > 0:        
            # pick random intra-class pattern
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]
            
            # calculate dtw between all
            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.
                    else:
                        dtw_matrix[p, s] = dtw.dtw(prototype, sample, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                        
            # get medoid
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]
            
            # start weighted DBA
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern 
                    weighted_sums += np.ones_like(weighted_sums) 
                else:
                    path = dtw.dtw(medoid_pattern, random_prototypes[nid], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    weight = np.exp(np.log(0.5)*dtw_value/dtw_matrix[medoid_id, nearest_order[1]])
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight 
            
            ret[i,:] = average_pattern / weighted_sums[:,np.newaxis]
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average"%l[i])
            ret[i,:] = x[i]
    return ret



def random_guided_warp(x, labels, slope_constraint="symmetric", use_window=True, dtw_type="normal", verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or DTW. "normal" or "shape"
    
    import dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:        
            # pick random intra-class pattern
            random_prototype = x[np.random.choice(choices)]
            
            if dtw_type == "shape":
                path = dtw.shape_dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                path = dtw.dtw(random_prototype, pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                            
            # Time warp
            warped = pat[path[1]]
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping timewarping"%l[i])
            ret[i,:] = pat
    return ret

def random_guided_warp_shape(x, labels, slope_constraint="symmetric", use_window=True):
    return random_guided_warp(x, labels, slope_constraint, use_window, dtw_type="shape")

def discriminative_guided_warp(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, dtw_type="normal", use_variable_slice=True, verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or DTW. "normal" or "shape"
    
    import dtw as dtw
    
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    
    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)
        
    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        
        # remove ones of different classes
        positive = np.where(l[choices] == l[i])[0]
        negative = np.where(l[choices] != l[i])[0]
        
        if positive.size > 0 and negative.size > 0:
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[np.random.choice(positive, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(negative, neg_k, replace=False)]
                        
            # vector embedding and nearest prototype in one
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))
            if dtw_type == "shape":
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.shape_dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.shape_dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.shape_dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
            else:
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1./(pos_k-1.))*dtw.dtw(pos_prot, pos_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1./neg_k)*dtw.dtw(pos_prot, neg_samp, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw.dtw(positive_prototypes[selected_id], pat, dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                   
            # Time warp
            warped = pat[path[1]]
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps-warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i,:,dim] = np.interp(orig_steps, np.linspace(0, x.shape[1]-1., num=warped.shape[0]), warped[:,dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d"%l[i])
            ret[i,:] = pat
            warp_amount[i] = 0.
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            # unchanged
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            for i, pat in enumerate(ret):
                # Variable Sllicing
                ret[i] = window_slice(pat[np.newaxis,:,:], reduce_ratio=0.9+0.1*warp_amount[i]/max_warp)[0]
    return ret

def discriminative_guided_warp_shape(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True):
    return discriminative_guided_warp(x, labels, batch_size, slope_constraint, use_window, dtw_type="shape")
