# Modified from MultiRocket (https://github.com/ChangWeiTan/MultiRocket)
# Copyright (C) 2025 Jafar Bakhshaliyev
# Licensed under GNU General Public License v3.0


import argparse
import os
import time
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from scipy.special import softmax

from multirocket.multirocket_multivariate import MultiRocket
from utils.data_loader import process_ts_data
from utils.tools import create_directory

import augmentation as aug 

pd.set_option('display.max_columns', 500)

def run_augmentation(x, y, args):
    """
    Apply data augmentation to the input data based on args.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Original time series data
    y : numpy.ndarray
        Original labels
    args : argparse.Namespace
        Command line arguments containing augmentation options
        
    Returns:
    --------
    x_aug : numpy.ndarray
        Augmented time series data
    y_aug : numpy.ndarray
        Augmented labels
    augmentation_tags : str
        String describing the applied augmentations
    """
    print("Augmenting data for dataset %s" % args.problem)
    np.random.seed(args.seed)
    x_aug = x.copy()
    y_aug = y.copy()
    
    augmentation_tags = ""
    
    if args.augmentation_ratio > 0:
        augmentation_tags = "%d" % args.augmentation_ratio
        print(f"Original training size: {x.shape[0]} samples")
        
        for n in range(args.augmentation_ratio):
            x_temp, current_tags = augment(x, y, args)
            
            if x_temp.shape != x.shape:
                print(f"Warning: Augmented data shape {x_temp.shape} doesn't match original shape {x.shape}")
                continue
                
            x_aug = np.concatenate((x_aug, x_temp), axis=0)
            y_aug = np.append(y_aug, y)
            
            print(f"Round {n+1}: {current_tags} done - Added {x_temp.shape[0]} samples")
            
            if n == 0:
                augmentation_tags += current_tags
                
        print(f"Augmented training size: {x_aug.shape[0]} samples")
        
        if args.extra_tag:
            augmentation_tags += "_" + args.extra_tag
    else:
        augmentation_tags = "none"
        if args.extra_tag:
            augmentation_tags = args.extra_tag
            
    return x_aug, y_aug, augmentation_tags

    
def augment(x, y, args):
    """
    Apply specified augmentations to the multivariate time series data.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Original time series data with shape (n_samples, n_dimensions, n_timesteps)
    y : numpy.ndarray
        Original labels
    args : argparse.Namespace
        Command line arguments containing augmentation options
        
    Returns:
    --------
    x : numpy.ndarray
        Augmented time series data
    augmentation_tags : str
        String describing the applied augmentations
    """
    augmentation_tags = ""
    
    x_aug = x.copy()
    

    if len(x_aug.shape) != 3:
        if len(x_aug.shape) == 2:
            x_aug = x_aug.reshape(x_aug.shape[0], 1, x_aug.shape[1])
            print(f"Reshaped to {x_aug.shape} for processing")
    
    if args.jitter:
        x_aug = aug.jitter(x_aug)
        augmentation_tags += "_jitter"

    if args.tps and args.patch_len > 0:
        x_aug = aug.tps(x_aug, y, args.patch_len, args.stride, args.shuffle_rate)
        augmentation_tags += "_tps"

    if args.tips and args.patch_len > 0:
        x_aug = aug.tips(x_aug, y, args.patch_len, args.stride, args.shuffle_rate)
        augmentation_tags += "_tips"
        
    if args.scaling:
        x_aug = aug.scaling(x_aug)
        augmentation_tags += "_scaling"
        
    if args.rotation:
        x_aug = aug.rotation(x_aug)
        augmentation_tags += "_rotation"
        
    if args.permutation:
        x_aug = aug.permutation(x_aug)
        augmentation_tags += "_permutation"
        
    if args.randompermutation:
        x_aug = aug.permutation(x_aug, seg_mode="random")
        augmentation_tags += "_randomperm"
        
    if args.magwarp:
        x_aug = aug.magnitude_warp(x_aug)
        augmentation_tags += "_magwarp"
        
    if args.timewarp:
        x_aug = aug.time_warp(x_aug)
        augmentation_tags += "_timewarp"
        
    if args.windowslice:
        x_aug = aug.window_slice(x_aug)
        augmentation_tags += "_windowslice"
        
    if args.windowwarp:
        x_aug = aug.window_warp(x_aug)
        augmentation_tags += "_windowwarp"
        
    if args.spawner:
        x_aug = aug.spawner(x_aug, y)
        augmentation_tags += "_spawner"
        
    if args.dtwwarp:
        x_aug = aug.random_guided_warp(x_aug, y)
        augmentation_tags += "_rgw"
        
    if args.shapedtwwarp:
        x_aug = aug.random_guided_warp_shape(x_aug, y)
        augmentation_tags += "_rgws"
        
    if args.wdba:
        x_aug = aug.wdba(x_aug, y)
        augmentation_tags += "_wdba"
        
    if args.discdtw:
        x_aug = aug.discriminative_guided_warp(x_aug, y)
        augmentation_tags += "_dgw"
        
    if args.discsdtw:
        x_aug = aug.discriminative_guided_warp_shape(x_aug, y)
        augmentation_tags += "_dgws"
        
    if not augmentation_tags:
        augmentation_tags = "_none"
    
    return x_aug, augmentation_tags

def run_multirocket_hyperparameter_tuning(args):
    """
    Run MultiRocket hyperparameter tuning on a dataset with train/validation split.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments containing options
    
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame containing results of the hyperparameter tuning
    """
    problem = args.problem
    data_path = args.datapath
    data_folder = data_path + problem + "/"
    
    # Set output directory
    output_path = os.getcwd() + "/output/"
    classifier_name = f"MultiRocket_{args.num_features}"
    
    output_dir = "{}/multirocket/hyperparameter_tuning/{}/{}/".format(
        output_path,
        classifier_name,
        problem
    )
    
    if args.save:
        create_directory(output_dir)
    
    train_file = data_folder + problem + "_TRAIN.ts"
    test_file = data_folder + problem + "_TEST.ts"
    
    print("Loading data")
    X_train_full, y_train_full = load_from_tsfile_to_dataframe(train_file)
    
    encoder = LabelEncoder()
    y_train_full = encoder.fit_transform(y_train_full)

    X_train_full_processed = process_ts_data(X_train_full, normalise=False)
    
    # Split the training set into training and validation sets (80/20)
    try:
        if len(np.unique(y_train_full)) > 1:
            class_counts = np.bincount(y_train_full.astype(int))
            if np.min(class_counts[class_counts > 0]) >= 2:
                train_indices, val_indices = train_test_split(
                    np.arange(len(y_train_full)), 
                    test_size=0.2, 
                    random_state=args.seed,
                    stratify=y_train_full
                )
            else:
                print("Warning: Some classes have only 1 sample. Using regular split instead of stratified split.")
                train_indices, val_indices = train_test_split(
                    np.arange(len(y_train_full)), 
                    test_size=0.2, 
                    random_state=args.seed,
                    stratify=None
                )
        else:
            train_indices, val_indices = train_test_split(
                np.arange(len(y_train_full)), 
                test_size=0.2, 
                random_state=args.seed,
                stratify=None
            )
    except Exception as e:
        print(f"Warning: Failed to perform stratified split: {e}")
        print("Falling back to regular random split.")
        train_indices, val_indices = train_test_split(
            np.arange(len(y_train_full)), 
            test_size=0.2, 
            random_state=args.seed,
            stratify=None
        )
    
    y_train = y_train_full[train_indices].copy()
    y_val = y_train_full[val_indices].copy()
    
    X_train = X_train_full_processed[train_indices]
    X_val = X_train_full_processed[val_indices]
    
    print(f"Split training data: Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
    
    # Apply augmentation 
    augmentation_tags = "none"
    if args.use_augmentation:
        X_train_aug, y_train_aug, augmentation_tags = run_augmentation(X_train, y_train, args)
    else:
        X_train_aug, y_train_aug = X_train.copy(), y_train.copy()
    
    train_accuracies = []
    val_accuracies = []
    val_cross_entropies = []
    train_times = []
    
    for iteration in range(args.iterations):
        print(f"Running iteration {iteration+1}/{args.iterations}")
        
        start_time = time.perf_counter()
        
        np.random.seed(args.seed + iteration)
        
        classifier = MultiRocket(
            num_features=args.num_features,
            classifier="logistic",
            verbose=args.verbose
        )
        
        yhat_train = classifier.fit(
            X_train_aug, y_train_aug,
            predict_on_train=True 
        )
        
        yhat_val = classifier.predict(X_val)

        train_acc = accuracy_score(y_train_aug, yhat_train)
        train_accuracies.append(train_acc)
        
        val_acc = accuracy_score(y_val, yhat_val)
        val_accuracies.append(val_acc)
        
        try:
            val_proba = classifier.predict_proba(X_val)
            
            try:
                all_classes = np.unique(np.concatenate((y_train_aug, y_val)))
                val_cross_entropy = log_loss(y_val, val_proba, labels=all_classes)
                val_cross_entropies.append(val_cross_entropy)
            except Exception as e:
                print(f"Warning: Could not calculate cross-entropy: {e}")
                val_cross_entropy = np.nan
                val_cross_entropies.append(val_cross_entropy)
                
        except (AttributeError, NotImplementedError) as e:
            print(f"Warning: Could not get probability estimates: {e}")
            val_cross_entropy = np.nan
            val_cross_entropies.append(val_cross_entropy)
        
        train_time = classifier.train_duration
        train_times.append(train_time)
        
        print(f"Iteration {iteration+1} - Train Accuracy: {train_acc:.4f}")
        print(f"Iteration {iteration+1} - Validation Accuracy: {val_acc:.4f}")
        if not np.isnan(val_cross_entropy):
            print(f"Iteration {iteration+1} - Validation Cross-Entropy: {val_cross_entropy:.4f}")
        print(f"Iteration {iteration+1} - Train Time: {train_time:.2f} seconds")
    
    # Calculate mean and standard deviation
    mean_train_accuracy = np.mean(train_accuracies)
    std_train_accuracy = np.std(train_accuracies)
    mean_val_accuracy = np.mean(val_accuracies)
    std_val_accuracy = np.std(val_accuracies)
    mean_val_cross_entropy = np.nanmean(val_cross_entropies) if not all(np.isnan(val_cross_entropies)) else np.nan
    std_val_cross_entropy = np.nanstd(val_cross_entropies) if not all(np.isnan(val_cross_entropies)) else np.nan
    mean_train_time = np.mean(train_times)
    
    print(f"\nHyperparameter Tuning Results for {problem} with augmentation: {augmentation_tags}")
    print(f"Original train size: {X_train.shape[0]} samples")
    print(f"Augmented train size: {X_train_aug.shape[0]} samples")
    print(f"Validation size: {X_val.shape[0]} samples")
    print(f"Mean Train Accuracy: {mean_train_accuracy:.4f} ± {std_train_accuracy:.4f}")
    print(f"Mean Validation Accuracy: {mean_val_accuracy:.4f} ± {std_val_accuracy:.4f}")
    if not np.isnan(mean_val_cross_entropy):
        print(f"Mean Validation Cross-Entropy: {mean_val_cross_entropy:.4f} ± {std_val_cross_entropy:.4f}")
    print(f"Mean Train Time: {mean_train_time:.2f} seconds")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'dataset': [problem],
        'augmentation': [augmentation_tags],
        'train_size': [X_train.shape[0]],
        'train_size_after_aug': [X_train_aug.shape[0]],
        'val_size': [X_val.shape[0]],
        'mean_train_accuracy': [mean_train_accuracy],
        'train_accuracy_std': [std_train_accuracy],
        'mean_val_accuracy': [mean_val_accuracy],
        'val_accuracy_std': [std_val_accuracy],
        'mean_val_cross_entropy': [mean_val_cross_entropy],
        'val_cross_entropy_std': [std_val_cross_entropy],
        'mean_train_time': [mean_train_time],
        'iterations': [args.iterations],
        'features': [args.num_features],
        'individual_train_accuracies': [','.join(map(str, train_accuracies))],
        'individual_val_accuracies': [','.join(map(str, val_accuracies))],
        'individual_val_cross_entropies': [','.join(map(str, val_cross_entropies))],
        'patch_len': [args.patch_len],
        'stride': [args.stride],
        'shuffle_rate': [args.shuffle_rate]
    })
    
    if args.save:
        results_filename = f"{output_dir}/multirocket_hyperparameter_tuning_{problem}_{augmentation_tags}.csv"
        if os.path.exists(results_filename):
            try:
                existing_df = pd.read_csv(results_filename)
                combined_df = pd.concat([existing_df, results_df], ignore_index=True)
                combined_df.to_csv(results_filename, index=False)
                print(f"Results appended to {results_filename}")
            except Exception as e:
                print(f"Error appending to existing file: {e}")
                results_df.to_csv(results_filename, index=False)
                print(f"Created new file instead: {results_filename}")
        else:
            results_df.to_csv(results_filename, index=False)
            print(f"Results saved to new file {results_filename}")
    
    return results_df

def list_available_datasets(args):
    """
    List all available datasets in the data path.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments containing options
    """
    data_path = args.datapath
    try:
        datasets = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        print("Available datasets:")
        for dataset in sorted(datasets):
            print(f"  - {dataset}")
        return sorted(datasets)
    except Exception as e:
        print(f"Error listing datasets: {e}")
        return []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for MultiRocket on Multivariate Time Series')
    
    # Dataset selection
    parser.add_argument("-d", "--datapath", type=str, required=False, default="/home/bakhshaliyev/classification-aug/MultiRocket/data/Multivariate_ts/")
    parser.add_argument("-p", "--problem", type=str, required=False, default="UWaveGestureLibrary")
    parser.add_argument("-n", "--num_features", type=int, required=False, default=50000)
    parser.add_argument("-t", "--num_threads", type=int, required=False, default=-1)
    parser.add_argument("-s", "--save", type=bool, required=False, default=True)
    parser.add_argument("-v", "--verbose", type=int, required=False, default=2)
    
    # Added arguments for tuning
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations for each experiment (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    
    # Augmentation control
    parser.add_argument('--use-augmentation', action='store_true', help='Use data augmentation')
    parser.add_argument('--augmentation-ratio', type=int, default=0, 
                      help='Number of augmented copies to add (default: 0)')
    parser.add_argument('--extra-tag', type=str, default='', 
                      help='Extra tag to add to augmentation tags')
    
    # Augmentation methods
    parser.add_argument('--jitter', action='store_true', help='Apply jitter augmentation')
    parser.add_argument('--scaling', action='store_true', help='Apply scaling augmentation')
    parser.add_argument('--rotation', action='store_true', help='Apply rotation augmentation')
    parser.add_argument('--permutation', action='store_true', help='Apply permutation augmentation')
    parser.add_argument('--randompermutation', action='store_true', help='Apply random permutation augmentation')
    parser.add_argument('--magwarp', action='store_true', help='Apply magnitude warp augmentation')
    parser.add_argument('--timewarp', action='store_true', help='Apply time warp augmentation')
    parser.add_argument('--windowslice', action='store_true', help='Apply window slice augmentation')
    parser.add_argument('--windowwarp', action='store_true', help='Apply window warp augmentation')
    parser.add_argument('--spawner', action='store_true', help='Apply spawner augmentation')
    parser.add_argument('--dtwwarp', action='store_true', help='Apply DTW-based warp augmentation')
    parser.add_argument('--shapedtwwarp', action='store_true', help='Apply shape DTW warp augmentation')
    parser.add_argument('--wdba', action='store_true', help='Apply WDBA augmentation')
    parser.add_argument('--discdtw', action='store_true', help='Apply discriminative DTW augmentation')
    parser.add_argument('--discsdtw', action='store_true', help='Apply discriminative shape DTW augmentation')
    parser.add_argument('--tps', action='store_true', help='Apply TPS augmentation')
    parser.add_argument('--tips', action='store_true', help='Apply TIPS augmentation')

    # TPS specific parameters
    parser.add_argument('--stride', type=int, default=0, help='# of patches stride')
    parser.add_argument('--patch_len', type=int, default=0, help='# of patches')
    parser.add_argument('--shuffle_rate', type=float, default=0.0, help='shuffle rate')
    
    args = parser.parse_args()
    
    if args.num_threads > 0:
        import numba
        numba.set_num_threads(args.num_threads)
    
    if args.list:
        list_available_datasets(args)
        sys.exit(0)
    
    # Run hyperparameter tuning on specified dataset
    print(f"Running MultiRocket hyperparameter tuning on {args.problem} dataset")
    print(f"Using {args.num_features} features and {args.iterations} iterations")
    if args.use_augmentation:
        print(f"Using data augmentation with ratio {args.augmentation_ratio}")
    
    run_multirocket_hyperparameter_tuning(args)