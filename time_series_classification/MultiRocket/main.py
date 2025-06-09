# Modified from MultiRocket (https://github.com/ChangWeiTan/MultiRocket)
# Copyright (C) 2025 Jafar Bakhshaliyev
# Licensed under GNU General Public License v3.0


import argparse
import os
import time
import sys
import socket
import platform
from datetime import datetime

import numba
import numpy as np
import pandas as pd
import psutil
import pytz
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sktime.utils.data_io import load_from_tsfile_to_dataframe

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
    
    if args.tps and args.patch_len > 0:
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

def run_multirocket_experiment(args):
    """
    Run MultiRocket on a dataset with multiple iterations and optional augmentation.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments containing options
    
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame containing results of the experiment
    """
    problem = args.problem
    data_path = args.datapath
    data_folder = data_path + problem + "/"
    
    # Set output directory
    output_path = os.getcwd() + "/output/"
    classifier_name = f"MultiRocket_{args.num_features}"
    
    output_dir = "{}/multirocket/resample_{}/{}/{}/".format(
        output_path,
        args.iter,
        classifier_name,
        problem
    )
    
    if args.save:
        create_directory(output_dir)
    
    train_file = data_folder + problem + "_TRAIN.ts"
    test_file = data_folder + problem + "_TEST.ts"
    
    # Loading data
    X_train, y_train = load_from_tsfile_to_dataframe(train_file)
    X_test, y_test = load_from_tsfile_to_dataframe(test_file)
    
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    
    X_train_processed = process_ts_data(X_train, normalise=False)
    X_test_processed = process_ts_data(X_test, normalise=False)
    
    # Apply augmentation 
    augmentation_tags = "none"
    if args.use_augmentation:
        X_train_processed, y_train, augmentation_tags = run_augmentation(X_train_processed, y_train, args)
    
    accuracies = []
    train_times = []
    test_times = []
    
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
            X_train_processed, y_train,
            predict_on_train=False
        )
        
        yhat_test = classifier.predict(X_test_processed)

        test_acc = accuracy_score(y_test, yhat_test)
        
        if yhat_train is not None:
            train_acc = accuracy_score(y_train, yhat_train)
        else:
            train_acc = -1
        
        accuracies.append(test_acc)
        train_times.append(classifier.train_duration)
        test_times.append(classifier.test_duration)
        
        print(f"Iteration {iteration+1} - Test Accuracy: {test_acc:.4f}, Train Time: {classifier.train_duration:.2f} seconds")
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_train_time = np.mean(train_times)
    mean_test_time = np.mean(test_times)
    
    print(f"\nResults for {problem} with augmentation: {augmentation_tags}")
    print(f"Train size: {X_train_processed.shape[0]} samples")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Mean Train Time: {mean_train_time:.2f} seconds")
    print(f"Mean Test Time: {mean_test_time:.2f} seconds")
    print(f"Individual Accuracies: {accuracies}")
    
    results_df = pd.DataFrame({
        'dataset': [problem],
        'augmentation': [augmentation_tags],
        'train_size': [X_train_processed.shape[0]],
        'test_size': [X_test_processed.shape[0]],
        'mean_accuracy': [mean_accuracy],
        'std_accuracy': [std_accuracy],
        'mean_train_time': [mean_train_time],
        'mean_test_time': [mean_test_time],
        'iterations': [args.iterations],
        'features': [args.num_features],
        'individual_accuracies': [','.join(map(str, accuracies))],
        'patch_len': [args.patch_len] if hasattr(args, 'patch_len') else [0],
        'stride': [args.stride] if hasattr(args, 'stride') else [0],
        'shuffle_rate': [args.shuffle_rate] if hasattr(args, 'shuffle_rate') else [0.0],
    })
    
    if args.save:
        results_filename = f"{output_dir}/multirocket_results_{problem}_{augmentation_tags}.csv"
        if os.path.exists(results_filename):
            existing_df = pd.read_csv(results_filename)
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            combined_df.to_csv(results_filename, index=False)
            print(f"Results appended to {results_filename}")
        else:
            results_df.to_csv(results_filename, index=False)
            print(f"Results saved to new file {results_filename}")
    
    return results_df

def run_all_datasets(args):
    """
    Run MultiRocket on all available datasets in the data path.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments containing options
    """
    # list of available datasets
    data_path = args.datapath
    datasets = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    if not datasets:
        print(f"No datasets found in {data_path}")
        return
    
    print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    
    results = []
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            args.problem = dataset_name
            
            result_df = run_multirocket_experiment(args)
            results.append(result_df)
        
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
    
    if results:
        all_results_df = pd.concat(results, ignore_index=True)
        overall_mean = all_results_df['mean_accuracy'].mean()
        
        print("\n" + "="*80)
        print("SUMMARY OF RESULTS")
        print("="*80)
        print(f"{'Dataset':<25} {'Augmentation':<25} {'Mean Accuracy':<15} {'Std Dev':<10}")
        print("-"*80)
        
        for _, row in all_results_df.iterrows():
            print(f"{row['dataset']:<25} {row['augmentation']:<25} {row['mean_accuracy']:.4f}{' '*8} {row['std_accuracy']:.4f}")
        
        print("-"*80)
        print(f"{'OVERALL':<25} {'':<25} {overall_mean:.4f}")
        print("="*80)
        
        aug_tag = "none" if not args.use_augmentation else "aug"
        output_path = os.getcwd() + "/output/"
        all_results_df.to_csv(f"{output_path}/multirocket_summary_results_{aug_tag}.csv", index=False)
        print(f"\nSummary results saved to {output_path}/multirocket_summary_results_{aug_tag}.csv")

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
    except Exception as e:
        print(f"Error listing datasets: {e}")
        return []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MultiRocket on multivariate time series datasets with optional augmentation')
    
    # Dataset selection
    parser.add_argument("-d", "--datapath", type=str, required=False, default="/home/bakhshaliyev/classification-aug/MultiRocket/data/Multivariate_ts/") # change to your data path
    parser.add_argument("-p", "--problem", type=str, required=False, default="UWaveGestureLibrary")
    parser.add_argument("-i", "--iter", type=int, required=False, default=0)
    parser.add_argument("-n", "--num_features", type=int, required=False, default=50000)
    parser.add_argument("-t", "--num_threads", type=int, required=False, default=-1)
    parser.add_argument("-s", "--save", type=bool, required=False, default=True)
    parser.add_argument("-v", "--verbose", type=int, required=False, default=2)
    
    # Added arguments for augmentation
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations for each experiment (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--all', action='store_true', help='Run on all available datasets')
    
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
        numba.set_num_threads(args.num_threads)
    
    if args.list:
        list_available_datasets(args)
        sys.exit(0)
    
    # Run on all datasets 
    if args.all:
        print(f"Running MultiRocket on all available datasets")
        print(f"Using {args.num_features} features and {args.iterations} iterations")
        if args.use_augmentation:
            print(f"Using data augmentation with ratio {args.augmentation_ratio}")
        run_all_datasets(args)
        sys.exit(0)
    
    # Run on specific dataset
    print(f"Running MultiRocket on {args.problem} dataset")
    print(f"Using {args.num_features} features and {args.iterations} iterations")
    if args.use_augmentation:
        print(f"Using data augmentation with ratio {args.augmentation_ratio}")
    
    run_multirocket_experiment(args)