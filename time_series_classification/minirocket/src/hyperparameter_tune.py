# Modified from MiniRocket (https://github.com/angus924/minirocket)
# Original authors: Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb
# Copyright (C) 2025 Jafar Bakhshaliyev
# Licensed under GNU General Public License v3.0

import numpy as np
import pandas as pd
import os
import time
import sys
import argparse
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import augmentation as aug
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from minirocket import fit, transform

UCR_PATH = "/home/bakhshaliyev/classification-aug/minirocket/UCR"  # Update this path to your environment


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
    print("Augmenting data for dataset %s" % args.dataset)

    np.random.seed(args.seed)
    x_aug = x.copy()
    y_aug = y.copy()
    
    augmentation_tags = ""
    
    if args.augmentation_ratio > 0:
        augmentation_tags = "%d" % args.augmentation_ratio
        print(f"Original training size: {x.shape[0]} samples")
        
        for n in range(args.augmentation_ratio):
            x_temp, current_tags = augment(x, y, args)
            
            x_temp = x_temp.astype(np.float32)
            x_aug = np.vstack((x_aug, x_temp))
            y_aug = np.append(y_aug, y)
            
            print(f"Round {n+1}: {current_tags} done - Added {x_temp.shape[0]} samples")
            
            if n == 0:
                augmentation_tags += current_tags
                
        print(f"Augmented training size: {x_aug.shape[0]} samples")
        print(f"Augmented data type: {x_aug.dtype}")
        
        if args.extra_tag:
            augmentation_tags += "_" + args.extra_tag
    else:
        augmentation_tags = "none"
        if args.extra_tag:
            augmentation_tags = args.extra_tag
    
    x_aug = x_aug.astype(np.float32)

    return x_aug, y_aug, augmentation_tags


    
def augment(x, y, args):
    """
    Apply specified augmentations to the data including SFCC.
    
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
    x : numpy.ndarray
        Augmented time series data
    augmentation_tags : str
        String describing the applied augmentations
    """
    augmentation_tags = ""
    x_aug = x.copy()
    
    needs_reshape = False
    original_shape = x_aug.shape
    
    if len(x_aug.shape) == 2:
        # Reshape from (n_samples, timesteps) to (n_samples, timesteps, 1)
        x_aug = x_aug.reshape(x_aug.shape[0], x_aug.shape[1], 1)
        needs_reshape = True
    
    print('after needs reshape', x_aug.shape)
    
    
    if args.jitter:
        x_aug = aug.jitter(x_aug)
        augmentation_tags += "_jitter"

    if args.tps:
        x_aug = aug.tps(x_aug, y, args.patch_len, args.stride, args.shuffle_rate)
        augmentation_tags += "_tps"
    
    if args.tips:
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
        
    if needs_reshape:
         x_aug = x_aug.reshape(original_shape)

        
    if not augmentation_tags:
        augmentation_tags = "_none"
    
    return x_aug.astype(np.float32), augmentation_tags

def load_ucr_dataset(dataset_name):
    """
    Load a UCR dataset from TSV files.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., 'FordB')
    
    Returns:
    --------
    X_train : numpy.ndarray
        Training data (time series)
    y_train : numpy.ndarray
        Training labels
    X_test : numpy.ndarray
        Test data (time series)
    y_test : numpy.ndarray
        Test labels
    """
    # training data
    train_file = os.path.join(UCR_PATH, dataset_name, f"{dataset_name}_TRAIN.tsv")
    if not os.path.exists(train_file):
        train_file = os.path.join(UCR_PATH, dataset_name, f"{dataset_name}_Train.tsv")
    
    # testing data
    test_file = os.path.join(UCR_PATH, dataset_name, f"{dataset_name}_TEST.tsv")
    if not os.path.exists(test_file):
        test_file = os.path.join(UCR_PATH, dataset_name, f"{dataset_name}_Test.tsv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Dataset files for {dataset_name} not found: {train_file}, {test_file}")
    
    print(f"Loading files: {train_file}, {test_file}")
    
    # Load data 
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    
    y_train = train_df.iloc[:, 0].values
    X_train = train_df.iloc[:, 1:].values
    
    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values
    
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")
    print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    print(f"Data loaded successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train data type: {X_train.dtype}, Test data type: {X_test.dtype}")
    
    return X_train, y_train, X_test, y_test

def run_minirocket_experiment(dataset_name, args):
    """
    Run MiniRocket on a UCR dataset with validation split for hyperparameter tuning.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., 'FordB')
    args : argparse.Namespace
        Command line arguments containing options
    
    Returns:
    --------
    mean_train_accuracy : float
        Mean training accuracy across iterations
    std_train_accuracy : float
        Standard deviation of training accuracy across iterations
    mean_val_accuracy : float
        Mean validation accuracy across iterations
    std_val_accuracy : float
        Standard deviation of validation accuracy across iterations
    """
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    X_train_full, y_train_full, X_test, y_test = load_ucr_dataset(dataset_name)
    
    # Split original training set into train (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, 
        random_state=args.seed,
        stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None
    )
    
    print(f"Split training data: Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
    

    if args.use_augmentation:
        X_train_aug, y_train_aug, augmentation_tags = run_augmentation(X_train, y_train, args)
    else:
        X_train_aug, y_train_aug = X_train.copy(), y_train.copy()
        augmentation_tags = "none"
    
    train_accuracies = []
    val_accuracies = []
    val_cross_entropies = []
    runtimes = []
    
    for iteration in range(args.iterations):
        print(f"Running iteration {iteration+1}/{args.iterations}")
        
        start_time = time.time()
        
        np.random.seed(args.seed + iteration)

        parameters = fit(X_train_aug, num_features=args.features)

        X_train_transform = transform(X_train_aug, parameters)
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier.fit(X_train_transform, y_train_aug)

        train_predictions = classifier.predict(X_train_transform)
        train_accuracy = accuracy_score(y_train_aug, train_predictions)
        train_accuracies.append(train_accuracy)

        X_val_transform = transform(X_val, parameters)
        
        val_predictions = classifier.predict(X_val_transform)
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_accuracies.append(val_accuracy)
        
        # Get decision function values for validation set
        val_decision_scores = classifier.decision_function(X_val_transform)
        
        if len(np.unique(y_val)) > 2:  # Multi-class
            val_probabilities = softmax(val_decision_scores, axis=1)
        else:  # Binary case
            
            val_scores = np.vstack([-val_decision_scores, val_decision_scores]).T
            val_probabilities = softmax(val_scores, axis=1)
        
        try:
            all_classes = np.unique(np.concatenate((y_train_aug, y_val)))
            val_cross_entropy = log_loss(y_val, val_probabilities, labels=all_classes)
            val_cross_entropies.append(val_cross_entropy)
        except Exception as e:
            print(f"Warning: Could not calculate cross-entropy: {e}")
            val_cross_entropy = np.nan
            val_cross_entropies.append(val_cross_entropy)
        
        runtime = time.time() - start_time
        runtimes.append(runtime)
        
        print(f"Iteration {iteration+1} - Train Accuracy: {train_accuracy:.4f}")
        print(f"Iteration {iteration+1} - Validation Accuracy: {val_accuracy:.4f}, Cross-Entropy: {val_cross_entropy:.4f}")
        print(f"Iteration {iteration+1} - Runtime: {runtime:.2f} seconds")

    mean_train_accuracy = np.mean(train_accuracies)
    std_train_accuracy = np.std(train_accuracies)
    mean_val_accuracy = np.mean(val_accuracies)
    std_val_accuracy = np.std(val_accuracies)
    mean_val_cross_entropy = np.mean(val_cross_entropies)
    std_val_cross_entropy = np.std(val_cross_entropies)
    mean_runtime = np.mean(runtimes)

    print(f"\nResults for {dataset_name} with augmentation: {augmentation_tags}")
    print(f"Train size (after augmentation): {X_train_aug.shape[0]} samples")
    print(f"Validation size: {X_val.shape[0]} samples")
    print(f"Mean Train Accuracy: {mean_train_accuracy:.4f} ± {std_train_accuracy:.4f}")
    print(f"Mean Validation Accuracy: {mean_val_accuracy:.4f} ± {std_val_accuracy:.4f}")
    print(f"Mean Validation Cross-Entropy: {mean_val_cross_entropy:.4f} ± {std_val_cross_entropy:.4f}")
    print(f"Mean Runtime: {mean_runtime:.2f} seconds")

    results_df = pd.DataFrame({
        'Dataset': [dataset_name],
        'Augmentation': [augmentation_tags],
        'Train_Size': [X_train.shape[0]],
        'Train_Size_After_Aug': [X_train_aug.shape[0]],
        'Val_Size': [X_val.shape[0]],
        'Mean_Train_Accuracy': [mean_train_accuracy],
        'Train_Accuracy_STD': [std_train_accuracy],
        'Mean_Val_Accuracy': [mean_val_accuracy],
        'Val_Accuracy_STD': [std_val_accuracy],
        'Mean_Val_Cross_Entropy': [mean_val_cross_entropy],
        'Val_Cross_Entropy_STD': [std_val_cross_entropy],
        'Mean_Runtime': [mean_runtime],
        'Iterations': [args.iterations],
        'Features': [args.features],
        'Individual_Train_Accuracies': [','.join(map(str, train_accuracies))],
        'Individual_Val_Accuracies': [','.join(map(str, val_accuracies))],
        'Individual_Val_Cross_Entropies': [','.join(map(str, val_cross_entropies))],
        'patch_len': [args.patch_len],
        'stride': [args.stride],
        'shuffle_rate': [args.shuffle_rate],
        'sfcc_groups': [args.sfcc_groups]
    })
    
    results_filename = f"hyperparameter_tuning_{dataset_name}_{augmentation_tags}.csv"
    
    if os.path.exists(results_filename):
        existing_df = pd.read_csv(results_filename)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        combined_df.to_csv(results_filename, index=False)
        print(f"Results appended to {results_filename}")
    else:
        results_df.to_csv(results_filename, index=False)
        print(f"Results saved to new file {results_filename}")
        
    return mean_train_accuracy, std_train_accuracy, mean_val_accuracy, std_val_accuracy, mean_val_cross_entropy, std_val_cross_entropy, runtimes, augmentation_tags

def list_ucr_datasets():
    """List all available UCR datasets in the UCR_PATH directory"""
    try:
        datasets = [d for d in os.listdir(UCR_PATH) if os.path.isdir(os.path.join(UCR_PATH, d))]
        return sorted(datasets)
    except Exception as e:
        print(f"Error listing datasets: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Time Series Augmentation')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, help='Dataset name (default: FordB)')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    
    # MiniRocket parameters
    parser.add_argument('--features', type=int, default=10000, help='Number of features (default: 10000)')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    # Augmentation control
    parser.add_argument('--use-augmentation', action='store_true', help='Use data augmentation')
    parser.add_argument('--augmentation-ratio', type=int, default=0, 
                        help='Number of augmented copies to add (default: 0)')
    parser.add_argument('--extra-tag', type=str, default='', 
                        help='Extra tag to add to augmentation tags')
    
    # Augmentation methods
    parser.add_argument('--sfcc', action='store_true', help='Apply SFCC augmentation')
    parser.add_argument('--sfcc_groups', type=int, default=4, help='Number of groups for SFCC augmentation (default: 4)')
    parser.add_argument('--jitter', action='store_true', help='Apply jitter augmentation')
    parser.add_argument('--scaling', action='store_true', help='Apply scaling augmentation')
    parser.add_argument('--rotation', action='store_true', help='Apply rotation augmentation')
    parser.add_argument('--permutation', action='store_true', help='Apply permutation augmentation')
    parser.add_argument('--randompermutation', action='store_true', help='Apply random permutation augmentation')
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
    
    parser.add_argument('--stride', type=int, default=2, help='# of patches stride (default: 2)')
    parser.add_argument('--patch_len', type=int, default=10, help='Patch length (default: 10)')
    parser.add_argument('--shuffle_rate', type=float, default=0.3, help='Shuffle rate (default: 0.3)')

    args = parser.parse_args()
    
    if args.list:
        datasets = list_ucr_datasets()
        if datasets:
            print("Available datasets:")
            for dataset in datasets:
                print(f"  - {dataset}")
        else:
            print("No datasets found or UCR_PATH is incorrect.")
        sys.exit(0)
    
    # Run on specific dataset
    dataset_name = args.dataset if args.dataset else "FordB"
    
    print(f"Running hyperparameter tuning on {dataset_name} dataset")
    print(f"Using {args.features} features and {args.iterations} iterations")
    if args.use_augmentation:
        print(f"Using data augmentation with ratio {args.augmentation_ratio}")
    
    run_minirocket_experiment(dataset_name, args)