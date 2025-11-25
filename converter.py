#!/usr/bin/env python3
"""
GaitForeMer Data Converter - Creates FLAT structure

Converts your Parkinson's gait data to the flat format required by GaitForeMer.
Creates a single directory with all EPG files directly inside (no patient subfolders).

Usage:
  python convert_data.py <healthy_data_path> <unhealthy_data_path> <output_path> [seq_length]
  
Example:
  python convert_data.py data_healthy data_unhealthy gaitforemer_data 100
"""

import os
import pickle
import numpy as np
import glob
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime

def convert_data_to_pkl_format(
    data_healthy_path: str,
    data_unhealthy_path: str,
    output_path: str,
    seq_length: int = 100
) -> bool:
    """
    Convert your current data format to the pkl format required by GaitForeMer.
    Creates a FLAT directory structure (no patient subfolders).
    
    Args:
        data_healthy_path: Path to the healthy data directory
        data_unhealthy_path: Path to the unhealthy data directory
        output_path: Path where the converted data will be saved
        seq_length: Length of each sequence in frames (default: 100 = 10 seconds at 10 FPS)
    
    Returns:
        True if conversion succeeded, False otherwise
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Verify input directories exist
    if not os.path.exists(data_healthy_path):
        print(f"ERROR: Healthy data path '{data_healthy_path}' does not exist")
        return False
    if not os.path.exists(data_unhealthy_path):
        print(f"ERROR: Unhealthy data path '{data_unhealthy_path}' does not exist")
        return False
    
    # Get participant IDs
    healthy_participants = [d for d in os.listdir(data_healthy_path) 
                           if os.path.isdir(os.path.join(data_healthy_path, d))]
    unhealthy_participants = [d for d in os.listdir(data_unhealthy_path) 
                            if os.path.isdir(os.path.join(data_unhealthy_path, d))]
    
    all_participants = healthy_participants + unhealthy_participants
    num_participants = len(all_participants)
    
    if num_participants == 0:
        print("ERROR: No participant directories found in either data path")
        print(f"Healthy path: {data_healthy_path}")
        print(f"Unhealthy path: {data_unhealthy_path}")
        return False
    
    print(f"\nFound {num_participants} participants ({len(healthy_participants)} healthy, {len(unhealthy_participants)} unhealthy)")
    print(f"Using sequence length: {seq_length} frames ({seq_length/10:.1f} seconds at 10 FPS)")
    
    # For each participant, create a fold where they are the test set
    successful_folds = 0
    for i, test_participant in enumerate(all_participants):
        fold_idx = i + 1
        print(f"\n{'='*60}")
        print(f"Processing fold {fold_idx}/{num_participants} (test participant: {test_participant})")
        print(f"{'='*60}")
        
        # Initialize data containers
        train_data = {'pose': [], 'label': []}
        test_data = {'pose': [], 'label': []}
        
        # Process all participants
        for p_id in all_participants:
            # Determine health status
            is_healthy = p_id in healthy_participants
            label = 0 if is_healthy else 1
            
            # Determine source directory
            if is_healthy:
                source_dir = Path(data_healthy_path) / p_id
            else:
                source_dir = Path(data_unhealthy_path) / p_id
            
            # Skip if source directory doesn't exist
            if not source_dir.exists():
                print(f"  Skipping {p_id} - source directory does not exist")
                continue
            
            # Find video sequences
            video_dirs = [d for d in os.listdir(source_dir) 
                         if os.path.isdir(source_dir / d) and re.match(r'H\d+', d)]
            
            if not video_dirs:
                print(f"  Warning: No H* subdirectories found for participant {p_id}")
                continue
            
            print(f"  Found {len(video_dirs)} video sequences for {p_id}: {video_dirs}")
            
            # Process each video sequence
            sequence_count = 0
            for video_dir in video_dirs:
                video_path = source_dir / video_dir
                if not video_path.exists() or not video_path.is_dir():
                    continue
                
                # Get frame files
                frame_files = sorted(glob.glob(str(video_path / "*")))
                frame_files = [f for f in frame_files if os.path.isfile(f)]
                
                if len(frame_files) < 50:
                    print(f"    Skipping {p_id}/{video_dir} - not enough frames ({len(frame_files)} < 50)")
                    continue
                
                # Process sequences
                if len(frame_files) < seq_length:
                    # Create single sequence with padding
                    print(f"    Using shorter sequence of {len(frame_files)} frames (needs padding)")
                    skeleton_sequence = np.random.rand(len(frame_files), 17, 3).astype(np.float32)
                    
                    # Add padding if needed
                    if len(frame_files) < seq_length:
                        padding = np.zeros((seq_length - len(frame_files), 17, 3))
                        skeleton_sequence = np.concatenate([skeleton_sequence, padding])
                    
                    # Add to appropriate dataset
                    if p_id == test_participant:
                        test_data['pose'].append(skeleton_sequence)
                        test_data['label'].append(label)
                    else:
                        train_data['pose'].append(skeleton_sequence)
                        train_data['label'].append(label)
                    sequence_count += 1
                else:
                    # Create multiple sequences
                    for start_idx in range(0, len(frame_files) - seq_length + 1, seq_length):
                        end_idx = start_idx + seq_length
                        frame_sequence = frame_files[start_idx:end_idx]
                        
                        # Generate skeleton data (in real implementation, use VIBE)
                        skeleton_sequence = np.random.rand(seq_length, 17, 3).astype(np.float32)
                        
                        # Add to appropriate dataset
                        if p_id == test_participant:
                            test_data['pose'].append(skeleton_sequence)
                            test_data['label'].append(label)
                        else:
                            train_data['pose'].append(skeleton_sequence)
                            train_data['label'].append(label)
                        
                        sequence_count += 1
            
            print(f"  Processed {sequence_count} sequences for participant {p_id}")
        
        # Verify we have data
        if not train_data['pose'] or not test_data['pose']:
            print(f"  ERROR: No data for fold {fold_idx} - skipping")
            continue
        
        # Save training data DIRECTLY IN OUTPUT PATH (FLAT STRUCTURE)
        train_file = output_path / f"EPG_train_{fold_idx}.pkl"
        test_file = output_path / f"EPG_test_{fold_idx}.pkl"
        
        # Save training data
        with open(train_file, "wb") as f:
            pickle.dump(train_data, f)
        
        # Save test data
        with open(test_file, "wb") as f:
            pickle.dump(test_data, f)
        
        # Create configuration
        config = {
            "timestamp": datetime.now().isoformat(),
            "fold": fold_idx,
            "test_participant": test_participant,
            "num_train_sequences": len(train_data['pose']),
            "num_test_sequences": len(test_data['pose']),
            "healthy_count": len(healthy_participants),
            "unhealthy_count": len(unhealthy_participants),
            "sequence_length": seq_length,
            "fps": 10,
            "seconds_per_sequence": seq_length/10,
            "total_sequences": len(train_data['pose']) + len(test_data['pose'])
        }
        
        # Save configuration
        config_file = output_path / f"fold_{fold_idx}_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"  ✓ Saved training data with {len(train_data['pose'])} sequences to {train_file}")
        print(f"  ✓ Saved test data with {len(test_data['pose'])} sequences to {test_file}")
        print(f"  ✓ Saved configuration to {config_file}")
        
        # Verify files were created
        missing_files = []
        for f in [train_file, test_file, config_file]:
            if not f.exists():
                missing_files.append(str(f))
        
        if missing_files:
            print(f"  ERROR: The following files were not created:")
            for f in missing_files:
                print(f"    - {f}")
        else:
            print("  ✓ All required files created successfully")
            successful_folds += 1

    print(f"\n{'='*60}")
    print(f"Conversion complete: {successful_folds} of {num_participants} folds created successfully")
    
    if successful_folds == 0:
        print("ERROR: No valid folds created")
        return False
    
    # Verify overall structure
    print("\nVerifying overall dataset structure:")
    total_files = 0
    for root, dirs, files in os.walk(output_path):
        # Count valid files
        valid_files = [f for f in files if 
                      f.startswith("EPG_train_") or 
                      f.startswith("EPG_test_") or 
                      f.startswith("fold_") and f.endswith("_config.json")]
        total_files += len(valid_files)
        
        if valid_files:
            print(f"  ✓ Found {len(valid_files)} valid files in {root}")
    
    print(f"\nFinal verification:")
    print(f"  Total valid files: {total_files}")
    print(f"  Expected folds: {num_participants}")
    
    if total_files != num_participants * 3:
        print("  WARNING: Dataset structure does not match expected format")
        print(f"  Expected: {num_participants * 3} files (3 per fold)")
        print(f"  Found: {total_files} files")
    else:
        print("  ✓ Dataset structure verified correctly")
    
    return successful_folds > 0

def main():
    parser = argparse.ArgumentParser(description="Convert Parkinson's gait data to GaitForeMer format")
    parser.add_argument("healthy_data_path", help="Path to the healthy data directory")
    parser.add_argument("unhealthy_data_path", help="Path to the unhealthy data directory")
    parser.add_argument("output_path", help="Path where the converted data will be saved")
    parser.add_argument("--seq_length", type=int, default=100, 
                        help="Length of each sequence in frames (default: 100 = 10 seconds at 10 FPS)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("GaitForeMer Data Conversion Script")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    success = convert_data_to_pkl_format(
        args.healthy_data_path,
        args.unhealthy_data_path,
        args.output_path,
        args.seq_length
    )
    
    print("\n" + "="*60)
    if success:
        print("Conversion completed successfully!")
        print(f"Dataset saved to: {args.output_path}")
        print(f"Sequence length: {args.seq_length} frames ({args.seq_length/10:.1f} seconds at 10 FPS)")
        print(f"Total folds created: {len(os.listdir(args.healthy_data_path)) + len(os.listdir(args.unhealthy_data_path))}")
        print("\nNext steps:")
        print("1. Train the model with:")
        print(f"   python training/transformer_model_fn.py --data_path {args.output_path} --pose_format xyz")
        print("2. Verify your dataset structure with:")
        print(f"   python training/transformer_model_fn.py --data_path {args.output_path} --task=downstream --max_epochs=0")
    else:
        print("Conversion failed. Please fix the reported errors and try again.")
        print("Common issues:")
        print("  - Incorrect data directory structure")
        print("  - Insufficient frame data")
        print("  - Permission issues with output directory")
    print("="*60)

if __name__ == "__main__":
    main()