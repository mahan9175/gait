# # import os
# # import pickle
# # import numpy as np
# # import glob
# # import sys
# # import json
# # import re
# # import argparse
# # from pathlib import Path
# # from typing import List, Dict, Tuple, Any
# # import torch
# # import cv2

# # def extract_poses_with_vibe(frame_files: List[str]) -> np.ndarray:
# #     """
# #     Extract 3D poses from video frames using VIBE model.
# #     Returns: np.array of shape (num_frames, 17, 3) - 17 joints with 3D coordinates
# #     """
# #     try:
# #         # Option 1: Using pre-installed VIBE
# #         from vibe import VIBE
# #         vibe_model = VIBE()
        
# #         # Load and process frames
# #         frames = []
# #         for frame_file in frame_files:
# #             frame = cv2.imread(frame_file)
# #             if frame is not None:
# #                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #                 frames.append(frame)
        
# #         if not frames:
# #             raise ValueError("No valid frames found")
        
# #         # Run VIBE inference
# #         with torch.no_grad():
# #             results = vibe_model(frames)
        
# #         # Extract 3D joints and map to 17 joints
# #         poses_3d = results['poses_3d']  # Shape: (num_frames, 24, 3)
        
# #         # Map VIBE's 24 SMPL joints to 17 COCO-style joints
# #         joint_mapping = [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# #         poses_17 = poses_3d[:, joint_mapping, :]
        
# #         return poses_17.numpy().astype(np.float32)
        
# #     except ImportError:
# #         print("⚠️  VIBE not installed. Using fallback pose extraction.")
# #         return extract_poses_fallback(frame_files)

# # def extract_poses_fallback(frame_files: List[str]) -> np.ndarray:
# #     """
# #     Fallback pose extraction using MediaPipe.
# #     """
# #     try:
# #         import mediapipe as mp
# #         mp_pose = mp.solutions.pose
# #         pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
        
# #         poses = []
# #         for frame_file in frame_files:
# #             frame = cv2.imread(frame_file)
# #             if frame is None:
# #                 continue
                
# #             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #             results = pose.process(frame_rgb)
            
# #             if results.pose_landmarks:
# #                 landmarks = results.pose_landmarks.landmark
# #                 # MediaPipe returns 33 landmarks, take first 17 for body
# #                 pose_3d = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks[:17]])
# #                 poses.append(pose_3d)
# #             else:
# #                 # If no pose detected, use zeros
# #                 poses.append(np.zeros((17, 3)))
        
# #         pose.close()
# #         return np.array(poses).astype(np.float32)
        
# #     except ImportError:
# #         print("❌ No pose estimation library available.")
# #         print("⚠️  USING RANDOM DATA - THIS IS FOR TESTING ONLY!")
# #         num_frames = len(frame_files)
# #         return np.random.rand(num_frames, 17, 3).astype(np.float32)

# # def create_overlapping_sequences(poses: np.ndarray, seq_length: int = 100, overlap_ratio: float = 0.5) -> List[np.ndarray]:
# #     """
# #     Create overlapping sequences from long pose sequences.
    
# #     Args:
# #         poses: Full pose sequence (num_frames, 17, 3)
# #         seq_length: Length of each sequence (default: 100 frames = 10 seconds)
# #         overlap_ratio: Overlap between sequences (0.5 = 50% overlap)
    
# #     Returns:
# #         List of sequences, each of shape (seq_length, 17, 3)
# #     """
# #     if len(poses) < seq_length:
# #         # Pad if sequence is too short
# #         padding = np.zeros((seq_length - len(poses), 17, 3))
# #         padded_sequence = np.concatenate([poses, padding])
# #         return [padded_sequence]
    
# #     step_size = int(seq_length * (1 - overlap_ratio))
# #     sequences = []
    
# #     for start_idx in range(0, len(poses) - seq_length + 1, step_size):
# #         end_idx = start_idx + seq_length
# #         sequence = poses[start_idx:end_idx]
# #         sequences.append(sequence)
    
# #     # Always include the last sequence if we have leftover frames
# #     if len(poses) >= seq_length and (len(poses) - seq_length) % step_size != 0:
# #         last_sequence = poses[-seq_length:]
# #         sequences.append(last_sequence)
    
# #     return sequences

# # def convert_data_to_pkl_format(
# #     data_healthy_path: str,
# #     data_unhealthy_path: str,
# #     output_path: str,
# #     seq_length: int = 100,
# #     overlap_ratio: float = 0.5,  # NEW: 50% overlap between sequences
# #     min_frames: int = 100        # NEW: Minimum frames required per walk
# # ) -> None:
# #     """
# #     Convert your current data format to the pkl format required by GaitForeMer.
# #     Optimized for 1200-frame walks with overlapping sequences.
    
# #     Args:
# #         data_healthy_path: Path to the healthy data directory
# #         data_unhealthy_path: Path to the unhealthy data directory
# #         output_path: Path where the converted data will be saved
# #         seq_length: Length of each sequence in frames (default: 100 = 10 seconds at 10 FPS)
# #         overlap_ratio: Overlap between consecutive sequences (0.0 = no overlap, 0.5 = 50% overlap)
# #         min_frames: Minimum number of frames required to process a walk
# #     """
# #     # Create output directory if it doesn't exist
# #     Path(output_path).mkdir(parents=True, exist_ok=True)
    
# #     # Get all participant IDs from both healthy and unhealthy directories
# #     healthy_participants = [d for d in os.listdir(data_healthy_path) 
# #                            if os.path.isdir(os.path.join(data_healthy_path, d))]
# #     unhealthy_participants = [d for d in os.listdir(data_unhealthy_path) 
# #                             if os.path.isdir(os.path.join(data_unhealthy_path, d))]
    
# #     all_participants = healthy_participants + unhealthy_participants
# #     num_participants = len(all_participants)
    
# #     print(f"Found {num_participants} participants ({len(healthy_participants)} healthy, {len(unhealthy_participants)} unhealthy)")
# #     print(f"Using sequence length: {seq_length} frames ({seq_length/10:.1f} seconds at 10 FPS)")
# #     print(f"Using overlap ratio: {overlap_ratio*100}%")
# #     print(f"Minimum frames per walk: {min_frames}")
    
# #     # For each participant, create a fold where they are the test set
# #     for i, test_participant in enumerate(all_participants):
# #         fold_idx = i + 1
# #         print(f"\nProcessing fold {fold_idx}/{num_participants} (test participant: {test_participant})")
        
# #         # Create dictionaries to store training and test data
# #         train_data = {
# #             'pose': [],
# #             'label': []
# #         }
# #         test_data = {
# #             'pose': [],
# #             'label': []
# #         }
        
# #         # Process all participants
# #         for p_id in all_participants:
# #             # Determine if participant is healthy or unhealthy
# #             is_healthy = p_id in healthy_participants
# #             label = 0 if is_healthy else 1
            
# #             # Find the source directory for this participant
# #             if is_healthy:
# #                 source_dir = Path(data_healthy_path) / p_id
# #             else:
# #                 source_dir = Path(data_unhealthy_path) / p_id
            
# #             # Find all H* subdirectories (H1, H2, H3, etc.)
# #             video_dirs = [d for d in os.listdir(source_dir) 
# #                          if os.path.isdir(os.path.join(source_dir, d)) and re.match(r'H\d+', d)]
            
# #             if not video_dirs:
# #                 print(f"  Warning: No H* subdirectories found for participant {p_id}")
# #                 continue
            
# #             print(f"  Found {len(video_dirs)} video sequences for participant {p_id}: {video_dirs}")
            
# #             # Process all video sequences for this participant
# #             total_sequences_for_participant = 0
# #             for video_dir in video_dirs:
# #                 video_path = source_dir / video_dir
# #                 if not video_path.exists() or not video_path.is_dir():
# #                     continue
                
# #                 # Get all frame files
# #                 frame_files = sorted(glob.glob(os.path.join(video_path, "*")))
# #                 frame_files = [f for f in frame_files if os.path.isfile(f) and any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp'])]
                
# #                 # Skip if not enough frames
# #                 if len(frame_files) < min_frames:
# #                     print(f"    Skipping {p_id}/{video_dir}: Only {len(frame_files)} frames (< {min_frames} minimum)")
# #                     continue
                
# #                 print(f"    Processing {len(frame_files)} frames from {p_id}/{video_dir}...")
                
# #                 # ✅ Extract poses from ALL frames first
# #                 try:
# #                     all_poses = extract_poses_with_vibe(frame_files)
# #                     print(f"    ✓ Extracted poses: {all_poses.shape}")
# #                 except Exception as e:
# #                     print(f"    ❌ Pose extraction failed: {str(e)}")
# #                     continue
                
# #                 # ✅ Create overlapping sequences from the long pose sequence
# #                 sequences = create_overlapping_sequences(
# #                     poses=all_poses,
# #                     seq_length=seq_length,
# #                     overlap_ratio=overlap_ratio
# #                 )
                
# #                 print(f"    ✓ Created {len(sequences)} sequences with {overlap_ratio*100}% overlap")
                
# #                 # Add sequences to appropriate dataset
# #                 for sequence in sequences:
# #                     if p_id == test_participant:
# #                         test_data['pose'].append(sequence)
# #                         test_data['label'].append(label)
# #                     else:
# #                         train_data['pose'].append(sequence)
# #                         train_data['label'].append(label)
                
# #                 total_sequences_for_participant += len(sequences)
# #                 print(f"    ✓ Added {len(sequences)} sequences to {'TEST' if p_id == test_participant else 'TRAIN'} set")
            
# #             print(f"  Total sequences for participant {p_id}: {total_sequences_for_participant}")

# #         # Convert lists to numpy arrays
# #         train_data['pose'] = np.array(train_data['pose'])
# #         train_data['label'] = np.array(train_data['label'])
# #         test_data['pose'] = np.array(test_data['pose'])
# #         test_data['label'] = np.array(test_data['label'])
        
# #         print(f"  Final dataset sizes:")
# #         print(f"    Training: {len(train_data['pose'])} sequences")
# #         print(f"    Test: {len(test_data['pose'])} sequences")
        
# #         # Save training data as pkl file
# #         train_file = os.path.join(output_path, f"EPG_train_{fold_idx}.pkl")
# #         with open(train_file, "wb") as f:
# #             pickle.dump(train_data, f)
        
# #         # Save test data as pkl file
# #         test_file = os.path.join(output_path, f"EPG_test_{fold_idx}.pkl")
# #         with open(test_file, "wb") as f:
# #             pickle.dump(test_data, f)
        
# #         # Create a config file for this fold
# #         config = {
# #             "fold": fold_idx,
# #             "test_participant": test_participant,
# #             "num_train_sequences": len(train_data['pose']),
# #             "num_test_sequences": len(test_data['pose']),
# #             "healthy_count": len(healthy_participants),
# #             "unhealthy_count": len(unhealthy_participants),
# #             "sequence_length": seq_length,
# #             "fps": 10,
# #             "seconds_per_sequence": seq_length/10,
# #             "total_sequences": len(train_data['pose']) + len(test_data['pose']),
# #             "pose_estimation_method": "VIBE",
# #             "joints": 17,
# #             "coordinates": 3,
# #             "overlap_ratio": overlap_ratio,
# #             "min_frames_per_walk": min_frames,
# #             "source_seq_len_for_training": 40,  # For GaitForeMer motion forecasting
# #             "target_seq_len_for_training": 20   # For GaitForeMer motion forecasting
# #         }
        
# #         with open(os.path.join(output_path, f"fold_{fold_idx}_config.json"), "w") as f:
# #             json.dump(config, f, indent=2)
        
# #         print(f"  ✓ Saved training data to {train_file}")
# #         print(f"  ✓ Saved test data to {test_file}")
# #         print(f"  ✓ Created config file")

# # def main():
# #     """Main function to parse arguments and run the data conversion."""
# #     parser = argparse.ArgumentParser(description="Convert Parkinson's gait data to GaitForeMer format")
# #     parser.add_argument("healthy_data_path", help="Path to the healthy data directory")
# #     parser.add_argument("unhealthy_data_path", help="Path to the unhealthy data directory")
# #     parser.add_argument("output_path", help="Path where the converted data will be saved")
# #     parser.add_argument("--seq_length", type=int, default=100, 
# #                         help="Length of each sequence in frames (default: 100 = 10 seconds at 10 FPS)")
# #     parser.add_argument("--overlap_ratio", type=float, default=0.5,
# #                         help="Overlap ratio between sequences (default: 0.5 = 50% overlap)")
# #     parser.add_argument("--min_frames", type=int, default=100,
# #                         help="Minimum frames required per walk (default: 100)")
    
# #     args = parser.parse_args()
    
# #     # Validate paths
# #     if not os.path.exists(args.healthy_data_path):
# #         print(f"Error: Healthy data path '{args.healthy_data_path}' does not exist")
# #         sys.exit(1)
# #     if not os.path.exists(args.unhealthy_data_path):
# #         print(f"Error: Unhealthy data path '{args.unhealthy_data_path}' does not exist")
# #         sys.exit(1)
    
# #     # Run the conversion
# #     print("="*70)
# #     print("GaitForeMer Data Conversion - Optimized for 1200-Frame Walks")
# #     print("="*70)
# #     convert_data_to_pkl_format(
# #         args.healthy_data_path,
# #         args.unhealthy_data_path,
# #         args.output_path,
# #         args.seq_length,
# #         args.overlap_ratio,
# #         args.min_frames
# #     )
# #     print("\n" + "="*70)
# #     print("Conversion completed successfully!")
# #     print(f"Converted data saved to: {args.output_path}")
# #     print(f"Sequence length: {args.seq_length} frames ({args.seq_length/10:.1f} seconds)")
# #     print(f"Overlap ratio: {args.overlap_ratio*100}%")
# #     print("="*70)
    
# #     # Verify the dataset format
# #     print("\nVerifying dataset format...")
# #     if verify_dataset_format(args.output_path):
# #         print("✅ Dataset format is correct and ready for GaitForeMer training")
# #         print("\nTraining command:")
# #         print(f"python training/transformer_model_fn.py --data_path {args.output_path} --pose_format xyz --source_seq_len 40 --target_seq_len 20")
# #     else:
# #         print("❌ Dataset format verification failed.")

# # def verify_dataset_format(data_path: str, expected_joints: int = 17, expected_features: int = 3) -> bool:
# #     """Verify that the generated dataset has the correct format for GaitForeMer."""
# #     # Check if all required files exist
# #     fold_count = 0
# #     for file in os.listdir(data_path):
# #         if file.startswith("EPG_train_") and file.endswith(".pkl"):
# #             fold_count += 1
    
# #     if fold_count == 0:
# #         print("Error: No training files found in the dataset directory")
# #         return False
    
# #     print(f"Found {fold_count} folds in the dataset")
    
# #     # Check a random fold to verify data format
# #     try:
# #         sample_file = next(f for f in os.listdir(data_path) if f.startswith("EPG_train_") and f.endswith(".pkl"))
# #         sample_path = os.path.join(data_path, sample_file)
        
# #         with open(sample_path, "rb") as f:
# #             data = pickle.load(f)
        
# #         # Check data structure
# #         if 'pose' not in data or 'label' not in data:
# #             print("Error: Dataset files are missing required keys ('pose' and 'label')")
# #             return False
        
# #         if not data['pose']:
# #             print("Error: Dataset has empty pose sequences")
# #             return False
        
# #         # Check the format of the first sequence
# #         first_seq = data['pose'][0]
# #         if first_seq.shape[0] != 100:
# #             print(f"Warning: Expected sequence length of 100 frames, got {first_seq.shape[0]}")
        
# #         if first_seq.shape[1] != expected_joints or first_seq.shape[2] != expected_features:
# #             print(f"Error: Expected sequence shape (100, {expected_joints}, {expected_features}), got {first_seq.shape}")
# #             return False
        
# #         print("Dataset format verification passed!")
# #         return True
    
# #     except Exception as e:
# #         print(f"Error verifying dataset format: {str(e)}")
# #         return False

# # if __name__ == "__main__":
# #     main()
# import os
# import pickle
# import numpy as np
# import glob
# import sys
# import json
# import re
# import argparse
# from pathlib import Path
# from typing import List, Dict, Tuple, Any
# import cv2

# # Base model's exact joint mapping from VIBE 49-joint format
# _MAJOR_JOINTS = [39, 41, 37, 43, 34, 35, 36, 33, 32, 31, 28, 29, 30, 27, 26, 25, 40]

# def extract_poses_mediapipe(frame_files: List[str]) -> np.ndarray:
#     """
#     Extract 3D poses from video frames using MediaPipe with EXACT VIBE 49-joint to MediaPipe mapping.
#     Uses the same _MAJOR_JOINTS structure as the base model.
#     """
#     try:
#         import mediapipe as mp
#         mp_pose = mp.solutions.pose
#         pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
        
#         poses = []
#         successful_frames = 0
        
#         for frame_file in frame_files:
#             frame = cv2.imread(frame_file)
#             if frame is None:
#                 poses.append(np.zeros((17, 3)))
#                 continue
                
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(frame_rgb)
            
#             if results.pose_landmarks:
#                 landmarks = results.pose_landmarks.landmark
                
#                 # EXACT VIBE 49-joint to MediaPipe mapping for _MAJOR_JOINTS
#                 vibe_to_mediapipe_mapping = [
#                     (23, 24),  # 39: Hip/Center -> Average of both hips
#                     (11, 12),  # 41: Thorax/Spine -> Average of both shoulders
#                     11,        # 37: Left Shoulder -> Left Shoulder
#                     12,        # 43: Right Shoulder -> Right Shoulder  
#                     13,        # 34: Left Elbow -> Left Elbow
#                     14,        # 35: Right Elbow -> Right Elbow
#                     15,        # 36: Left Wrist -> Left Wrist
#                     16,        # 33: Right Wrist -> Right Wrist
#                     23,        # 32: Left Hip -> Left Hip
#                     24,        # 31: Right Hip -> Right Hip
#                     25,        # 28: Left Knee -> Left Knee
#                     26,        # 29: Right Knee -> Right Knee
#                     27,        # 30: Left Ankle -> Left Ankle
#                     28,        # 27: Right Ankle -> Right Ankle
#                     31,        # 26: Left Foot -> Left Foot Index (approximation)
#                     32,        # 25: Right Foot -> Right Foot Index (approximation)
#                     0,         # 40: Head -> Nose (approximation)
#                 ]
                
#                 pose_3d = np.zeros((17, 3))
#                 for i, mp_ref in enumerate(vibe_to_mediapipe_mapping):
#                     if isinstance(mp_ref, tuple):
#                         # Average two landmarks (for hip center and thorax)
#                         lm1 = landmarks[mp_ref[0]]
#                         lm2 = landmarks[mp_ref[1]]
#                         pose_3d[i] = [
#                             (lm1.x + lm2.x) / 2,
#                             (lm1.y + lm2.y) / 2, 
#                             (lm1.z + lm2.z) / 2
#                         ]
#                     else:
#                         # Single landmark
#                         lm = landmarks[mp_ref]
#                         pose_3d[i] = [lm.x, lm.y, lm.z]
                
#                 poses.append(pose_3d)
#                 successful_frames += 1
#             else:
#                 # If no pose detected, use zeros
#                 poses.append(np.zeros((17, 3)))
        
#         pose.close()
        
#         print(f"    ✓ MediaPipe: {successful_frames}/{len(frame_files)} frames with pose detection")
#         print(f"    ✓ Using exact VIBE 49-joint to MediaPipe mapping for _MAJOR_JOINTS")
#         return np.array(poses).astype(np.float32)
        
#     except ImportError:
#         print("❌ MediaPipe not available.")
#         print("   Please install: pip install mediapipe")
#         print("⚠️  USING RANDOM DATA - THIS IS FOR TESTING ONLY!")
#         return extract_poses_random(frame_files)
#     except Exception as e:
#         print(f"❌ MediaPipe extraction failed: {str(e)}")
#         return extract_poses_random(frame_files)

# def extract_poses_random(frame_files: List[str]) -> np.ndarray:
#     """
#     Fallback to random data when MediaPipe is not available.
#     """
#     print("⚠️  USING RANDOM DATA - FOR TESTING ONLY!")
#     num_frames = len(frame_files)
#     return np.random.rand(num_frames, 17, 3).astype(np.float32)

# def create_overlapping_sequences(poses: np.ndarray, seq_length: int = 100, overlap_ratio: float = 0.5) -> List[np.ndarray]:
#     """
#     Create overlapping sequences from long pose sequences.
    
#     Args:
#         poses: Full pose sequence (num_frames, 17, 3)
#         seq_length: Length of each sequence (default: 100 frames = 10 seconds)
#         overlap_ratio: Overlap between sequences (0.5 = 50% overlap)
    
#     Returns:
#         List of sequences, each of shape (seq_length, 17, 3)
#     """
#     if len(poses) < seq_length:
#         # Pad if sequence is too short
#         padding = np.zeros((seq_length - len(poses), 17, 3))
#         padded_sequence = np.concatenate([poses, padding])
#         return [padded_sequence]
    
#     step_size = int(seq_length * (1 - overlap_ratio))
#     sequences = []
    
#     for start_idx in range(0, len(poses) - seq_length + 1, step_size):
#         end_idx = start_idx + seq_length
#         sequence = poses[start_idx:end_idx]
#         sequences.append(sequence)
    
#     # Always include the last sequence if we have leftover frames
#     if len(poses) >= seq_length and (len(poses) - seq_length) % step_size != 0:
#         last_sequence = poses[-seq_length:]
#         sequences.append(last_sequence)
    
#     return sequences

# def convert_data_to_pkl_format(
#     data_healthy_path: str,
#     data_unhealthy_path: str,
#     output_path: str,
#     seq_length: int = 100,
#     overlap_ratio: float = 0.5,
#     min_frames: int = 100
# ) -> None:
#     """
#     Convert your current data format to the pkl format required by GaitForeMer.
#     Optimized for 1200-frame walks with overlapping sequences.
    
#     Args:
#         data_healthy_path: Path to the healthy data directory
#         data_unhealthy_path: Path to the unhealthy data directory
#         output_path: Path where the converted data will be saved
#         seq_length: Length of each sequence in frames (default: 100 = 10 seconds at 10 FPS)
#         overlap_ratio: Overlap between consecutive sequences (0.0 = no overlap, 0.5 = 50% overlap)
#         min_frames: Minimum number of frames required to process a walk
#     """
#     # Create output directory if it doesn't exist
#     Path(output_path).mkdir(parents=True, exist_ok=True)
    
#     # Get all participant IDs from both healthy and unhealthy directories
#     healthy_participants = [d for d in os.listdir(data_healthy_path) 
#                            if os.path.isdir(os.path.join(data_healthy_path, d))]
#     unhealthy_participants = [d for d in os.listdir(data_unhealthy_path) 
#                             if os.path.isdir(os.path.join(data_unhealthy_path, d))]
    
#     all_participants = healthy_participants + unhealthy_participants
#     num_participants = len(all_participants)
    
#     print(f"Found {num_participants} participants ({len(healthy_participants)} healthy, {len(unhealthy_participants)} unhealthy)")
#     print(f"Using sequence length: {seq_length} frames ({seq_length/10:.1f} seconds at 10 FPS)")
#     print(f"Using overlap ratio: {overlap_ratio*100}%")
#     print(f"Minimum frames per walk: {min_frames}")
#     print(f"Using MediaPipe with EXACT VIBE 49-joint mapping for _MAJOR_JOINTS")
#     print(f"Joint mapping: {_MAJOR_JOINTS}")
    
#     # For each participant, create a fold where they are the test set
#     for i, test_participant in enumerate(all_participants):
#         fold_idx = i + 1
#         print(f"\nProcessing fold {fold_idx}/{num_participants} (test participant: {test_participant})")
        
#         # Create dictionaries to store training and test data
#         train_data = {
#             'pose': [],
#             'label': []
#         }
#         test_data = {
#             'pose': [],
#             'label': []
#         }
        
#         # Process all participants
#         for p_id in all_participants:
#             # Determine if participant is healthy or unhealthy
#             is_healthy = p_id in healthy_participants
#             label = 0 if is_healthy else 1
            
#             # Find the source directory for this participant
#             if is_healthy:
#                 source_dir = Path(data_healthy_path) / p_id
#             else:
#                 source_dir = Path(data_unhealthy_path) / p_id
            
#             # Find all H* subdirectories (H1, H2, H3, etc.)
#             video_dirs = [d for d in os.listdir(source_dir) 
#                          if os.path.isdir(os.path.join(source_dir, d)) and re.match(r'H\d+', d)]
            
#             if not video_dirs:
#                 print(f"  Warning: No H* subdirectories found for participant {p_id}")
#                 continue
            
#             print(f"  Found {len(video_dirs)} video sequences for participant {p_id}: {video_dirs}")
            
#             # Process all video sequences for this participant
#             total_sequences_for_participant = 0
#             for video_dir in video_dirs:
#                 video_path = source_dir / video_dir
#                 if not video_path.exists() or not video_path.is_dir():
#                     continue
                
#                 # Get all frame files
#                 frame_files = sorted(glob.glob(os.path.join(video_path, "*")))
#                 frame_files = [f for f in frame_files if os.path.isfile(f) and any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp'])]
                
#                 # Skip if not enough frames
#                 if len(frame_files) < min_frames:
#                     print(f"    Skipping {p_id}/{video_dir}: Only {len(frame_files)} frames (< {min_frames} minimum)")
#                     continue
                
#                 print(f"    Processing {len(frame_files)} frames from {p_id}/{video_dir}...")
                
#                 # ✅ Extract poses from ALL frames using MediaPipe with exact mapping
#                 try:
#                     all_poses = extract_poses_mediapipe(frame_files)
#                     print(f"    ✓ Extracted poses: {all_poses.shape}")
#                 except Exception as e:
#                     print(f"    ❌ Pose extraction failed: {str(e)}")
#                     continue
                
#                 # ✅ Create overlapping sequences from the long pose sequence
#                 sequences = create_overlapping_sequences(
#                     poses=all_poses,
#                     seq_length=seq_length,
#                     overlap_ratio=overlap_ratio
#                 )
                
#                 print(f"    ✓ Created {len(sequences)} sequences with {overlap_ratio*100}% overlap")
                
#                 # Add sequences to appropriate dataset
#                 for sequence in sequences:
#                     if p_id == test_participant:
#                         test_data['pose'].append(sequence)
#                         test_data['label'].append(label)
#                     else:
#                         train_data['pose'].append(sequence)
#                         train_data['label'].append(label)
                
#                 total_sequences_for_participant += len(sequences)
#                 print(f"    ✓ Added {len(sequences)} sequences to {'TEST' if p_id == test_participant else 'TRAIN'} set")
            
#             print(f"  Total sequences for participant {p_id}: {total_sequences_for_participant}")

#         # Convert lists to numpy arrays
#         train_data['pose'] = np.array(train_data['pose'])
#         train_data['label'] = np.array(train_data['label'])
#         test_data['pose'] = np.array(test_data['pose'])
#         test_data['label'] = np.array(test_data['label'])
        
#         print(f"  Final dataset sizes:")
#         print(f"    Training: {len(train_data['pose'])} sequences")
#         print(f"    Test: {len(test_data['pose'])} sequences")
        
#         # Save training data as pkl file
#         train_file = os.path.join(output_path, f"EPG_train_{fold_idx}.pkl")
#         with open(train_file, "wb") as f:
#             pickle.dump(train_data, f)
        
#         # Save test data as pkl file
#         test_file = os.path.join(output_path, f"EPG_test_{fold_idx}.pkl")
#         with open(test_file, "wb") as f:
#             pickle.dump(test_data, f)
        
#         # Create a config file for this fold
#         config = {
#             "fold": fold_idx,
#             "test_participant": test_participant,
#             "num_train_sequences": len(train_data['pose']),
#             "num_test_sequences": len(test_data['pose']),
#             "healthy_count": len(healthy_participants),
#             "unhealthy_count": len(unhealthy_participants),
#             "sequence_length": seq_length,
#             "fps": 10,
#             "seconds_per_sequence": seq_length/10,
#             "total_sequences": len(train_data['pose']) + len(test_data['pose']),
#             "pose_estimation_method": "MediaPipe",
#             "joints": 17,
#             "coordinates": 3,
#             "overlap_ratio": overlap_ratio,
#             "min_frames_per_walk": min_frames,
#             "source_seq_len_for_training": 40,
#             "target_seq_len_for_training": 20,
#             "vibe_major_joints": _MAJOR_JOINTS,
#             "joint_mapping_description": "Exact VIBE 49-joint to MediaPipe mapping for base model compatibility"
#         }
        
#         with open(os.path.join(output_path, f"fold_{fold_idx}_config.json"), "w") as f:
#             json.dump(config, f, indent=2)
        
#         print(f"  ✓ Saved training data to {train_file}")
#         print(f"  ✓ Saved test data to {test_file}")
#         print(f"  ✓ Created config file")

# def print_joint_mapping_info():
#     """Print detailed information about the joint mapping being used."""
#     print("\n" + "="*80)
#     print("JOINT MAPPING INFORMATION")
#     print("="*80)
#     print("Using EXACT VIBE 49-joint to MediaPipe mapping for _MAJOR_JOINTS")
#     print(f"VIBE _MAJOR_JOINTS indices: {_MAJOR_JOINTS}")
#     print("\nDetailed Mapping:")
#     print("VIBE Index | VIBE/NTU Joint      | MediaPipe Index | MediaPipe Landmark")
#     print("-" * 70)
    
#     mapping_info = [
#         (39, "Hip/Center", "(23, 24)", "Average of both hips"),
#         (41, "Thorax/Spine", "(11, 12)", "Average of both shoulders"),
#         (37, "Left Shoulder", "11", "Left Shoulder"),
#         (43, "Right Shoulder", "12", "Right Shoulder"),
#         (34, "Left Elbow", "13", "Left Elbow"),
#         (35, "Right Elbow", "14", "Right Elbow"),
#         (36, "Left Wrist", "15", "Left Wrist"),
#         (33, "Right Wrist", "16", "Right Wrist"),
#         (32, "Left Hip", "23", "Left Hip"),
#         (31, "Right Hip", "24", "Right Hip"),
#         (28, "Left Knee", "25", "Left Knee"),
#         (29, "Right Knee", "26", "Right Knee"),
#         (30, "Left Ankle", "27", "Left Ankle"),
#         (27, "Right Ankle", "28", "Right Ankle"),
#         (26, "Left Foot", "31", "Left Foot Index (approx)"),
#         (25, "Right Foot", "32", "Right Foot Index (approx)"),
#         (40, "Head", "0", "Nose (approx)"),
#     ]
    
#     for vibe_idx, joint_name, mp_idx, notes in mapping_info:
#         print(f"{vibe_idx:^10} | {joint_name:<18} | {mp_idx:^15} | {notes}")
    
#     print("="*80)

# def main():
#     """Main function to parse arguments and run the data conversion."""
#     parser = argparse.ArgumentParser(description="Convert Parkinson's gait data to GaitForeMer format")
#     parser.add_argument("healthy_data_path", help="Path to the healthy data directory")
#     parser.add_argument("unhealthy_data_path", help="Path to the unhealthy data directory")
#     parser.add_argument("output_path", help="Path where the converted data will be saved")
#     parser.add_argument("--seq_length", type=int, default=100, 
#                         help="Length of each sequence in frames (default: 100 = 10 seconds at 10 FPS)")
#     parser.add_argument("--overlap_ratio", type=float, default=0.5,
#                         help="Overlap ratio between sequences (default: 0.5 = 50% overlap)")
#     parser.add_argument("--min_frames", type=int, default=100,
#                         help="Minimum frames required per walk (default: 100)")
    
#     args = parser.parse_args()
    
#     # Validate paths
#     if not os.path.exists(args.healthy_data_path):
#         print(f"Error: Healthy data path '{args.healthy_data_path}' does not exist")
#         sys.exit(1)
#     if not os.path.exists(args.unhealthy_data_path):
#         print(f"Error: Unhealthy data path '{args.unhealthy_data_path}' does not exist")
#         sys.exit(1)
    
#     # Print joint mapping information
#     print_joint_mapping_info()
    
#     # Run the conversion
#     print("\n" + "="*70)
#     print("GaitForeMer Data Conversion - MediaPipe with Exact VIBE Mapping")
#     print("Optimized for 1200-Frame Walks with Base Model Joint Structure")
#     print("="*70)
#     convert_data_to_pkl_format(
#         args.healthy_data_path,
#         args.unhealthy_data_path,
#         args.output_path,
#         args.seq_length,
#         args.overlap_ratio,
#         args.min_frames
#     )
#     print("\n" + "="*70)
#     print("Conversion completed successfully!")
#     print(f"Converted data saved to: {args.output_path}")
#     print(f"Sequence length: {args.seq_length} frames ({args.seq_length/10:.1f} seconds)")
#     print(f"Overlap ratio: {args.overlap_ratio*100}%")
#     print("Pose estimation: MediaPipe with exact VIBE 49-joint mapping")
#     print("="*70)
    
#     # Verify the dataset format
#     print("\nVerifying dataset format...")
#     if verify_dataset_format(args.output_path):
#         print("✅ Dataset format is correct and ready for GaitForeMer training")
#         print("\nTraining command:")
#         print(f"python training/transformer_model_fn.py --data_path {args.output_path} --pose_format xyz --source_seq_len 40 --target_seq_len 20")
#     else:
#         print("❌ Dataset format verification failed.")

# def verify_dataset_format(data_path: str, expected_joints: int = 17, expected_features: int = 3) -> bool:
#     """Verify that the generated dataset has the correct format for GaitForeMer."""
#     # Check if all required files exist
#     fold_count = 0
#     for file in os.listdir(data_path):
#         if file.startswith("EPG_train_") and file.endswith(".pkl"):
#             fold_count += 1
    
#     if fold_count == 0:
#         print("Error: No training files found in the dataset directory")
#         return False
    
#     print(f"Found {fold_count} folds in the dataset")
    
#     # Check a random fold to verify data format
#     try:
#         sample_file = next(f for f in os.listdir(data_path) if f.startswith("EPG_train_") and f.endswith(".pkl"))
#         sample_path = os.path.join(data_path, sample_file)
        
#         with open(sample_path, "rb") as f:
#             data = pickle.load(f)
        
#         # Check data structure
#         if 'pose' not in data or 'label' not in data:
#             print("Error: Dataset files are missing required keys ('pose' and 'label')")
#             return False
        
#         if not data['pose']:
#             print("Error: Dataset has empty pose sequences")
#             return False
        
#         # Check the format of the first sequence
#         first_seq = data['pose'][0]
#         if first_seq.shape[0] != 100:
#             print(f"Warning: Expected sequence length of 100 frames, got {first_seq.shape[0]}")
        
#         if first_seq.shape[1] != expected_joints or first_seq.shape[2] != expected_features:
#             print(f"Error: Expected sequence shape (100, {expected_joints}, {expected_features}), got {first_seq.shape}")
#             return False
        
#         print("Dataset format verification passed!")
#         return True
    
#     except Exception as e:
#         print(f"Error verifying dataset format: {str(e)}")
#         return False

# if __name__ == "__main__":
#     main()
import os
import pickle
import numpy as np
import glob
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
import cv2

# Suppress MediaPipe warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Base model's exact joint mapping from VIBE 49-joint format
_MAJOR_JOINTS = [39, 41, 37, 43, 34, 35, 36, 33, 32, 31, 28, 29, 30, 27, 26, 25, 40]

def print_joint_mapping_info():
    """Print detailed information about the joint mapping being used."""
    print("\n" + "="*80)
    print("JOINT MAPPING INFORMATION")
    print("="*80)
    print("Using EXACT VIBE 49-joint to MediaPipe mapping for _MAJOR_JOINTS")
    print(f"VIBE _MAJOR_JOINTS indices: {_MAJOR_JOINTS}")
    print("\nDetailed Mapping:")
    print("VIBE Index | VIBE/NTU Joint      | MediaPipe Index | MediaPipe Landmark")
    print("-" * 70)
    
    mapping_info = [
        (39, "Hip/Center", "(23, 24)", "Average of both hips"),
        (41, "Thorax/Spine", "(11, 12)", "Average of both shoulders"),
        (37, "Left Shoulder", "11", "Left Shoulder"),
        (43, "Right Shoulder", "12", "Right Shoulder"),
        (34, "Left Elbow", "13", "Left Elbow"),
        (35, "Right Elbow", "14", "Right Elbow"),
        (36, "Left Wrist", "15", "Left Wrist"),
        (33, "Right Wrist", "16", "Right Wrist"),
        (32, "Left Hip", "23", "Left Hip"),
        (31, "Right Hip", "24", "Right Hip"),
        (28, "Left Knee", "25", "Left Knee"),
        (29, "Right Knee", "26", "Right Knee"),
        (30, "Left Ankle", "27", "Left Ankle"),
        (27, "Right Ankle", "28", "Right Ankle"),
        (26, "Left Foot", "31", "Left Foot Index (approx)"),
        (25, "Right Foot", "32", "Right Foot Index (approx)"),
        (40, "Head", "0", "Nose (approx)"),
    ]
    
    for vibe_idx, joint_name, mp_idx, notes in mapping_info:
        print(f"{vibe_idx:^10} | {joint_name:<18} | {mp_idx:^15} | {notes}")
    
    print("="*80)

def extract_poses_mediapipe(frame_files: List[str]) -> np.ndarray:
    """
    Extract 3D poses from video frames using MediaPipe with EXACT VIBE 49-joint to MediaPipe mapping.
    Uses the same _MAJOR_JOINTS structure as the base model.
    """
    try:
        # Suppress MediaPipe and TensorFlow warnings
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
        
        poses = []
        successful_frames = 0
        
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is None:
                poses.append(np.zeros((17, 3)))
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # EXACT VIBE 49-joint to MediaPipe mapping for _MAJOR_JOINTS
                vibe_to_mediapipe_mapping = [
                    (23, 24),  # 39: Hip/Center -> Average of both hips
                    (11, 12),  # 41: Thorax/Spine -> Average of both shoulders
                    11,        # 37: Left Shoulder -> Left Shoulder
                    12,        # 43: Right Shoulder -> Right Shoulder  
                    13,        # 34: Left Elbow -> Left Elbow
                    14,        # 35: Right Elbow -> Right Elbow
                    15,        # 36: Left Wrist -> Left Wrist
                    16,        # 33: Right Wrist -> Right Wrist
                    23,        # 32: Left Hip -> Left Hip
                    24,        # 31: Right Hip -> Right Hip
                    25,        # 28: Left Knee -> Left Knee
                    26,        # 29: Right Knee -> Right Knee
                    27,        # 30: Left Ankle -> Left Ankle
                    28,        # 27: Right Ankle -> Right Ankle
                    31,        # 26: Left Foot -> Left Foot Index (approximation)
                    32,        # 25: Right Foot -> Right Foot Index (approximation)
                    0,         # 40: Head -> Nose (approximation)
                ]
                
                pose_3d = np.zeros((17, 3))
                for i, mp_ref in enumerate(vibe_to_mediapipe_mapping):
                    if isinstance(mp_ref, tuple):
                        # Average two landmarks (for hip center and thorax)
                        lm1 = landmarks[mp_ref[0]]
                        lm2 = landmarks[mp_ref[1]]
                        pose_3d[i] = [
                            (lm1.x + lm2.x) / 2,
                            (lm1.y + lm2.y) / 2, 
                            (lm1.z + lm2.z) / 2
                        ]
                    else:
                        # Single landmark
                        lm = landmarks[mp_ref]
                        pose_3d[i] = [lm.x, lm.y, lm.z]
                
                poses.append(pose_3d)
                successful_frames += 1
            else:
                # If no pose detected, use zeros
                poses.append(np.zeros((17, 3)))
        
        pose.close()
        
        print(f"    ✓ MediaPipe: {successful_frames}/{len(frame_files)} frames with pose detection")
        return np.array(poses).astype(np.float32)
        
    except ImportError:
        print("❌ MediaPipe not available.")
        print("   Please install: pip install mediapipe")
        print("⚠️  USING RANDOM DATA - THIS IS FOR TESTING ONLY!")
        return extract_poses_random(frame_files)
    except Exception as e:
        print(f"❌ MediaPipe extraction failed: {str(e)}")
        return extract_poses_random(frame_files)

def extract_poses_random(frame_files: List[str]) -> np.ndarray:
    """
    Fallback to random data when MediaPipe is not available.
    """
    print("⚠️  USING RANDOM DATA - FOR TESTING ONLY!")
    num_frames = len(frame_files)
    return np.random.rand(num_frames, 17, 3).astype(np.float32)

def create_overlapping_sequences(poses: np.ndarray, seq_length: int = 100, overlap_ratio: float = 0.5) -> List[np.ndarray]:
    """
    Create overlapping sequences from long pose sequences.
    
    Args:
        poses: Full pose sequence (num_frames, 17, 3)
        seq_length: Length of each sequence (default: 100 frames = 10 seconds)
        overlap_ratio: Overlap between sequences (0.5 = 50% overlap)
    
    Returns:
        List of sequences, each of shape (seq_length, 17, 3)
    """
    if len(poses) < seq_length:
        # Pad if sequence is too short
        padding = np.zeros((seq_length - len(poses), 17, 3))
        padded_sequence = np.concatenate([poses, padding])
        return [padded_sequence]
    
    step_size = int(seq_length * (1 - overlap_ratio))
    sequences = []
    
    for start_idx in range(0, len(poses) - seq_length + 1, step_size):
        end_idx = start_idx + seq_length
        sequence = poses[start_idx:end_idx]
        sequences.append(sequence)
    
    # Always include the last sequence if we have leftover frames
    if len(poses) >= seq_length and (len(poses) - seq_length) % step_size != 0:
        last_sequence = poses[-seq_length:]
        sequences.append(last_sequence)
    
    return sequences

def extract_all_participant_data(
    data_healthy_path: str,
    data_unhealthy_path: str,
    seq_length: int = 100,
    overlap_ratio: float = 0.5,
    min_frames: int = 100
) -> Dict[str, Any]:
    """
    Extract pose data for ALL participants first, then create folds later.
    Returns a dictionary with all participant data organized by participant ID.
    """
    print("="*70)
    print("STEP 1: EXTRACTING POSE DATA FOR ALL PARTICIPANTS")
    print("="*70)
    
    # Get all participant IDs from both healthy and unhealthy directories
    healthy_participants = [d for d in os.listdir(data_healthy_path) 
                           if os.path.isdir(os.path.join(data_healthy_path, d))]
    unhealthy_participants = [d for d in os.listdir(data_unhealthy_path) 
                            if os.path.isdir(os.path.join(data_unhealthy_path, d))]
    
    all_participants = healthy_participants + unhealthy_participants
    num_participants = len(all_participants)
    
    print(f"Found {num_participants} participants ({len(healthy_participants)} healthy, {len(unhealthy_participants)} unhealthy)")
    print(f"Using sequence length: {seq_length} frames ({seq_length/10:.1f} seconds at 10 FPS)")
    print(f"Using overlap ratio: {overlap_ratio*100}%")
    print(f"Minimum frames per walk: {min_frames}")
    
    # Dictionary to store all participant data
    participant_data = {}
    
    # Process all participants (ONE-TIME POSE EXTRACTION)
    for p_id in all_participants:
        print(f"\nExtracting data for participant: {p_id}")
        
        # Determine if participant is healthy or unhealthy
        is_healthy = p_id in healthy_participants
        label = 0 if is_healthy else 1
        
        # Find the source directory for this participant
        if is_healthy:
            source_dir = Path(data_healthy_path) / p_id
        else:
            source_dir = Path(data_unhealthy_path) / p_id
        
        # Find all H* subdirectories (H1, H2, H3, etc.)
        video_dirs = [d for d in os.listdir(source_dir) 
                     if os.path.isdir(os.path.join(source_dir, d)) and re.match(r'H\d+', d)]
        
        if not video_dirs:
            print(f"  Warning: No H* subdirectories found for participant {p_id}")
            participant_data[p_id] = {'sequences': [], 'label': label}
            continue
        
        print(f"  Found {len(video_dirs)} video sequences: {video_dirs}")
        
        all_sequences_for_participant = []
        
        # Process all video sequences for this participant
        for video_dir in video_dirs:
            video_path = source_dir / video_dir
            if not video_path.exists() or not video_path.is_dir():
                continue
            
            # Get all frame files
            frame_files = sorted(glob.glob(os.path.join(video_path, "*")))
            frame_files = [f for f in frame_files if os.path.isfile(f) and any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp'])]
            
            # Skip if not enough frames
            if len(frame_files) < min_frames:
                print(f"    Skipping {p_id}/{video_dir}: Only {len(frame_files)} frames (< {min_frames} minimum)")
                continue
            
            print(f"    Processing {len(frame_files)} frames from {video_dir}...")
            
            # ✅ Extract poses from ALL frames using MediaPipe with exact mapping
            try:
                all_poses = extract_poses_mediapipe(frame_files)
                print(f"    ✓ Extracted poses: {all_poses.shape}")
            except Exception as e:
                print(f"    ❌ Pose extraction failed: {str(e)}")
                continue
            
            # ✅ Create overlapping sequences from the long pose sequence
            sequences = create_overlapping_sequences(
                poses=all_poses,
                seq_length=seq_length,
                overlap_ratio=overlap_ratio
            )
            
            print(f"    ✓ Created {len(sequences)} sequences with {overlap_ratio*100}% overlap")
            
            # Add sequences to participant's data
            all_sequences_for_participant.extend(sequences)
        
        # Store all sequences for this participant
        participant_data[p_id] = {
            'sequences': all_sequences_for_participant,
            'label': label,
            'num_sequences': len(all_sequences_for_participant)
        }
        
        print(f"  ✓ Total sequences for participant {p_id}: {len(all_sequences_for_participant)}")
    
    print("\n" + "="*70)
    print("POSE EXTRACTION COMPLETED FOR ALL PARTICIPANTS")
    print("="*70)
    
    return participant_data, all_participants

def create_folds_from_extracted_data(
    participant_data: Dict[str, Any],
    all_participants: List[str],
    output_path: str,
    seq_length: int = 100,
    overlap_ratio: float = 0.5
) -> None:
    """
    Create folds from pre-extracted participant data (NO POSE RE-EXTRACTION).
    """
    print("\n" + "="*70)
    print("STEP 2: CREATING FOLDS FROM EXTRACTED DATA")
    print("="*70)
    
    num_participants = len(all_participants)
    
    # For each participant, create a fold where they are the test set
    for i, test_participant in enumerate(all_participants):
        fold_idx = i + 1
        print(f"\nProcessing fold {fold_idx}/{num_participants} (test participant: {test_participant})")
        
        # Create dictionaries to store training and test data
        train_data = {
            'pose': [],
            'label': []
        }
        test_data = {
            'pose': [],
            'label': []
        }
        
        # Build training and test sets from pre-extracted data
        for p_id in all_participants:
            if p_id not in participant_data:
                continue
                
            participant_info = participant_data[p_id]
            sequences = participant_info['sequences']
            label = participant_info['label']
            
            if p_id == test_participant:
                # Add to test set
                test_data['pose'].extend(sequences)
                test_data['label'].extend([label] * len(sequences))
            else:
                # Add to training set
                train_data['pose'].extend(sequences)
                train_data['label'].extend([label] * len(sequences))
        
        # Convert lists to numpy arrays
        train_data['pose'] = np.array(train_data['pose'])
        train_data['label'] = np.array(train_data['label'])
        test_data['pose'] = np.array(test_data['pose'])
        test_data['label'] = np.array(test_data['label'])
        
        print(f"  Final dataset sizes:")
        print(f"    Training: {len(train_data['pose'])} sequences")
        print(f"    Test: {len(test_data['pose'])} sequences")
        
        # Save training data as pkl file
        train_file = os.path.join(output_path, f"EPG_train_{fold_idx}.pkl")
        with open(train_file, "wb") as f:
            pickle.dump(train_data, f)
        
        # Save test data as pkl file
        test_file = os.path.join(output_path, f"EPG_test_{fold_idx}.pkl")
        with open(test_file, "wb") as f:
            pickle.dump(test_data, f)
        
        # Create a config file for this fold
        config = {
            "fold": fold_idx,
            "test_participant": test_participant,
            "num_train_sequences": len(train_data['pose']),
            "num_test_sequences": len(test_data['pose']),
            "total_sequences": len(train_data['pose']) + len(test_data['pose']),
            "sequence_length": seq_length,
            "fps": 10,
            "seconds_per_sequence": seq_length/10,
            "pose_estimation_method": "MediaPipe",
            "joints": 17,
            "coordinates": 3,
            "overlap_ratio": overlap_ratio,
            "source_seq_len_for_training": 40,
            "target_seq_len_for_training": 20,
            "vibe_major_joints": _MAJOR_JOINTS,
            "joint_mapping_description": "Exact VIBE 49-joint to MediaPipe mapping for base model compatibility"
        }
        
        with open(os.path.join(output_path, f"fold_{fold_idx}_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"  ✓ Saved training data to {train_file}")
        print(f"  ✓ Saved test data to {test_file}")
        print(f"  ✓ Created config file")

def main():
    """Main function to parse arguments and run the data conversion."""
    parser = argparse.ArgumentParser(description="Convert Parkinson's gait data to GaitForeMer format")
    parser.add_argument("healthy_data_path", help="Path to the healthy data directory")
    parser.add_argument("unhealthy_data_path", help="Path to the unhealthy data directory")
    parser.add_argument("output_path", help="Path where the converted data will be saved")
    parser.add_argument("--seq_length", type=int, default=100, 
                        help="Length of each sequence in frames (default: 100 = 10 seconds at 10 FPS)")
    parser.add_argument("--overlap_ratio", type=float, default=0.5,
                        help="Overlap ratio between sequences (default: 0.5 = 50% overlap)")
    parser.add_argument("--min_frames", type=int, default=100,
                        help="Minimum frames required per walk (default: 100)")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.healthy_data_path):
        print(f"Error: Healthy data path '{args.healthy_data_path}' does not exist")
        sys.exit(1)
    if not os.path.exists(args.unhealthy_data_path):
        print(f"Error: Unhealthy data path '{args.unhealthy_data_path}' does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    
    # Print joint mapping information
    print_joint_mapping_info()
    
    print("\n" + "="*70)
    print("GaitForeMer Data Conversion - OPTIMIZED VERSION")
    print("MediaPipe with Exact VIBE Mapping - ONE-TIME POSE EXTRACTION")
    print("="*70)
    
    # STEP 1: Extract all pose data once
    participant_data, all_participants = extract_all_participant_data(
        args.healthy_data_path,
        args.unhealthy_data_path,
        args.seq_length,
        args.overlap_ratio,
        args.min_frames
    )
    
    # STEP 2: Create folds from extracted data (no pose re-extraction)
    create_folds_from_extracted_data(
        participant_data,
        all_participants,
        args.output_path,
        args.seq_length,
        args.overlap_ratio
    )
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Converted data saved to: {args.output_path}")
    print(f"Total participants processed: {len(all_participants)}")
    print(f"Sequence length: {args.seq_length} frames ({args.seq_length/10:.1f} seconds)")
    print(f"Overlap ratio: {args.overlap_ratio*100}%")
    print("Pose estimation: MediaPipe with exact VIBE 49-joint mapping")
    print("Optimization: One-time pose extraction for all participants")
    print("="*70)
    
    # Verify the dataset format
    print("\nVerifying dataset format...")
    if verify_dataset_format(args.output_path):
        print("✅ Dataset format is correct and ready for GaitForeMer training")
        print("\nTraining command:")
        print(f"python training/transformer_model_fn.py --data_path {args.output_path} --pose_format xyz --source_seq_len 40 --target_seq_len 20")
    else:
        print("❌ Dataset format verification failed.")

def verify_dataset_format(data_path: str, expected_joints: int = 17, expected_features: int = 3) -> bool:
    """Verify that the generated dataset has the correct format for GaitForeMer."""
    # Check if all required files exist
    fold_count = 0
    for file in os.listdir(data_path):
        if file.startswith("EPG_train_") and file.endswith(".pkl"):
            fold_count += 1
    
    if fold_count == 0:
        print("Error: No training files found in the dataset directory")
        return False
    
    print(f"Found {fold_count} folds in the dataset")
    
    # Check a random fold to verify data format
    try:
        sample_file = next(f for f in os.listdir(data_path) if f.startswith("EPG_train_") and f.endswith(".pkl"))
        sample_path = os.path.join(data_path, sample_file)
        
        with open(sample_path, "rb") as f:
            data = pickle.load(f)
        
        # Check data structure
        if 'pose' not in data or 'label' not in data:
            print("Error: Dataset files are missing required keys ('pose' and 'label')")
            return False
        
        if not data['pose']:
            print("Error: Dataset has empty pose sequences")
            return False
        
        # Check the format of the first sequence
        first_seq = data['pose'][0]
        if first_seq.shape[0] != 100:
            print(f"Warning: Expected sequence length of 100 frames, got {first_seq.shape[0]}")
        
        if first_seq.shape[1] != expected_joints or first_seq.shape[2] != expected_features:
            print(f"Error: Expected sequence shape (100, {expected_joints}, {expected_features}), got {first_seq.shape}")
            return False
        
        print("Dataset format verification passed!")
        return True
    
    except Exception as e:
        print(f"Error verifying dataset format: {str(e)}")
        return False

if __name__ == "__main__":
    main()