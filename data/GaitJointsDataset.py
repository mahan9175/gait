import os
import sys
import numpy as np
import torch
import argparse
import tqdm
import pickle
import random

_TOTAL_ACTIONS = 4

# Mapping from 1-base of NTU to vibe 49 joints
# hip, thorax, 
_MAJOR_JOINTS = [39, 41, 37, 43, 34, 35, 36, 33, 32, 31, 28, 29, 30, 27, 26, 25, 40]
_NMAJOR_JOINTS = len(_MAJOR_JOINTS)
_MIN_STD = 1e-4
_SPINE_ROOT = 0  # after only taking major joints (ie index in _MAJOR_JOINTS)

# CRITICAL CHANGE: Set to 3 to match pre-trained model's expected dimension
_JOINT_DIMENSIONS = 3  # Was 2 - now 3 to match checkpoint dimensions

def collate_fn(batch):
    """Collate function for data loaders."""
    e_inp = torch.from_numpy(np.stack([e['encoder_inputs'] for e in batch]))
    d_inp = torch.from_numpy(np.stack([e['decoder_inputs'] for e in batch]))
    d_out = torch.from_numpy(np.stack([e['decoder_outputs'] for e in batch]))
    action_id = torch.from_numpy(np.stack([e['action_id'] for e in batch]))
    action = [e['action_str'] for e in batch]

    batch_ = {
        'encoder_inputs': e_inp,
        'decoder_inputs': d_inp,
        'decoder_outputs': d_out,
        'action_str': action,
        'action_ids': action_id
    }

    return batch_

class GaitJointsDataset(torch.utils.data.Dataset):
    def __init__(self, params=None, mode='train', fold=1):
        super(GaitJointsDataset, self).__init__()
        self._params = params
        self._mode = mode
        thisname = self.__class__.__name__
        self._monitor_action = 'normal'

        self._action_str = ['normal', 'slight', 'moderate', 'severe']
        self.data_dir = self._params['data_path']
        self.fold = fold

        self.load_data()

    def load_data(self):
        train_data = pickle.load(open(os.path.join(self.data_dir, f"EPG_train_{self.fold}.pkl"), "rb"))
        test_data = pickle.load(open(os.path.join(self.data_dir, f"EPG_test_{self.fold}.pkl"), "rb"))
        
        if self._mode == 'train':
            X_1, Y = self.data_generator(train_data, mode='train', fold_number=self.fold) 
        else:
            X_1, Y = self.data_generator(test_data) 
        
        self.X_1 = X_1
        self.Y = Y
        self._action_str = ['none', 'mild', 'moderate', 'severe']

        # Now using 3D dimensions (51 features)
        self._pose_dim = _JOINT_DIMENSIONS * _NMAJOR_JOINTS  # 3 × 17 = 51
        self._data_dim = self._pose_dim
        
        # Print shape info for debugging
        print(f"Dataset loaded with joint dimensions: {_JOINT_DIMENSIONS}")
        print(f"Input size: {self._pose_dim} features (joints × dimensions)")
        print(f"Number of samples: {len(self.Y)}")
        if len(self.X_1) > 0:
            sample_shape = self.X_1[0].shape
            print(f"Sample pose shape: {sample_shape}")
            print(f"Flattened features per frame: {sample_shape[1] * sample_shape[2]}")
            # Verify compatibility
            expected_features = _NMAJOR_JOINTS * _JOINT_DIMENSIONS
            actual_features = sample_shape[1] * sample_shape[2]
            if actual_features != expected_features:
                print(f"⚠️ WARNING: Expected {expected_features} features, got {actual_features}")
                print(f"   This might cause shape mismatch errors!")

    def data_generator(self, T, mode='test', fold_number=1):
        X_1 = []
        Y = []

        total_num_clips = 0
        for i in range(len(T['pose'])): 
            total_num_clips += 1
            p = np.copy(T['pose'][i])
            # CRITICAL: Convert 2D poses (N,17,2) to 3D (N,17,3) if needed
            if p.shape[-1] == 2:  # Only 2 dimensions
                p_3d = np.zeros((p.shape[0], p.shape[1], 3), dtype=p.dtype)
                p_3d[..., :2] = p  # Copy x,y coordinates
                p = p_3d
            y_label_index = T['label'][i]
            label = y_label_index
            X_1.append(p)
            Y.append(label)
        
        Y = np.stack(Y)
        return X_1, Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self._get_item_train(idx)

    def _get_item_train(self, idx):
        """Get item for the training mode."""
        x = self.X_1[idx]  # Shape: [frames, 17, 3] after padding
        
        # CRITICAL: Ensure 3D format even if stored as 2D
        if x.shape[-1] == 2:  # Safety check
            x_padded = np.zeros((x.shape[0], x.shape[1], 3), dtype=x.dtype)
            x_padded[..., :2] = x
            x = x_padded

        y = self.Y[idx]

        action_id = y
        source_seq_len = self._params['source_seq_len']
        target_seq_len = self._params['target_seq_len']
        
        # Now using 3D dimensions (51 features)
        input_size = _JOINT_DIMENSIONS * _NMAJOR_JOINTS  # 51
        pose_size = _JOINT_DIMENSIONS * _NMAJOR_JOINTS   # 51
        
        total_frames = source_seq_len + target_seq_len
        src_seq_len = source_seq_len - 1

        encoder_inputs = np.zeros((src_seq_len, input_size), dtype=np.float32)
        decoder_inputs = np.zeros((target_seq_len, input_size), dtype=np.float32)
        decoder_outputs = np.zeros((target_seq_len, pose_size), dtype=np.float32)

        # Reshape to [frames, features] - now 51 features
        N = x.shape[0]
        x = x.reshape(N, -1)  # Flattens to (frames, 51)
        
        # Ensure we have enough frames
        if N < total_frames:
            padding = np.zeros((total_frames - N, x.shape[1]), dtype=x.dtype)
            x = np.vstack([x, padding])
            N = x.shape[0]
        
        # Random start frame
        start_frame = random.randint(0, max(0, N - total_frames))
        
        data_sel = x[start_frame:(start_frame + total_frames), :]
        
        # Assign to encoder and decoder inputs/outputs
        encoder_inputs[:, :] = data_sel[0:src_seq_len, :input_size]
        decoder_inputs[:, :] = data_sel[src_seq_len:src_seq_len + target_seq_len, :input_size]
        decoder_outputs[:, :] = data_sel[source_seq_len:, :pose_size]

        if self._params['pad_decoder_inputs']:
            query = decoder_inputs[0:1, :]
            decoder_inputs = np.repeat(query, target_seq_len, axis=0)

        return {
            'encoder_inputs': encoder_inputs, 
            'decoder_inputs': decoder_inputs, 
            'decoder_outputs': decoder_outputs,
            'action_id': action_id,
            'action_str': self._action_str[action_id],
        }

def dataset_factory(params, fold):
    """Defines the datasets that will be used for training and validation."""
    params['num_activities'] = _TOTAL_ACTIONS
    params['virtual_dataset_size'] = params['steps_per_epoch'] * params['batch_size']
    params['n_joints'] = _NMAJOR_JOINTS
    
    # Update joint dimensions to 3D for model compatibility
    params['joint_dimensions'] = _JOINT_DIMENSIONS  # Now 3

    eval_mode = 'test' if 'test_phase' in params.keys() else 'eval'
    if eval_mode == 'test':
        train_dataset_fn = None
    else:
        train_dataset = GaitJointsDataset(params, mode='train', fold=fold)
        train_dataset_fn = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )
        eval_dataset = GaitJointsDataset(
            params, 
            mode=eval_mode,
            fold=fold,
        )
        eval_dataset_fn = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        ) 

    return train_dataset_fn, eval_dataset_fn
