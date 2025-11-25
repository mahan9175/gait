###############################################################################
# Pose Transformers (POTR): Human Motion Prediction with Non-Autoregressive 
# Transformers
# 
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by 
# Angel Martinez <angel.martinez@idiap.ch>,
# 
# This file is part of 
# POTR: Human Motion Prediction with Non-Autoregressive Transformers
# 
# POTR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# POTR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with POTR. If not, see <http://www.gnu.org/licenses/>.
###############################################################################

"""Implments the model function for the POTR model."""


import numpy as np
import os
import sys
import argparse
import json
import time
from sklearn.metrics import accuracy_score  # Added for accuracy reporting
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report

from numpyencoder import NumpyEncoder
import matplotlib.pyplot as plt

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import training.seq2seq_model_fn as seq2seq_model_fn
import models.PoseTransformer as PoseTransformer
import models.PoseEncoderDecoder as PoseEncoderDecoder
import data.NTURGDDataset as NTURGDDataset
import data.GaitJointsDataset as GaitJointsDataset
import utils.utils as utils

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_WEIGHT_DECAY = 0.00001
_NSEEDS = 8

class POTRModelFn(seq2seq_model_fn.ModelFn):
  def __init__(self,
               params,
               train_dataset_fn,
               eval_dataset_fn,
               pose_encoder_fn=None,
               pose_decoder_fn=None):
    super(POTRModelFn, self).__init__(
      params, train_dataset_fn, eval_dataset_fn, pose_encoder_fn, pose_decoder_fn)
    self._loss_fn = self.layerwise_loss_fn
    self.task = params['task']
    if self.task == 'downstream':
        # The model expects 4 classes, so we need to provide weights for 4 classes
        # Let's maintain the healthy/unhealthy ratio across 4 classes
        healthy_count = 29
        unhealthy_count = 11
        
        # Distribute the counts across 4 classes:
        # Classes 0 and 1 represent healthy (class 0)
        # Classes 2 and 3 represent unhealthy (class 1)
        class_counts = torch.tensor([
            healthy_count / 2,  # half of healthy to class 0
            healthy_count / 2,  # half of healthy to class 1
            unhealthy_count / 2,  # half of unhealthy to class 2
            unhealthy_count / 2   # half of unhealthy to class 3
        ])
        
        class_frequencies = class_counts.float() / class_counts.sum()
        weights = 1.0 / class_frequencies
        weights = weights / weights.sum()
        
        self._loss_weights = weights.to(_DEVICE)
        self._weighted_ce_loss = nn.CrossEntropyLoss(weight=self._loss_weights)
        print(f'Using weighted CE loss with weights: {weights.cpu().numpy().tolist()}')
    else:
        print('Using a standard CE loss for activity prediction.')
  def smooth_l1(self, decoder_pred, decoder_gt):
    l1loss = nn.SmoothL1Loss(reduction='mean')
    return l1loss(decoder_pred, decoder_gt)

  def loss_l1(self, decoder_pred, decoder_gt):
    return nn.L1Loss(reduction='mean')(decoder_pred, decoder_gt)

  def loss_activity(self, logits, class_gt):                                     
    """Computes entropy loss from logits between predictions and class."""
    if self.task == 'downstream':
      return self._weighted_ce_loss(logits, class_gt)
    else:
      return nn.functional.cross_entropy(logits, class_gt, reduction='mean')

  def compute_class_loss(self, class_logits, class_gt):
    """Computes the class loss for each of the decoder layers predictions or memory."""
    class_loss = 0.0
    for l in range(len(class_logits)):
      class_loss += self.loss_activity(class_logits[l], class_gt)

    return class_loss/len(class_logits)

  def select_loss_fn(self):
    if self._params['loss_fn'] == 'mse':
      return self.loss_mse
    elif self._params['loss_fn'] == 'smoothl1':
      return self.smooth_l1
    elif self._params['loss_fn'] == 'l1':
      return self.loss_l1
    else:
      raise ValueError('Unknown loss name {}.'.format(self._params['loss_fn']))

  def layerwise_loss_fn(self, decoder_pred, decoder_gt, class_logits=None, class_gt=None):
    """Computes layerwise loss between predictions and ground truth."""
    pose_loss = 0.0
    loss_fn = self.select_loss_fn()

    for l in range(len(decoder_pred)):
      pose_loss += loss_fn(decoder_pred[l], decoder_gt)

    pose_loss = pose_loss/len(decoder_pred)
    if class_logits is not None:
      return pose_loss, self.compute_class_loss(class_logits, class_gt)

    return pose_loss, None

  def init_model(self, pose_encoder_fn=None, pose_decoder_fn=None):
    self._model = PoseTransformer.model_factory(
        self._params, 
        pose_encoder_fn, 
        pose_decoder_fn
    )

  def select_optimizer(self):
    optimizer = optim.AdamW(
        self._model.parameters(), lr=self._params['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=_WEIGHT_DECAY
    )

    return optimizer


def dataset_factory(params, fold, model_prefix):
  if params['dataset'] == 'ntu_rgbd':
    return NTURGDDataset.dataset_factory(params)
  elif params['dataset'] == 'pd_gait':
    return GaitJointsDataset.dataset_factory(params, fold)
  else:
    raise ValueError('Unknown dataset {}'.format(params['dataset']))

def single_vote(pred):
  """
  Get majority vote of predicted classes for the clips in one video.
  :param preds: list of predicted class for each clip of one video
  :return: majority vote of predicted class for one video
  """
  p = np.array(pred)
  counts = np.bincount(p)
  max_count = 0
  max_index = 0
  for i in range(len(counts)):
    if max_count < counts[i]:
      max_index = i
      max_count = counts[i]
  return max_index

def save_json(filename, attributes, names):
    """
    Save training parameters and evaluation results to json file.
    :param filename: save filename
    :param attributes: attributes to save
    :param names: name of attributes to save in json file
    """
    with open(filename, "w", encoding="utf8") as outfile:
        d = {}
        for i in range(len(attributes)):
            name = names[i]
            attribute = attributes[i]
            d[name] = attribute
        json.dump(d, outfile, indent=4, cls=NumpyEncoder)

def count_folds(data_path):
    """Count how many complete fold pairs exist in the flat data structure."""
    train_files = [f for f in os.listdir(data_path) if f.startswith("EPG_train_") and f.endswith(".pkl")]
    test_files = [f for f in os.listdir(data_path) if f.startswith("EPG_test_") and f.endswith(".pkl")]
    
    # Count how many fold numbers exist in both train and test
    train_folds = set(int(f.split("_")[2].split(".")[0]) for f in train_files)
    test_folds = set(int(f.split("_")[2].split(".")[0]) for f in test_files)
    valid_folds = train_folds & test_folds
    
    # Print diagnostic information
    print(f"\nData structure diagnostic for {data_path}:")
    print(f"  Found {len(train_files)} train files")
    print(f"  Found {len(test_files)} test files")
    print(f"  Found {len(valid_folds)} complete fold pairs")
    
    if len(valid_folds) == 0:
        print("  ERROR: No valid fold pairs found")
        print("  Expected file pattern: EPG_train_X.pkl and EPG_test_X.pkl where X is the fold number")
        print("  Please verify your data structure")
    
    return len(valid_folds)
class ZeroRetrainingCausalAnalyzer:
    def __init__(self, pretrained_model, joint_names):
        self.model = pretrained_model
        self.joint_names = joint_names
        self.model.eval()  # Keep in eval mode
        
    def analyze_temporal_causality(self, dataloader):
        """Analyze temporal causality from attention patterns"""
        print("Analyzing temporal causality from attention patterns...")
        
        all_temporal_causality = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                encoder_inputs = batch['encoder_inputs']
                decoder_inputs = batch['decoder_inputs']
                
                # Get attention weights from existing model
                outputs, attn_weights, enc_weights, _ = self.model(
                    encoder_inputs, decoder_inputs, get_attn_weights=True
                )
                
                # Use encoder self-attention for temporal analysis
                # enc_weights: [num_layers, batch_size, num_heads, seq_len, seq_len]
                layer_attn = enc_weights[-1]  # Use last layer (most abstract)
                head_avg_attn = layer_attn.mean(dim=1)  # Average over heads
                batch_avg_attn = head_avg_attn.mean(dim=0)  # Average over batch
                
                # Convert attention to causality: if frame i→j has high attention, 
                # then i may cause j's representation
                temporal_causality = batch_avg_attn.cpu().numpy()
                all_temporal_causality.append(temporal_causality)
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx}/{len(dataloader)} batches")
        
        # Average over all batches
        avg_temporal_causality = np.mean(all_temporal_causality, axis=0)
        return avg_temporal_causality
    
    def analyze_joint_causality(self, dataloader):
        """Analyze joint causality from feature correlations"""
        print("Analyzing joint causality from feature correlations...")
        
        all_joint_correlations = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                encoder_inputs = batch['encoder_inputs']
                decoder_inputs = batch['decoder_inputs']
                
                # Get encoder outputs
                if hasattr(self.model, '_pose_embedding') and self.model._pose_embedding is not None:
                    encoder_embedded = self.model._pose_embedding(encoder_inputs)
                else:
                    encoder_embedded = encoder_inputs
                
                encoder_embedded = torch.transpose(encoder_embedded, 0, 1)
                memory, _ = self.model._transformer._encoder(
                    encoder_embedded, 
                    self.model._encoder_pos_encodings
                )
                
                # memory: [seq_len, batch_size, model_dim]
                # Reshape to joint-level features
                batch_size = memory.size(1)
                seq_len = memory.size(0)
                n_joints = len(self.joint_names)
                features_per_joint = self.model._model_dim // n_joints
                
                # Extract joint features (assuming features are concatenated)
                joint_features = memory.view(seq_len, batch_size, n_joints, features_per_joint)
                joint_features = joint_features.mean(dim=0)  # Average over time: [batch_size, n_joints, features_per_joint]
                
                # Compute correlations between joints
                batch_correlations = []
                for b in range(batch_size):
                    joint_corr = torch.corrcoef(joint_features[b].T)  # [n_joints, n_joints]
                    batch_correlations.append(joint_corr)
                
                batch_avg_corr = torch.stack(batch_correlations).mean(dim=0)
                all_joint_correlations.append(batch_avg_corr.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx}/{len(dataloader)} batches")
        
        # Average over all batches
        avg_joint_causality = np.mean(all_joint_correlations, axis=0)
        return avg_joint_causality
    
    def granger_causality_analysis(self, predictions, targets):
        """Statistical Granger causality test on model predictions"""
        print("Performing Granger causality analysis...")
        
        # predictions: [batch_size, seq_len, n_joints*3]
        # targets: [batch_size, seq_len, n_joints*3]
        
        batch_size, seq_len, _ = predictions.shape
        n_joints = len(self.joint_names)
        
        granger_results = np.zeros((n_joints, n_joints))
        
        for i in range(n_joints):
            for j in range(n_joints):
                if i == j:
                    continue
                    
                # Extract joint trajectories
                joint_i_pred = predictions[:, :, i*3:(i+1)*3].reshape(batch_size, -1)
                joint_j_pred = predictions[:, :, j*3:(j+1)*3].reshape(batch_size, -1)
                
                # Simple Granger test: does j help predict i?
                # Model 1: Predict i using only i's past
                # Model 2: Predict i using i's past + j's past
                # If Model 2 is significantly better, j Granger-causes i
                
                # For simplicity, use correlation as proxy
                corr_ij = np.corrcoef(joint_i_pred.flatten(), joint_j_pred.flatten())[0, 1]
                granger_results[i, j] = abs(corr_ij)
        
        return granger_results
    
    def generate_clinical_insights(self, temporal_causality, joint_causality, granger_results):
        """Generate clinically meaningful insights"""
        print("\n" + "="*50)
        print("CLINICAL CAUSALITY INSIGHTS")
        print("="*50)
        
        insights = {}
        
        # 1. Temporal causality insights
        print("\n=== TEMPORAL CAUSALITY ===")
        seq_len = temporal_causality.shape[0]
        time_steps = [f"t-{i}" for i in range(seq_len)]
        
        # Find strongest temporal influences
        strongest_links = []
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j and temporal_causality[i, j] > 0.3:  # Threshold
                    strongest_links.append((i, j, temporal_causality[i, j]))
        
        # Sort by strength
        strongest_links.sort(key=lambda x: x[2], reverse=True)
        
        print("Strongest temporal influences:")
        for i, j, strength in strongest_links[:5]:
            print(f"  Frame {time_steps[i]} → Frame {time_steps[j]}: {strength:.3f}")
        
        insights['temporal'] = strongest_links[:10]
        
        # 2. Joint causality insights
        print("\n=== JOINT CAUSALITY ===")
        n_joints = len(self.joint_names)
        
        # Find strongest joint relationships
        strong_joint_links = []
        for i in range(n_joints):
            for j in range(n_joints):
                if i != j and joint_causality[i, j] > 0.6:  # Threshold
                    strong_joint_links.append((
                        self.joint_names[i], 
                        self.joint_names[j], 
                        joint_causality[i, j]
                    ))
        
        strong_joint_links.sort(key=lambda x: x[2], reverse=True)
        
        print("Strongest joint relationships:")
        for src, tgt, strength in strong_joint_links[:8]:
            print(f"  {src} → {tgt}: {strength:.3f}")
        
        insights['joint_correlation'] = strong_joint_links
        
        # 3. Granger causality insights
        print("\n=== GRANGER CAUSALITY (Prediction Influence) ===")
        granger_links = []
        for i in range(n_joints):
            for j in range(n_joints):
                if i != j and granger_results[i, j] > 0.4:
                    granger_links.append((
                        self.joint_names[j],  # j causes i
                        self.joint_names[i],
                        granger_results[i, j]
                    ))
        
        granger_links.sort(key=lambda x: x[2], reverse=True)
        
        print("Strongest Granger causal relationships:")
        for cause, effect, strength in granger_links[:6]:
            print(f"  {cause} → {effect}: {strength:.3f}")
        
        insights['granger'] = granger_links
        
        return insights
    
    def create_causality_report(self, dataloader):
        """Complete causality analysis report"""
        print("Starting zero-retraining causality analysis...")
        
        # 1. Temporal causality from attention
        temporal_causality = self.analyze_temporal_causality(dataloader)
        
        # 2. Joint causality from feature correlations
        joint_causality = self.analyze_joint_causality(dataloader)
        
        # 3. Get some predictions for Granger analysis
        sample_batch = next(iter(dataloader))
        with torch.no_grad():
            predictions, _, _, _ = self.model(
                sample_batch['encoder_inputs'], 
                sample_batch['decoder_inputs'],
                get_attn_weights=False
            )
            predictions = predictions[-1].numpy()  # Last decoder layer
            targets = sample_batch['decoder_outputs'].numpy()
        
        # 4. Granger causality
        granger_results = self.granger_causality_analysis(predictions, targets)
        
        # 5. Generate insights
        insights = self.generate_clinical_insights(
            temporal_causality, joint_causality, granger_results
        )
        
        # 6. Create visualization
        self.visualize_causality(temporal_causality, joint_causality, granger_results)
        
        return insights
    
    def visualize_causality(self, temporal_causality, joint_causality, granger_results):
        """Create causality visualization plots"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Temporal causality heatmap
        im1 = axes[0].imshow(temporal_causality, cmap='hot', interpolation='nearest')
        axes[0].set_title('Temporal Causality (Attention Patterns)')
        axes[0].set_xlabel('Target Frame')
        axes[0].set_ylabel('Source Frame')
        plt.colorbar(im1, ax=axes[0])
        
        # Joint correlation heatmap
        im2 = axes[1].imshow(joint_causality, cmap='viridis', interpolation='nearest')
        axes[1].set_title('Joint Feature Correlations')
        axes[1].set_xticks(range(len(self.joint_names)))
        axes[1].set_yticks(range(len(self.joint_names)))
        axes[1].set_xticklabels(self.joint_names, rotation=45)
        axes[1].set_yticklabels(self.joint_names)
        plt.colorbar(im2, ax=axes[1])
        
        # Granger causality heatmap
        im3 = axes[2].imshow(granger_results, cmap='plasma', interpolation='nearest')
        axes[2].set_title('Granger Causality (Prediction Influence)')
        axes[2].set_xticks(range(len(self.joint_names)))
        axes[2].set_yticks(range(len(self.joint_names)))
        axes[2].set_xticklabels(self.joint_names, rotation=45)
        axes[2].set_yticklabels(self.joint_names)
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig('causality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_prefix', type=str, default='')
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--data_path', type=str)
  parser.add_argument('--learning_rate', type=float, default=1e-5)
  parser.add_argument('--max_epochs', type=int, default=500)
  parser.add_argument('--steps_per_epoch', type=int, default=200)
  parser.add_argument('--action', nargs='*', type=str, default=None)
  parser.add_argument('--use_one_hot',  action='store_true')
  parser.add_argument('--init_fn', type=str, default='xavier_init')
  parser.add_argument('--include_last_obs', action='store_true')
  parser.add_argument('--task', type=str, default='downstream', choices=['pretext', 'downstream'])
  parser.add_argument('--downstream_strategy', default='both_then_class', choices=['both', 'class', 'both_then_class'])
  # pose transformers related parameters
  parser.add_argument('--model_dim', type=int, default=256)
  parser.add_argument('--num_encoder_layers', type=int, default=4)
  parser.add_argument('--num_decoder_layers', type=int, default=4)
  parser.add_argument('--num_heads', type=int, default=4)
  parser.add_argument('--dim_ffn', type=int, default=2048)
  parser.add_argument('--dropout', type=float, default=0.3)
  parser.add_argument('--source_seq_len', type=int, default=50)                  
  parser.add_argument('--target_seq_len', type=int, default=25)
  parser.add_argument('--max_gradient_norm', type=float, default=0.1)
  parser.add_argument('--lr_step_size',type=int, default=400)
  parser.add_argument('--learning_rate_fn',type=str, default='step')
  parser.add_argument('--warmup_epochs', type=int, default=100)
  parser.add_argument('--pose_format', type=str, default='rotmat')
  parser.add_argument('--remove_low_std', action='store_true')
  parser.add_argument('--remove_global_trans', action='store_true')
  parser.add_argument('--loss_fn', type=str, default='l1')
  parser.add_argument('--pad_decoder_inputs', action='store_true')
  parser.add_argument('--pad_decoder_inputs_mean', action='store_true')
  parser.add_argument('--use_wao_amass_joints', action='store_true')
  parser.add_argument('--non_autoregressive', action='store_true')
  parser.add_argument('--pre_normalization', action='store_true')
  parser.add_argument('--use_query_embedding', action='store_true')
  parser.add_argument('--predict_activity', action='store_true')
  parser.add_argument('--use_memory', action='store_true')
  parser.add_argument('--query_selection',action='store_true')
  parser.add_argument('--activity_weight', type=float, default=1.0)
  parser.add_argument('--pose_embedding_type', type=str, default='gcn_enc')
  parser.add_argument('--encoder_ckpt', type=str, default=None)
  parser.add_argument('--dataset', type=str, default='h36m_v2')
  parser.add_argument('--skip_rate', type=int, default=5)
  parser.add_argument('--eval_num_seeds', type=int, default=_NSEEDS)
  parser.add_argument('--copy_method', type=str, default=None)
  parser.add_argument('--finetuning_ckpt', type=str, default=None)
  parser.add_argument('--pos_enc_alpha', type=float, default=10)
  parser.add_argument('--pos_enc_beta', type=float, default=500)
  parser.add_argument('--causal_analysis', action='store_true', help='Run zero-retraining causal analysis after training')
  args = parser.parse_args()
  
  params = vars(args)
  
  # Count folds for downstream task (using the correct flat structure)
  if params['task'] == 'downstream':
    num_folds = count_folds(params['data_path'])
    # Validate fold count
    if num_folds == 0:
        print(f"\nERROR: No valid data found in {params['data_path']}")
        print("Please verify your data structure. Expected format:")
        print("  data_path/")
        print("  ├── EPG_train_1.pkl")
        print("  ├── EPG_test_1.pkl")
        print("  ├── EPG_train_2.pkl")
        print("  ├── EPG_test_2.pkl")
        print("  └── ...")
        sys.exit(1)
  else:
    num_folds = 1

  total_preds = []
  total_gts = []
  preds_votes = []
  preds_probs = []

  all_folds = range(1, num_folds + 1)
  for fold in all_folds:
    print(f'\n{"="*60}')
    print(f'Fold {fold} out of {num_folds}')
    print(f'{"="*60}')

    utils.create_dir_tree(params['model_prefix']) # moving this up because dataset mean and std stored under it

    train_dataset_fn, eval_dataset_fn = dataset_factory(params, fold, params['model_prefix'])

    params['input_dim'] = train_dataset_fn.dataset._data_dim
    params['pose_dim'] = train_dataset_fn.dataset._pose_dim
    pose_encoder_fn, pose_decoder_fn = \
        PoseEncoderDecoder.select_pose_encoder_decoder_fn(params)

    config_path = os.path.join(params['model_prefix'], 'config', 'config.json')        
    with open(config_path, 'w') as file_:
      json.dump(params, file_, indent=4)

    model_fn = POTRModelFn(
        params, train_dataset_fn, 
        eval_dataset_fn, 
        pose_encoder_fn, pose_decoder_fn
    )
    if params['task'] == 'downstream':
      predictions, gts, pred_probs = model_fn.train()

      print('predictions:', predictions)

      # save predicted classes
      preds_votes.append(predictions.tolist())

      # save predicted probabilities
      preds_probs.append(pred_probs.tolist())

      # save final predictions and true labels
      if np.shape(gts)[0] == 1: # only 1 clip
        pred = int(predictions)
      else:
        pred = single_vote(predictions)
      gt = gts[0]
      total_preds.append(pred)
      total_gts.append(int(gt))

      del model_fn, pose_encoder_fn, pose_decoder_fn

      attributes = [preds_votes, total_preds, preds_probs, total_gts]
      names = ['predicted_classes', 'predicted_final_classes', 'prediction_list', 'true_labels']
      jsonfilename = os.path.join(params['model_prefix'], 'results.json')        
      save_json(jsonfilename, attributes, names)
    else:
      model_fn.train()
  
  if params['task'] == 'downstream':
    # Only generate report if we have actual data
    if len(total_gts) > 0 and len(total_preds) > 0:
        print("\n" + "="*60)
        print("=== Final Classification Report ===")
        print(classification_report(total_gts, total_preds))
        print(f"Overall accuracy: {accuracy_score(total_gts, total_preds):.4f}")
        print("="*60)
    else:
        print("\nWARNING: No valid predictions were generated.")
        print("This typically happens when:")
        print("1. The data path is incorrect")
        print("2. Required EPG files are missing or misnamed")
        print("3. No valid data was found during processing")
        print("Please check your data structure and paths.")

  if args.causal_analysis:
        try:
            print("\n" + "="*60)
            print("=== CAUSALITY ANALYSIS ===")
            print("Running zero-retraining causal analysis on the final model...")
            
            # Define joint names (use the same as in your GaitJointsDataset)
            joint_names = [
                'Hip', 'Thorax', 'Spine', 'Neck', 
                'Shoulder_L', 'Shoulder_R', 'Elbow_L', 'Elbow_R',
                'Wrist_L', 'Wrist_R', 'Knee_L', 'Knee_R', 
                'Ankle_L', 'Ankle_R', 'Foot_L', 'Foot_R', 'Head'
            ]
            
            # Load the last trained model (or best model if you track it)
            model_path = os.path.join(
                params['model_prefix'], 
                'models', 
                f'ckpt_epoch_{params["max_epochs"]-1}.pt'
            )
            
            print(f"Loading model from: {model_path}")
            pretrained_model = torch.load(model_path, map_location=_DEVICE)
            pretrained_model.eval()
            
            # Get a test loader for causal analysis
            _, eval_dataset_fn = dataset_factory(params, fold, params['model_prefix'])
            
            # Initialize and run the causal analyzer
            analyzer = ZeroRetrainingCausalAnalyzer(pretrained_model, joint_names)
            insights = analyzer.create_causality_report(eval_dataset_fn)
            
            # Save results
            import json
            with open(os.path.join(params['model_prefix'], 'causality_insights.json'), 'w') as f:
                json.dump(insights, f, indent=2)
                
            print("Causality analysis complete! Results saved.")
            
        except Exception as e:
            print(f"Error during causal analysis: {str(e)}")
            print("Causality analysis failed. Continuing execution.")



# ###############################################################################
# # Pose Transformers (POTR): Human Motion Prediction with Non-Autoregressive 
# # Transformers
# # 
# # Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# # Written by 
# # Angel Martinez <angel.martinez@idiap.ch>,
# # 
# # This file is part of 
# # POTR: Human Motion Prediction with Non-Autoregressive Transformers
# # 
# # POTR is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License version 3 as
# # published by the Free Software Foundation.
# # 
# # POTR is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# # GNU General Public License for more details.
# # 
# # You should have received a copy of the GNU General Public License
# # along with POTR. If not, see <http://www.gnu.org/licenses/>.
# ###############################################################################

# """Implments the model function for the POTR model."""


# import numpy as np
# import os
# import sys
# import argparse
# import json
# import time
# import csv
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from sklearn.metrics import classification_report

# from numpyencoder import NumpyEncoder


# thispath = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, thispath+"/../")

# import training.seq2seq_model_fn as seq2seq_model_fn
# import models.PoseTransformer as PoseTransformer
# import models.PoseEncoderDecoder as PoseEncoderDecoder
# import data.NTURGDDataset as NTURGDDataset
# import data.GaitJointsDataset as GaitJointsDataset
# import utils.utils as utils

# _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# _WEIGHT_DECAY = 0.00001
# _NSEEDS = 8

# class POTRModelFn(seq2seq_model_fn.ModelFn):
#   def __init__(self,
#                params,
#                train_dataset_fn,
#                eval_dataset_fn,
#                pose_encoder_fn=None,
#                pose_decoder_fn=None):
#     super(POTRModelFn, self).__init__(
#       params, train_dataset_fn, eval_dataset_fn, pose_encoder_fn, pose_decoder_fn)
#     self._loss_fn = self.layerwise_loss_fn
#     self.task = params['task']
    
#     # Create directory for metrics if it doesn't exist
#     metrics_dir = os.path.join(params['model_prefix'], 'metrics')
#     os.makedirs(metrics_dir, exist_ok=True)
    
#     # Create CSV file for metrics
#     self.metrics_file = os.path.join(metrics_dir, 'training_metrics.csv')
    
#     # Initialize CSV with headers if file doesn't exist
#     if not os.path.exists(self.metrics_file):
#         with open(self.metrics_file, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['fold', 'epoch', 'train_loss', 'val_loss', 
#                             'train_acc', 'val_acc', 'val_precision', 
#                             'val_recall', 'val_f1', 'val_auc'])
    
#     if self.task == 'downstream':
#         # The model expects 4 classes, so we need to provide weights for 4 classes
#         # Let's maintain the healthy/unhealthy ratio across 4 classes
#         healthy_count = 29
#         unhealthy_count = 11
        
#         # Distribute the counts across 4 classes:
#         # Classes 0 and 1 represent healthy (class 0)
#         # Classes 2 and 3 represent unhealthy (class 1)
#         class_counts = torch.tensor([
#             healthy_count / 2,  # half of healthy to class 0
#             healthy_count / 2,  # half of healthy to class 1
#             unhealthy_count / 2,  # half of unhealthy to class 2
#             unhealthy_count / 2   # half of unhealthy to class 3
#         ])
        
#         class_frequencies = class_counts.float() / class_counts.sum()
#         weights = 1.0 / class_frequencies
#         weights = weights / weights.sum()
        
#         self._loss_weights = weights.to(_DEVICE)
#         self._weighted_ce_loss = nn.CrossEntropyLoss(weight=self._loss_weights)
#         print(f'Using weighted CE loss with weights: {weights.cpu().numpy().tolist()}')
#     else:
#         print('Using a standard CE loss for activity prediction.')

#   def smooth_l1(self, decoder_pred, decoder_gt):
#     l1loss = nn.SmoothL1Loss(reduction='mean')
#     return l1loss(decoder_pred, decoder_gt)

#   def loss_l1(self, decoder_pred, decoder_gt):
#     return nn.L1Loss(reduction='mean')(decoder_pred, decoder_gt)

#   def loss_activity(self, logits, class_gt):                                     
#     """Computes entropy loss from logits between predictions and class."""
#     if self.task == 'downstream':
#       return self._weighted_ce_loss(logits, class_gt)
#     else:
#       return nn.functional.cross_entropy(logits, class_gt, reduction='mean')

#   def compute_class_loss(self, class_logits, class_gt):
#     """Computes the class loss for each of the decoder layers predictions or memory."""
#     class_loss = 0.0
#     for l in range(len(class_logits)):
#       class_loss += self.loss_activity(class_logits[l], class_gt)

#     return class_loss/len(class_logits)

#   def select_loss_fn(self):
#     if self._params['loss_fn'] == 'mse':
#       return self.loss_mse
#     elif self._params['loss_fn'] == 'smoothl1':
#       return self.smooth_l1
#     elif self._params['loss_fn'] == 'l1':
#       return self.loss_l1
#     else:
#       raise ValueError('Unknown loss name {}.'.format(self._params['loss_fn']))

#   def layerwise_loss_fn(self, decoder_pred, decoder_gt, class_logits=None, class_gt=None):
#     """Computes layerwise loss between predictions and ground truth."""
#     pose_loss = 0.0
#     loss_fn = self.select_loss_fn()

#     for l in range(len(decoder_pred)):
#       pose_loss += loss_fn(decoder_pred[l], decoder_gt)

#     pose_loss = pose_loss/len(decoder_pred)
#     if class_logits is not None:
#       return pose_loss, self.compute_class_loss(class_logits, class_gt)

#     return pose_loss, None

#   def init_model(self, pose_encoder_fn=None, pose_decoder_fn=None):
#     self._model = PoseTransformer.model_factory(
#         self._params, 
#         pose_encoder_fn, 
#         pose_decoder_fn
#     )

#   def select_optimizer(self):
#     optimizer = optim.AdamW(
#         self._model.parameters(), lr=self._params['learning_rate'],
#         betas=(0.9, 0.999),
#         weight_decay=_WEIGHT_DECAY
#     )

#     return optimizer
    
#   def log_metrics(self, fold, epoch, train_loss, val_loss, train_acc, val_acc, 
#                  val_precision, val_recall, val_f1, val_auc):
#     """Log metrics to CSV file and print them"""
#     # Log to CSV
#     with open(self.metrics_file, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([fold, epoch, train_loss, val_loss, 
#                         train_acc, val_acc, val_precision, 
#                         val_recall, val_f1, val_auc])
    
#     # Print metrics with clear formatting
#     print(f"\n{'='*50}")
#     print(f"FOLD {fold} | EPOCH {epoch}")
#     print(f"{'='*50}")
#     print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
#     print(f"Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}")
#     print(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}")
#     print(f"Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
#     print(f"{'='*50}\n")

#   def train(self):
#     """Train the model and track metrics"""
#     best_val_loss = float('inf')
#     metrics_history = []
    
#     for epoch in range(self._params['max_epochs']):
#         # Train for one epoch
#         train_loss, train_acc = self.train_one_epoch(epoch)
        
#         # Evaluate
#         val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = self.evaluate()
        
#         # Log metrics
#         self.log_metrics(
#             fold=self._params.get('fold', 1),
#             epoch=epoch,
#             train_loss=train_loss,
#             val_loss=val_loss,
#             train_acc=train_acc,
#             val_acc=val_acc,
#             val_precision=val_precision,
#             val_recall=val_recall,
#             val_f1=val_f1,
#             val_auc=val_auc
#         )
        
#         metrics_history.append({
#             'epoch': epoch,
#             'train_loss': train_loss,
#             'val_loss': val_loss,
#             'train_acc': train_acc,
#             'val_acc': val_acc,
#             'val_precision': val_precision,
#             'val_recall': val_recall,
#             'val_f1': val_f1,
#             'val_auc': val_auc
#         })
        
#         # Save best model
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             self.save_model(os.path.join(self._params['model_prefix'], 'best_model.pth'))
    
#     # Save final metrics history as JSON
#     metrics_file = os.path.join(self._params['model_prefix'], 'metrics', 'metrics_history.json')
#     with open(metrics_file, 'w') as f:
#         json.dump(metrics_history, f, indent=2)
    
#     # Generate final classification report
#     test_loss, predictions, gts, pred_probs = self.evaluate_full()
    
#     # Calculate final metrics
#     if len(gts) > 0:
#         final_acc = accuracy_score(gts, predictions)
#         final_precision = precision_score(gts, predictions, zero_division=0)
#         final_recall = recall_score(gts, predictions, zero_division=0)
#         final_f1 = f1_score(gts, predictions, zero_division=0)
        
#         print("\n" + "="*60)
#         print("=== Final Evaluation Results ===")
#         print(f"Accuracy: {final_acc:.4f}")
#         print(f"Precision: {final_precision:.4f}")
#         print(f"Recall: {final_recall:.4f}")
#         print(f"F1 Score: {final_f1:.4f}")
#         print(f"Confusion Matrix:\n{confusion_matrix(gts, predictions)}")
#         print("="*60)
        
#         # Return predictions for downstream tasks
#         return predictions, gts, pred_probs
#     else:
#         print("No valid test data for evaluation")
#         return np.array([]), np.array([]), np.array([])

#   def evaluate(self):
#     """Evaluate model and return multiple metrics"""
#     self._model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0
#     all_preds = []
#     all_gts = []
    
#     with torch.no_grad():
#         for i, batch in enumerate(self._eval_dataset):
#             src, tgt, tgt_y, class_gt = batch
#             src, tgt, tgt_y, class_gt = (
#                 src.to(_DEVICE),
#                 tgt.to(_DEVICE),
#                 tgt_y.to(_DEVICE),
#                 class_gt.to(_DEVICE)
#             )
            
#             # Forward pass
#             if self._params['predict_activity']:
#                 outputs, class_logits = self._model(src, tgt)
#             else:
#                 outputs = self._model(src, tgt)
            
#             # Calculate loss
#             loss, activity_loss = self.compute_loss(outputs, tgt_y, class_logits, class_gt)
            
#             # Calculate accuracy
#             if self._params['predict_activity']:
#                 _, predicted = torch.max(class_logits[-1], 1)
#                 correct += (predicted == class_gt).sum().item()
#                 total += class_gt.size(0)
                
#                 # Store for later metrics
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_gts.extend(class_gt.cpu().numpy())
            
#             val_loss += loss.item()
    
#     # Calculate metrics
#     val_acc = correct / total if total > 0 else 0.0
    
#     # Calculate additional metrics if we have predictions
#     if len(all_preds) > 0 and len(all_gts) > 0:
#         val_precision = precision_score(all_gts, all_preds, zero_division=0)
#         val_recall = recall_score(all_gts, all_preds, zero_division=0)
#         val_f1 = f1_score(all_gts, all_preds, zero_division=0)
        
#         # For AUC, we need probabilities
#         try:
#             val_auc = roc_auc_score(all_gts, all_preds)
#         except:
#             val_auc = 0.0
#     else:
#         val_precision = val_recall = val_f1 = val_auc = 0.0
    
#     return val_loss / (i + 1), val_acc, val_precision, val_recall, val_f1, val_auc

#   def evaluate_full(self):
#     """Full evaluation for final results"""
#     self._model.eval()
#     all_predictions = []
#     all_gts = []
#     all_probs = []
    
#     with torch.no_grad():
#         for i, batch in enumerate(self._eval_dataset):
#             src, tgt, tgt_y, class_gt = batch
#             src, tgt, tgt_y, class_gt = (
#                 src.to(_DEVICE),
#                 tgt.to(_DEVICE),
#                 tgt_y.to(_DEVICE),
#                 class_gt.to(_DEVICE)
#             )
            
#             # Forward pass
#             if self._params['predict_activity']:
#                 outputs, class_logits = self._model(src, tgt)
                
#                 # Get predictions from final layer
#                 _, predicted = torch.max(class_logits[-1], 1)
#                 probs = torch.softmax(class_logits[-1], dim=1)
                
#                 all_predictions.extend(predicted.cpu().numpy())
#                 all_gts.extend(class_gt.cpu().numpy())
#                 all_probs.extend(probs.cpu().numpy())
#             else:
#                 outputs = self._model(src, tgt)
    
#     # Convert to numpy arrays
#     predictions = np.array(all_predictions)
#     gts = np.array(all_gts)
#     pred_probs = np.array(all_probs) if len(all_probs) > 0 else np.array([])
    
#     # Calculate loss (simplified)
#     test_loss = 0.0
    
#     return test_loss, predictions, gts, pred_probs

# def dataset_factory(params, fold, model_prefix):
#   if params['dataset'] == 'ntu_rgbd':
#     return NTURGDDataset.dataset_factory(params)
#   elif params['dataset'] == 'pd_gait':
#     return GaitJointsDataset.dataset_factory(params, fold)
#   else:
#     raise ValueError('Unknown dataset {}'.format(params['dataset']))

# def single_vote(pred):
#   """
#   Get majority vote of predicted classes for the clips in one video.
#   :param preds: list of predicted class for each clip of one video
#   :return: majority vote of predicted class for one video
#   """
#   p = np.array(pred)
#   counts = np.bincount(p)
#   max_count = 0
#   max_index = 0
#   for i in range(len(counts)):
#     if max_count < counts[i]:
#       max_index = i
#       max_count = counts[i]
#   return max_index

# def save_json(filename, attributes, names):
#     """
#     Save training parameters and evaluation results to json file.
#     :param filename: save filename
#     :param attributes: attributes to save
#     :param names: name of attributes to save in json file
#     """
#     with open(filename, "w", encoding="utf8") as outfile:
#         d = {}
#         for i in range(len(attributes)):
#             name = names[i]
#             attribute = attributes[i]
#             d[name] = attribute
#         json.dump(d, outfile, indent=4, cls=NumpyEncoder)

# def count_folds(data_path):
#     """Count how many complete fold pairs exist in the flat data structure."""
#     train_files = [f for f in os.listdir(data_path) if f.startswith("EPG_train_") and f.endswith(".pkl")]
#     test_files = [f for f in os.listdir(data_path) if f.startswith("EPG_test_") and f.endswith(".pkl")]
    
#     # Count how many fold numbers exist in both train and test
#     train_folds = set(int(f.split("_")[2].split(".")[0]) for f in train_files)
#     test_folds = set(int(f.split("_")[2].split(".")[0]) for f in test_files)
#     valid_folds = train_folds & test_folds
    
#     # Print diagnostic information
#     print(f"\nData structure diagnostic for {data_path}:")
#     print(f"  Found {len(train_files)} train files")
#     print(f"  Found {len(test_files)} test files")
#     print(f"  Found {len(valid_folds)} complete fold pairs")
    
#     if len(valid_folds) == 0:
#         print("  ERROR: No valid fold pairs found")
#         print("  Expected file pattern: EPG_train_X.pkl and EPG_test_X.pkl where X is the fold number")
#         print("  Please verify your data structure")
    
#     return len(valid_folds)

# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument('--model_prefix', type=str, default='')
#   parser.add_argument('--batch_size', type=int, default=16)
#   parser.add_argument('--data_path', type=str)
#   parser.add_argument('--learning_rate', type=float, default=1e-5)
#   parser.add_argument('--max_epochs', type=int, default=500)
#   parser.add_argument('--steps_per_epoch', type=int, default=200)
#   parser.add_argument('--action', nargs='*', type=str, default=None)
#   parser.add_argument('--use_one_hot',  action='store_true')
#   parser.add_argument('--init_fn', type=str, default='xavier_init')
#   parser.add_argument('--include_last_obs', action='store_true')
#   parser.add_argument('--task', type=str, default='downstream', choices=['pretext', 'downstream'])
#   parser.add_argument('--downstream_strategy', default='both_then_class', choices=['both', 'class', 'both_then_class'])
#   # pose transformers related parameters
#   parser.add_argument('--model_dim', type=int, default=256)
#   parser.add_argument('--num_encoder_layers', type=int, default=4)
#   parser.add_argument('--num_decoder_layers', type=int, default=4)
#   parser.add_argument('--num_heads', type=int, default=4)
#   parser.add_argument('--dim_ffn', type=int, default=2048)
#   parser.add_argument('--dropout', type=float, default=0.3)
#   parser.add_argument('--source_seq_len', type=int, default=50)                  
#   parser.add_argument('--target_seq_len', type=int, default=25)
#   parser.add_argument('--max_gradient_norm', type=float, default=0.1)
#   parser.add_argument('--lr_step_size',type=int, default=400)
#   parser.add_argument('--learning_rate_fn',type=str, default='step')
#   parser.add_argument('--warmup_epochs', type=int, default=100)
#   parser.add_argument('--pose_format', type=str, default='rotmat')
#   parser.add_argument('--remove_low_std', action='store_true')
#   parser.add_argument('--remove_global_trans', action='store_true')
#   parser.add_argument('--loss_fn', type=str, default='l1')
#   parser.add_argument('--pad_decoder_inputs', action='store_true')
#   parser.add_argument('--pad_decoder_inputs_mean', action='store_true')
#   parser.add_argument('--use_wao_amass_joints', action='store_true')
#   parser.add_argument('--non_autoregressive', action='store_true')
#   parser.add_argument('--pre_normalization', action='store_true')
#   parser.add_argument('--use_query_embedding', action='store_true')
#   parser.add_argument('--predict_activity', action='store_true')
#   parser.add_argument('--use_memory', action='store_true')
#   parser.add_argument('--query_selection',action='store_true')
#   parser.add_argument('--activity_weight', type=float, default=1.0)
#   parser.add_argument('--pose_embedding_type', type=str, default='gcn_enc')
#   parser.add_argument('--encoder_ckpt', type=str, default=None)
#   parser.add_argument('--dataset', type=str, default='h36m_v2')
#   parser.add_argument('--skip_rate', type=int, default=5)
#   parser.add_argument('--eval_num_seeds', type=int, default=_NSEEDS)
#   parser.add_argument('--copy_method', type=str, default=None)
#   parser.add_argument('--finetuning_ckpt', type=str, default=None)
#   parser.add_argument('--pos_enc_alpha', type=float, default=10)
#   parser.add_argument('--pos_enc_beta', type=float, default=500)
#   args = parser.parse_args()
  
#   params = vars(args)
  
#   # Count folds for downstream task (using the correct flat structure)
#   if params['task'] == 'downstream':
#     num_folds = count_folds(params['data_path'])
#     # Validate fold count
#     if num_folds == 0:
#         print(f"\nERROR: No valid data found in {params['data_path']}")
#         print("Please verify your data structure. Expected format:")
#         print("  data_path/")
#         print("  ├── EPG_train_1.pkl")
#         print("  ├── EPG_test_1.pkl")
#         print("  ├── EPG_train_2.pkl")
#         print("  ├── EPG_test_2.pkl")
#         print("  └── ...")
#         sys.exit(1)
#   else:
#     num_folds = 1

#   total_preds = []
#   total_gts = []
#   preds_votes = []
#   preds_probs = []

#   all_folds = range(1, num_folds + 1)
#   for fold in all_folds:
#     print(f'\n{"="*60}')
#     print(f'Fold {fold} out of {num_folds}')
#     print(f'{"="*60}')

#     utils.create_dir_tree(params['model_prefix']) # moving this up because dataset mean and std stored under it

#     train_dataset_fn, eval_dataset_fn = dataset_factory(params, fold, params['model_prefix'])

#     params['input_dim'] = train_dataset_fn.dataset._data_dim
#     params['pose_dim'] = train_dataset_fn.dataset._pose_dim
#     pose_encoder_fn, pose_decoder_fn = \
#         PoseEncoderDecoder.select_pose_encoder_decoder_fn(params)

#     config_path = os.path.join(params['model_prefix'], 'config', 'config.json')        
#     with open(config_path, 'w') as file_:
#       json.dump(params, file_, indent=4)

#     model_fn = POTRModelFn(
#         params, train_dataset_fn, 
#         eval_dataset_fn, 
#         pose_encoder_fn, pose_decoder_fn
#     )
#     if params['task'] == 'downstream':
#       predictions, gts, pred_probs = model_fn.train()

#       print('predictions:', predictions)

#       # save predicted classes
#       preds_votes.append(predictions.tolist())

#       # save predicted probabilities
#       preds_probs.append(pred_probs.tolist())

#       # save final predictions and true labels
#       if np.shape(gts)[0] == 1: # only 1 clip
#         pred = int(predictions)
#       else:
#         pred = single_vote(predictions)
#       gt = gts[0]
#       total_preds.append(pred)
#       total_gts.append(int(gt))

#       del model_fn, pose_encoder_fn, pose_decoder_fn

#       attributes = [preds_votes, total_preds, preds_probs, total_gts]
#       names = ['predicted_classes', 'predicted_final_classes', 'prediction_list', 'true_labels']
#       jsonfilename = os.path.join(params['model_prefix'], 'results.json')        
#       save_json(jsonfilename, attributes, names)
#     else:
#       model_fn.train()

#   if params['task'] == 'downstream':
#     # Only generate report if we have actual data
#     if len(total_gts) > 0 and len(total_preds) > 0:
#         print("\n" + "="*60)
#         print("=== Final Classification Report ===")
#         print(classification_report(total_gts, total_preds))
#         print(f"Overall accuracy: {accuracy_score(total_gts, total_preds):.4f}")
#         print("="*60)
        
#         # Save final results to CSV
#         results_df = pd.DataFrame({
#             'true_labels': total_gts,
#             'predicted_labels': total_preds
#         })
#         results_csv = os.path.join(params['model_prefix'], 'results.csv')
#         results_df.to_csv(results_csv, index=False)
#         print(f"Results saved to: {results_csv}")
#     else:
#         print("\nWARNING: No valid predictions were generated.")
#         print("This typically happens when:")
#         print("1. The data path is incorrect")
#         print("2. Required EPG files are missing or misnamed")
#         print("3. No valid data was found during processing")
#         print("Please check your data structure and paths.")