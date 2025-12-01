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
import csv  # NEW: Added for CSV logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from datetime import datetime

from numpyencoder import NumpyEncoder


thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import training.seq2seq_model_fn as seq2seq_model_fn
import models.PoseTransformer as PoseTransformer
import models.PoseEncoderDecoder as PoseEncoderDecoder
import data.NTURGDDataset as NTURGDDataset
import data.GaitJointsDataset as GaitJointsDataset
import utils.utils as utils
import torch.nn.functional as F


_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_WEIGHT_DECAY = 0.00001
_NSEEDS = 8

def focal_loss(logits, targets, alpha=None, gamma=3):
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
    return ((1-pt)**gamma * ce).mean()

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
        
        # NEW: CSV logging setup
        self.csv_logger = CSVMetricsLogger(params['model_prefix'])
        
        # CRITICAL: Set correct number of classes based on your actual problem
        # This should match what's in your model configuration
        self.num_classes = params.get('num_classes', 4)  # Set to 2 for your binary classification
        
        if self.task == 'downstream':
            # We'll compute actual class weights per-fold during training
            self._loss_weights = None
            self._focal_loss_alpha = None  # Will be set per-fold
            self._focal_loss_gamma = 3.0
            
            print(f'Focal loss initialized for {self.num_classes} classes - weights will be computed per-fold with gamma={self._focal_loss_gamma}')
        else:
            print('Using a standard CE loss for activity prediction.')
            
    def update_fold_weights(self, train_loader):
        """Compute class weights from actual fold data distribution"""
        if self.task != 'downstream':
            return
        
        # Initialize counters for the correct number of classes
        class_counts = torch.zeros(self.num_classes, dtype=torch.float32)
        
        # Process ALL batches to get accurate class distribution
        total_samples = 0
        for batch in train_loader:
            if 'action_ids' not in batch:
                continue
                
            labels = batch['action_ids']
            # Convert to long and flatten all dimensions
            labels = labels.long().flatten()
            
            # Filter out any invalid class indices
            valid_mask = (labels >= 0) & (labels < self.num_classes)
            valid_labels = labels[valid_mask]
            
            if len(valid_labels) == 0:
                continue
                
            # Count classes in this batch
            batch_counts = torch.bincount(valid_labels, minlength=self.num_classes)
            class_counts += batch_counts
            total_samples += len(valid_labels)

        # Validate we have data
        if total_samples == 0:
            raise ValueError("No valid samples found in training data for class weight calculation")
        
        # Calculate class distribution percentages
        class_percentages = (class_counts / total_samples) * 100
        
        # Calculate imbalance ratio (larger class / smaller class)
        min_count = torch.min(class_counts)
        max_count = torch.max(class_counts)
        imbalance_ratio = max_count / (min_count + 1e-6)
        # Prevent division by zero for missing classes
        class_counts = torch.clamp(class_counts, min=1)
        
        # Compute normalized inverse frequency weights
        freq = class_counts / class_counts.sum()
        weights = 1.0 / (freq + 1e-6)
        weights = weights / weights.sum()  # Normalize weights
        
        # Update model weights
        self._focal_loss_alpha = weights.to(_DEVICE)
        self._loss_weights = weights.to(_DEVICE)
        
        # Log informative results
        print(f"📊 Class distribution: Class 0: {class_counts[0]:.0f} ({class_percentages[0]:.1f}%), Class 1: {class_counts[1]:.0f} ({class_percentages[1]:.1f}%)")
        print(f"⚠️  Imbalance ratio: {imbalance_ratio:.1f}:1 (larger:smaller)")
        print(f"⚖️  Computed focal loss weights: Class 0: {weights[0]:.4f}, Class 1: {weights[1]:.4f}")


    # The rest of your methods remain the same...
    def smooth_l1(self, decoder_pred, decoder_gt):
        l1loss = nn.SmoothL1Loss(reduction='mean')
        return l1loss(decoder_pred, decoder_gt)

    def loss_l1(self, decoder_pred, decoder_gt):
        return nn.L1Loss(reduction='mean')(decoder_pred, decoder_gt)


    def compute_class_loss(self, class_logits, class_gt):
        """Computes the class loss for each of the decoder layers predictions or memory."""
        class_loss = 0.0
        for l in range(len(class_logits)):
            class_loss += self.loss_activity(class_logits[l], class_gt)

        return class_loss/len(class_logits)
    def loss_activity(self, logits, class_gt):                                     
        """Computes entropy loss from logits between predictions and class."""
        class_gt = class_gt.long()
        
        # CRITICAL: Ensure class_gt values are within valid range
        class_gt = torch.clamp(class_gt, 0, self.num_classes-1)
        
        if self.task == 'downstream':
            # Create weight tensor with correct number of classes
            if self._focal_loss_alpha is not None:
                # Make sure weight tensor has the correct size
                if len(self._focal_loss_alpha) != self.num_classes:
                    # This is the critical fix - ensure weights match expected class count
                    print(f"⚠️ Weight tensor size mismatch: expected {self.num_classes} classes, got {len(self._focal_loss_alpha)}")
                    # Recreate weights with correct size
                    weights = torch.ones(self.num_classes, device=_DEVICE)
                    actual_classes = min(len(self._focal_loss_alpha), self.num_classes)
                    weights[:actual_classes] = self._focal_loss_alpha[:actual_classes]
                    weights = weights / weights.sum()
                    self._focal_loss_alpha = weights
                    print(f"✅ Fixed weight tensor to match {self.num_classes} classes")
                
                return focal_loss(logits, class_gt, alpha=self._focal_loss_alpha, gamma=self._focal_loss_gamma)
            return focal_loss(logits, class_gt, gamma=self._focal_loss_gamma)
        else:
            return nn.functional.cross_entropy(logits, class_gt, reduction='mean')
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
            # For imbalanced datasets, consider these optimizers:
            
            # Option 1: AdamW with adjusted parameters (recommended)
            # optimizer = optim.AdamW(
            #     self._model.parameters(), 
            #     lr=self._params['learning_rate'],
            #     betas=(0.9, 0.999),
            #     weight_decay=_WEIGHT_DECAY
            # )
            
            # Option 2: SGD with momentum (sometimes better for imbalanced data)
            optimizer = optim.SGD(
                self._model.parameters(),
                lr=self._params['learning_rate'],
                momentum=0.9,
                weight_decay=_WEIGHT_DECAY,
                nesterov=True
            )
            
            # Option 3: Adam with AMSGrad (more stable for imbalanced data)
            # optimizer = optim.Adam(
            #     self._model.parameters(),
            #     lr=self._params['learning_rate'],
            #     betas=(0.9, 0.999),
            #     weight_decay=_WEIGHT_DECAY,
            #     amsgrad=True
            # )
            
            return optimizer

    # NEW: Override train method to add CSV logging
    def train(self):
        """Override train method to add CSV logging for epochs."""
        start_time = time.time()
        
        # Call parent train method
        if self._params['task'] == 'downstream':
            predictions, gts, pred_probs = super().train()
            
            # Log final fold metrics
            if hasattr(self, 'current_fold'):
                self.csv_logger.log_fold_metrics(self.current_fold, predictions, gts)
            
            return predictions, gts, pred_probs
        else:
            super().train()
            return None, None, None

    # NEW: Method to log epoch metrics
    def log_epoch_metrics(self, epoch, train_loss, eval_loss, train_pose_loss=None, 
                         eval_pose_loss=None, train_activity_loss=None, eval_activity_loss=None,
                         learning_rate=None):
        """Log metrics for each epoch to CSV."""
        self.csv_logger.log_epoch_metrics(
            epoch, train_loss, eval_loss, train_pose_loss, eval_pose_loss,
            train_activity_loss, eval_activity_loss, learning_rate
        )

class CSVMetricsLogger:
    """CSV logger for tracking metrics during training across epochs and folds."""
    
    def __init__(self, model_prefix):
        self.model_prefix = model_prefix
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create logs directory
        self.logs_dir = os.path.join(model_prefix, 'training_logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize CSV files
        self.epochs_csv_path = os.path.join(self.logs_dir, f'epoch_metrics_{self.timestamp}.csv')
        self.folds_csv_path = os.path.join(self.logs_dir, f'fold_metrics_{self.timestamp}.csv')
        self.final_metrics_csv_path = os.path.join(self.logs_dir, f'final_metrics_{self.timestamp}.csv')
        
        # Initialize epoch metrics CSV
        with open(self.epochs_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'fold', 'epoch', 'train_loss', 'eval_loss',
                'train_pose_loss', 'eval_pose_loss', 'train_activity_loss', 
                'eval_activity_loss', 'learning_rate', 'epoch_time'
            ])
        
        # Initialize fold metrics CSV
        with open(self.folds_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'fold', 'accuracy', 'precision_macro', 'recall_macro', 
                'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted',
                'num_samples', 'fold_time'
            ])
        
        # Initialize final metrics CSV
        with open(self.final_metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'total_folds', 'overall_accuracy', 'overall_precision_macro',
                'overall_recall_macro', 'overall_f1_macro', 'overall_precision_weighted',
                'overall_recall_weighted', 'overall_f1_weighted', 'total_training_time'
            ])
        
        print(f"CSV logging initialized:")
        print(f"  Epoch metrics: {self.epochs_csv_path}")
        print(f"  Fold metrics: {self.folds_csv_path}")
        print(f"  Final metrics: {self.final_metrics_csv_path}")
    
    def log_epoch_metrics(self, epoch, train_loss, eval_loss, train_pose_loss=None,
                         eval_pose_loss=None, train_activity_loss=None, 
                         eval_activity_loss=None, learning_rate=None):
        """Log metrics for a single epoch."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.epochs_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                getattr(self, 'current_fold', 1),
                epoch,
                f"{train_loss:.6f}" if train_loss is not None else "N/A",
                f"{eval_loss:.6f}" if eval_loss is not None else "N/A",
                f"{train_pose_loss:.6f}" if train_pose_loss is not None else "N/A",
                f"{eval_pose_loss:.6f}" if eval_pose_loss is not None else "N/A",
                f"{train_activity_loss:.6f}" if train_activity_loss is not None else "N/A",
                f"{eval_activity_loss:.6f}" if eval_activity_loss is not None else "N/A",
                f"{learning_rate:.8f}" if learning_rate is not None else "N/A",
                datetime.now().strftime("%H:%M:%S")
            ])
    
    def log_fold_metrics(self, fold, predictions, true_labels):
        """Log metrics for a completed fold."""
        if predictions is None or true_labels is None:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=range(self.model_fn.num_classes) if hasattr(self.model_fn, 'num_classes') else None)
        recall_macro = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
        precision_weighted = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        with open(self.folds_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                fold,
                f"{accuracy:.4f}",
                f"{precision_macro:.4f}",
                f"{recall_macro:.4f}",
                f"{f1_macro:.4f}",
                f"{precision_weighted:.4f}",
                f"{recall_weighted:.4f}",
                f"{f1_weighted:.4f}",
                len(true_labels),
                datetime.now().strftime("%H:%M:%S")
            ])
    
    def log_final_metrics(self, total_folds, overall_metrics, total_training_time):
        """Log final overall metrics after all folds."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.final_metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                total_folds,
                f"{overall_metrics.get('accuracy', 0):.4f}",
                f"{overall_metrics.get('precision_macro', 0):.4f}",
                f"{overall_metrics.get('recall_macro', 0):.4f}",
                f"{overall_metrics.get('f1_macro', 0):.4f}",
                f"{overall_metrics.get('precision_weighted', 0):.4f}",
                f"{overall_metrics.get('recall_weighted', 0):.4f}",
                f"{overall_metrics.get('f1_weighted', 0):.4f}",
                f"{total_training_time:.2f}"
            ])


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

# def compute_comprehensive_metrics(y_true, y_pred, num_classes):
#     """
#     Compute comprehensive classification metrics including per-class and overall metrics.
#     """
#     metrics = {}
    
#     # Basic metrics
#     metrics['accuracy'] = accuracy_score(y_true, y_pred)
#     metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
#     metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
#     metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
#     metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
#     metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
#     metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
#     # Per-class metrics
#     if num_classes > 1:
#         metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
#         metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
#         metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
#     # Confusion matrix
#     metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
#     # Class distribution
#     metrics['true_class_distribution'] = np.bincount(y_true, minlength=num_classes).tolist()
#     metrics['pred_class_distribution'] = np.bincount(y_pred, minlength=num_classes).tolist()
    
#     return metrics
def compute_comprehensive_metrics(y_true, y_pred, num_classes):
    """
    Compute comprehensive classification metrics including per-class and overall metrics.
    """
    metrics = {}
    
    # Get actual number of classes from data
    actual_num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    if actual_num_classes > 1:
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # Class distribution
    metrics['true_class_distribution'] = np.bincount(y_true, minlength=actual_num_classes).tolist()
    metrics['pred_class_distribution'] = np.bincount(y_pred, minlength=actual_num_classes).tolist()
    
    return metrics
def print_detailed_report(y_true, y_pred, num_classes):
    """
    Print a detailed classification report with comprehensive metrics.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE CLASSIFICATION REPORT")
    print("="*80)
    
    # Get the actual number of classes from the data
    actual_num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision_macro:.4f} (macro)")
    print(f"  Recall:      {recall_macro:.4f} (macro)")
    print(f"  F1-Score:    {f1_macro:.4f} (macro)")
    
    # Weighted metrics
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"  Precision:   {precision_weighted:.4f} (weighted)")
    print(f"  Recall:      {recall_weighted:.4f} (weighted)")
    print(f"  F1-Score:    {f1_weighted:.4f} (weighted)")
    
    # Per-class metrics - use actual_num_classes instead of num_classes parameter
    if actual_num_classes > 1:
        print(f"\nPer-Class Metrics:")
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i in range(actual_num_classes):  # Use actual_num_classes here
            print(f"  Class {i}: Precision={precision_per_class[i]:.4f}, "
                  f"Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(" " * 8 + "".join([f"Pred {i:>6}" for i in range(actual_num_classes)]))  # Use actual_num_classes here
    for i in range(actual_num_classes):  # Use actual_num_classes here
        if i == 0:
            print(f"True {i}  " + " ".join([f"{val:6d}" for val in cm[i]]))
        else:
            print(f"     {i}  " + " ".join([f"{val:6d}" for val in cm[i]]))
    
    # Class distribution
    actual_num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    true_dist = np.bincount(y_true, minlength=actual_num_classes)
    pred_dist = np.bincount(y_pred, minlength=actual_num_classes)
    
    print(f"\nClass Distribution:")
    print(f"  True: {dict(enumerate(true_dist))}")
    print(f"  Pred: {dict(enumerate(pred_dist))}")
    
    print("="*80)
    
    return actual_num_classes  # Return the actual number of classes found
# def print_detailed_report(y_true, y_pred, num_classes):
#     """
#     Print a detailed classification report with comprehensive metrics.
#     """
#     print("\n" + "="*80)
#     print("COMPREHENSIVE CLASSIFICATION REPORT")
#     print("="*80)
    
#     # Basic metrics
#     accuracy = accuracy_score(y_true, y_pred)
#     precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
#     recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
#     f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
#     print(f"\nOverall Metrics:")
#     print(f"  Accuracy:    {accuracy:.4f}")
#     print(f"  Precision:   {precision_macro:.4f} (macro)")
#     print(f"  Recall:      {recall_macro:.4f} (macro)")
#     print(f"  F1-Score:    {f1_macro:.4f} (macro)")
    
#     # Weighted metrics
#     precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
#     recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
#     f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
#     print(f"  Precision:   {precision_weighted:.4f} (weighted)")
#     print(f"  Recall:      {recall_weighted:.4f} (weighted)")
#     print(f"  F1-Score:    {f1_weighted:.4f} (weighted)")
    
#     # Per-class metrics
#     if num_classes > 1:
#         print(f"\nPer-Class Metrics:")
#         precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
#         recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
#         f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
#         for i in range(num_classes):
#             print(f"  Class {i}: Precision={precision_per_class[i]:.4f}, "
#                   f"Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
    
#     # Confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     print(f"\nConfusion Matrix:")
#     print(" " * 8 + "".join([f"Pred {i:>6}" for i in range(num_classes)]))
#     for i in range(num_classes):
#         if i == 0:
#             print(f"True {i}  " + " ".join([f"{val:6d}" for val in cm[i]]))
#         else:
#             print(f"     {i}  " + " ".join([f"{val:6d}" for val in cm[i]]))
    
#     # Class distribution
#     true_dist = np.bincount(y_true, minlength=num_classes)
#     pred_dist = np.bincount(y_pred, minlength=num_classes)
    
#     print(f"\nClass Distribution:")
#     print(f"  True: {dict(enumerate(true_dist))}")
#     print(f"  Pred: {dict(enumerate(pred_dist))}")
    
#     print("="*80)

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
  # In the argument parser section (around line where other arguments are defined)
  parser.add_argument('--input_dim', type=int, default=51)
  parser.add_argument('--pose_dim', type=int, default=51)
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
  parser.add_argument('--num_classes', type=int, default=4)  # NEW: Add num_classes parameter
  args = parser.parse_args()
  
  params = vars(args)
  
  # Count folds for downstream task (using the correct flat structure)
  if params['task'] == 'downstream':
#    num_folds = count_folds(params['data_path'])
    num_folds = 1 
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
  
  # NEW: Track training time
  total_training_start = time.time()

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
    
    # NEW: Update model with actual fold class distribution
    model_fn.update_fold_weights(train_dataset_fn)  # Compute per-fold weights
    
    # NEW: Set current fold for CSV logging
    model_fn.csv_logger.current_fold = fold
    
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

  # NEW: Calculate total training time
  total_training_time = time.time() - total_training_start

  if params['task'] == 'downstream':
    # Only generate report if we have actual data
    if len(total_gts) > 0 and len(total_preds) > 0:
        # Get number of classes from parameter or infer from data
        num_classes = params.get('num_classes', len(np.unique(total_gts + total_preds)))
        
        print("\n" + "="*80)
        print("=== Final Classification Report ===")
        print(classification_report(total_gts, total_preds))
        print(f"Overall accuracy: {accuracy_score(total_gts, total_preds):.4f}")
        print("="*80)
        
        # NEW: Comprehensive metrics and detailed report
        comprehensive_metrics = compute_comprehensive_metrics(total_gts, total_preds, num_classes)
        print_detailed_report(total_gts, total_preds, num_classes)
        
        # NEW: Log final metrics to CSV
        if 'model_fn' in locals():
            model_fn.csv_logger.log_final_metrics(num_folds, comprehensive_metrics, total_training_time)
        
        # Save comprehensive metrics to JSON
        metrics_filename = os.path.join(params['model_prefix'], 'comprehensive_metrics.json')
        with open(metrics_filename, 'w') as f:
            json.dump(comprehensive_metrics, f, indent=4, cls=NumpyEncoder)
        print(f"✓ Comprehensive metrics saved to: {metrics_filename}")
        
        # NEW: Print CSV file locations
        print(f"\n✓ CSV logs saved to:")
        print(f"  Epoch metrics: {model_fn.csv_logger.epochs_csv_path}")
        print(f"  Fold metrics: {model_fn.csv_logger.folds_csv_path}")
        print(f"  Final metrics: {model_fn.csv_logger.final_metrics_csv_path}")
        print(f"✓ Total training time: {total_training_time:.2f} seconds")
        
    else:
        print("\nWARNING: No valid predictions were generated.")
        print("This typically happens when:")
        print("1. The data path is incorrect")
        print("2. Required EPG files are missing or misnamed")
        print("3. No valid data was found during processing")
        print("Please check your data structure and paths.")
