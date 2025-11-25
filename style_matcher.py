#!/usr/bin/env python3
"""
make_gaitforemer_dataset.py

Converts your frame folders (healthy / unhealthy) into GaitForeMer-style:
  GaitForeMer_dataset/
    fold_1/
      1/
        H1/
          seq_000000.npy
          seq_001000.npy
        H2/
      2/
        H5/
      labels.npy   # label per sequence (0 healthy, 1 unhealthy)
"""
import os
import argparse
import numpy as np
from glob import glob
import shutil

def make_sequences_from_frames(video_dir, seq_len, stride, pose_dim, target_dir, copy_frames=False):
    frame_paths = sorted(glob(os.path.join(video_dir, "*.jpg")) + glob(os.path.join(video_dir, "*.png")))
    n_frames = len(frame_paths)
    if n_frames < seq_len:
        return 0  # nothing created

    created = 0
    if copy_frames:
        os.makedirs(target_dir, exist_ok=True)
        for p in frame_paths:
            shutil.copy2(p, target_dir)

    rng = np.random.RandomState(12345)
    poses = [rng.rand(pose_dim).astype(np.float32) for _ in range(n_frames)]

    for start in range(0, n_frames - seq_len + 1, stride):
        seq = np.stack(poses[start:start + seq_len])
        seq_filename = os.path.join(target_dir, f"seq_{start:06d}.npy")
        np.save(seq_filename, seq)
        created += 1

    return created

def main():
    parser = argparse.ArgumentParser(description="Create GaitForeMer-style dataset from frame folders.")
    parser.add_argument("--healthy", required=True)
    parser.add_argument("--unhealthy", required=True)
    parser.add_argument("--target", default="GaitForeMer_dataset")
    parser.add_argument("--fold", default="fold_1")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--walk_sec", type=int, default=10)
    parser.add_argument("--pose_dim", type=int, default=51)
    parser.add_argument("--copy-frames", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    healthy_root = os.path.abspath(args.healthy)
    unhealthy_root = os.path.abspath(args.unhealthy)
    target_fold_root = os.path.join(os.path.abspath(args.target), args.fold)
    os.makedirs(target_fold_root, exist_ok=True)

    seq_len = args.fps * args.walk_sec
    stride = seq_len

    all_sequence_labels = []
    total_sequences = 0
    skipped_videos = 0

    # Process healthy + unhealthy
    for class_root, label in [(healthy_root, 0), (unhealthy_root, 1)]:
        for person_dir in sorted(glob(os.path.join(class_root, "*"))):
            if not os.path.isdir(person_dir):
                continue
            person_id = os.path.basename(person_dir)  # e.g., "1", "2"
            target_person_dir = os.path.join(target_fold_root, person_id)
            os.makedirs(target_person_dir, exist_ok=True)

            for video_dir in sorted(glob(os.path.join(person_dir, "*"))):
                if not os.path.isdir(video_dir):
                    continue
                video_id = os.path.basename(video_dir)
                target_video_dir = os.path.join(target_person_dir, video_id)
                os.makedirs(target_video_dir, exist_ok=True)

                created = make_sequences_from_frames(
                    video_dir=video_dir,
                    seq_len=seq_len,
                    stride=stride,
                    pose_dim=args.pose_dim,
                    target_dir=target_video_dir,
                    copy_frames=args.copy_frames
                )

                if created == 0:
                    skipped_videos += 1
                    if args.verbose:
                        print(f"SKIP {video_id}: not enough frames (<{seq_len}).")
                    continue

                all_sequence_labels.extend([label] * created)
                total_sequences += created
                if args.verbose:
                    print(f"Processed {video_id} (Person {person_id}): created {created} sequences, label={label}")

    labels_path = os.path.join(target_fold_root, "labels.npy")
    np.save(labels_path, np.array(all_sequence_labels, dtype=np.int32))

    print("=== DONE ===")
    print(f"Total sequences created: {total_sequences}")
    print(f"Videos skipped (too short): {skipped_videos}")
    print(f"Labels saved to: {labels_path}")
    print(f"Dataset root: {target_fold_root}")
    print("Note: currently generates DUMMY pose vectors (random). Replace with real poses later.")

if __name__ == "__main__":
    main()
