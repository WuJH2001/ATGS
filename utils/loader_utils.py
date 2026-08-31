
import os
import cv2
import random
import numpy as np
from PIL import Image
 
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms, utils
import random
def get_stamp_list(dataset, timestamp):
    frame_length = int(len(dataset)/len(dataset.dataset.poses))
    # print(frame_length)
    if timestamp > frame_length:
        raise IndexError("input timestamp bigger than total timestamp.")
    print("select index:",[i*frame_length+timestamp for i in range(len(dataset.dataset.poses))])
    return [dataset[i*frame_length+timestamp] for i in range(len(dataset.dataset.poses))]


class EncoderBalancedSampler(Sampler):
    """Yield one random frame/view from every routed encoder per logical batch."""

    def __init__(self, dataset, num_encoders):
        self.dataset = dataset
        self.num_encoders = max(int(num_encoders), 1)
        self.num_frames = int(dataset.dataset.N_frames)
        if self.num_frames <= 0 or len(dataset) % self.num_frames != 0:
            raise ValueError("EncoderBalancedSampler requires a camera-major dataset with a fixed frame count")

        self.num_views = len(dataset) // self.num_frames
        self.encoder_buckets = [[] for _ in range(self.num_encoders)]
        for view_id in range(self.num_views):
            view_offset = view_id * self.num_frames
            for frame_id in range(self.num_frames):
                encoder_id = min(int(frame_id * self.num_encoders / self.num_frames), self.num_encoders - 1)
                self.encoder_buckets[encoder_id].append(view_offset + frame_id)

        if any(not bucket for bucket in self.encoder_buckets):
            raise ValueError(
                f"Cannot sample {self.num_encoders} encoders from only {self.num_frames} frames"
            )
        self.logical_batches = max(len(bucket) for bucket in self.encoder_buckets)

    def __iter__(self):
        shuffled_buckets = [
            [bucket[i] for i in torch.randperm(len(bucket)).tolist()]
            for bucket in self.encoder_buckets
        ]
        for batch_id in range(self.logical_batches):
            for encoder_id in torch.randperm(self.num_encoders).tolist():
                bucket = shuffled_buckets[encoder_id]
                yield bucket[batch_id % len(bucket)]

    def __len__(self):
        return self.logical_batches * self.num_encoders


class FineSampler(Sampler):
    def __init__(self, dataset):
        self.len_dataset = len(dataset) 
        self.len_pose = len(dataset.dataset.poses)
        self.frame_length = int(self.len_dataset/ self.len_pose)

        sample_list = []
        for i in range(self.frame_length):
            for j in range(4):
                idx = torch.randperm(self.len_pose) *self.frame_length + i
                # print(idx)
                # breakpoint()
                now_list = []
                cnt = 0
                for item in idx.tolist():
                    now_list.append(item)
                    cnt+=1
                    if cnt % 2 == 0 and len(sample_list)>2:    
                        select_element = [x for x in random.sample(sample_list,2)]
                        now_list += select_element
            
            sample_list += now_list
            
        self.sample_list = sample_list
        # print(self.sample_list)
        # breakpoint()
        print("one epoch containing:",len(self.sample_list))
    def __iter__(self):

        return iter(self.sample_list)
    
    def __len__(self):
        return len(self.sample_list)
