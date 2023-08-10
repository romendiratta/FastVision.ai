import torch
import os
import math
import numpy as np
import SimpleITK as sitk


class CTDataset(torch.utils.data.IterableDataset):
    def __init__(self, metadata, base_file_path, samples_per_epoch):
        super(CTDataset).__init__()
        self.metadata = metadata
        self.base_file_path = base_file_path
        self.samples_per_epoch = samples_per_epoch
        self.counter = 0

    def __len__(self):
        return self.samples_per_epoch

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is None:  # single-process data loading, return the full iterator
        #     iter_start = self.start
        #     iter_end = self.end
        # else:  # in a worker process
        #     # split workload
        #     per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        #     worker_id = worker_info.id
        #     iter_start = self.start + worker_id * per_worker
        #     iter_end = min(iter_start + per_worker, self.end)
        # # Load data for the current worker's portion
        # for idx in range(iter_start, iter_end):
        # Get the data sample's metadata (e.g., file name, label)
        if self.counter > self.samples_per_epoch:
            raise StopIteration

        self.counter += 1
        choice = np.random.randint(2)
        sample_metadata = self.metadata[self.metadata["class"] == choice].sample(n=1)

        file_name = sample_metadata['seriesuid'].values[0]
        # Load the actual data sample (CT scan) based on the file name
        file_path = os.path.join(self.base_file_path, file_name + ".mhd")
        ct_scan = sitk.ReadImage(file_path)
        # Convert the SimpleITK image to a numpy array
        ct_scan = sitk.GetArrayFromImage(ct_scan)
        label = sample_metadata['class'].values[0]  # Get the label from the metadata
        # Yield the data sample (CT scan and label) to the DataLoader
        yield ct_scan, label
