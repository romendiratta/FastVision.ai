import torch
import os
import numpy as np
import SimpleITK as sitk
import numpy as np


class CTDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, metadata, base_file_path, samples_per_epoch):
        super(CTDatasetTrain).__init__()
        self.metadata = metadata
        self.base_file_path = base_file_path
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
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
        ct_scan = np.expand_dims(ct_scan, 0)
        return ct_scan, label
