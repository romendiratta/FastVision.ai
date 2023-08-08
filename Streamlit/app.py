# Standard Library
import os
import csv
import shutil
import sys
import time
import statistics
from io import BytesIO

# Third-party Packages
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pydicom
import imageio
import glob
import boto3
import SimpleITK as sitk
import torch
from model.Model import Model

# Local Modules
from lungmask import LMInferer


import torch
# torch.cuda.empty_cache()

##########################################################################################
##########################################################################################
##########################################################################################
def resample_volume(volume_path, interpolator = sitk.sitkLinear, new_spacing = [1,1, 1]):
    volume = sitk.ReadImage(volume_path) # read and cast to float32
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), interpolator,
                         volume.GetPixelID())


def inferer_mask(ct_scan_volume, offset=0):
    '''
    Input is 3D numpy array of original CT scan and output is 3D numpy array of generated mask
    '''
    inferer = LMInferer()
    #Defines numpy offset to allow segmentation to work across different datasets (LUNA16 doesn't need)
    ct_scan_volume -= offset
    mask_slices = []
    lung_mask_volume = inferer.apply(ct_scan_volume)
    for slice_idx in range(ct_scan_volume.shape[0]):
        mask_slices.append(lung_mask_volume[slice_idx, :, :])

    mask_slices = np.stack(mask_slices, axis=0)
    
    return mask_slices


def segmented_inferer_volume(ct_scan,offset=0):
    '''
    Input is 3D numpy array of original CT scan and output is 3D numpy array of segmented volume
    '''
    #Defines numpy offset to allow segmentation to work across different datasets (LUNA16 doesn't need)
    ct_scan -= offset
    print('max:',ct_scan.max())
    print('min:',ct_scan.min())
    lung_mask_volume = inferer_mask(ct_scan)
    segmented_slices = []
    tracker = []
    for slice_idx in range(ct_scan.shape[0]):
        ct_slice = ct_scan[slice_idx, :, :]
        lung_mask_slice = lung_mask_volume[slice_idx, :, :]
        masked_slice = np.where(lung_mask_slice != 0, ct_slice, np.min(ct_slice))
        segmented_slices.append(masked_slice)
        
        #keeps track of value ranges only in lungs
        tracker_slice = np.where(lung_mask_slice != 0, ct_slice, -99999)
        tracker.append(tracker_slice)
        
    segmented_slices = np.stack(segmented_slices, axis=0)
    
    # Concatenate the arrays into a single array
    concatenated_array = np.concatenate(tracker)
    
    # Filter the array to exclude values equal to -99999
    concatenated_array  = concatenated_array[concatenated_array  != -99999]

    # Find the minimum and maximum values inside lung volume
    min_value = np.min(concatenated_array)
    max_value = np.max(concatenated_array)

    # Normalize the segmented volume to a range of 0 to 1
    segmented_slices = (segmented_slices - min_value) / (max_value - min_value)
    
    #Replace background with 0's
    background_value = segmented_slices[0][0][0]
    segmented_slices = np.where(segmented_slices == background_value, 0, segmented_slices)
    
    
    return segmented_slices


def remove_blank_slices(segmented_ct_scan,ct_scan=None):
    '''
    Input is 3D numpy array of segmented volume and output is numpy array slices that only contain lung volume
    
    If ct_scan file is added, it will remove the correlated slice from the original CT scan for comparison
    '''
    segment_result = []
    ct = []
    # Iterate through each slice 
    for slice_2d in range(segmented_ct_scan.shape[0]):
        # Check if all elements in the slice have the same value
        if not np.all(segmented_ct_scan[slice_2d] == segmented_ct_scan[slice_2d][0]):
            segment_result.append(segmented_ct_scan[slice_2d, :, :])
            if ct_scan is not None:
                ct.append(ct_scan[slice_2d, :, :])

    # Convert the result back to a NumPy array
    segment_result = np.stack(segment_result, axis=0)
    
    if ct_scan is not None:
        ct = np.stack(ct, axis=0)
        return segment_result, ct
    else:
        return segment_result

def process_dicom_folder(dicom_dir):
    # Check if the directory contains .dcm files
    dicom_files = [file for file in os.listdir(dicom_dir) if file.endswith(".dcm")]

    # Load all DICOM files in the directory
    dicom_files = [pydicom.dcmread(os.path.join(dicom_dir, file)) for file in dicom_files]

    # Sort the DICOM files based on the filename
    dicom_files.sort(key=lambda x: x.filename)

    # Get the pixel arrays from the DICOM files
    slices = [dicom_file.pixel_array.astype(np.int16) for dicom_file in dicom_files]

    # Convert the list of pixel arrays to a 3D numpy array
    ct_scan = np.stack(slices)
    
    # Ensure that no value exceeds 5000
    ct_scan = np.clip(ct_scan, a_min=None, a_max=5000)

    # Normalize the values to a range between -3000 and 3000
    #ct_scan = ((ct_scan - ct_scan.min()) / (ct_scan.max() - ct_scan.min())) * 2000 - 1000
    
    dicom = pydicom.dcmread((os.path.join(dicom_dir, os.listdir(dicom_dir)[0])))
    
    thickness = float(dicom.SliceThickness)
    pixel_x, pixel_y = dicom.PixelSpacing
    
    spacing = (float(pixel_x),float(pixel_y), float(thickness))
    
    origins_x = [dicom_file.ImagePositionPatient[0] for dicom_file in dicom_files]
    origins_y = [dicom_file.ImagePositionPatient[1] for dicom_file in dicom_files]
    origins_z = [dicom_file.ImagePositionPatient[2] for dicom_file in dicom_files]
    
    origin = (float(statistics.mean(origins_x)), float(statistics.mean(origins_y)),float(statistics.mean(origins_z)))

    print('Min:', ct_scan.min())
    print('Max:', ct_scan.max())
    print('origin:', origin)
    print('spacing:', spacing)
    return ct_scan, origin, spacing

def check_dicom_folder(dicom_dir):
    # Check if the directory contains .dcm files
    dicom_files = [file for file in os.listdir(dicom_dir) if file.endswith(".dcm")]
    if not dicom_files:
        print(f"No DICOM files found in directory: {dicom_dir}")
        return

    # Check if the folder has more than 10 files
    if len(dicom_files) <= 10:
        print(f"Insufficient number of files in directory: {dicom_dir}")
        return
    
    # Load all DICOM files in the directory
    dicom_files = [pydicom.dcmread(os.path.join(dicom_dir, file)) for file in dicom_files]

    # Sort the DICOM files based on the filename
    dicom_files.sort(key=lambda x: x.filename)

    # Get the pixel arrays from the DICOM files
    slices = [dicom_file.pixel_array for dicom_file in dicom_files]

    # Convert the list of pixel arrays to a 3D numpy array
    ct_scan = np.stack(slices)

    return ct_scan

# Define a function to convert PIL Image to SimpleITK Image
def pil_to_sitk(pil_image):
    np_array = np.array(pil_image)
    sitk_image = sitk.GetImageFromArray(np_array)
    return sitk_image


def segment_single_scan(root_dir,output_file_path,mhd_file_name,voxel_size=2):
    #Initialize lungmask inferer
    inferer = LMInferer()

    # Get a list of all files in the root directory
    all_files = os.listdir(root_dir)

    # Filter DICOM files
    dicom_files = [filename for filename in all_files if filename.endswith(".dcm")]

    # Filter mhd files
    mhd_files = [filename for filename in all_files if filename.endswith(".mhd")]

    cube_length = 512/voxel_size

    # Check if there are any DICOM files in the root_dir
    if len(dicom_files) > 0 and len(dicom_files) == len(all_files):
        print("Segmenting .dcm files...")
        mhd_file = check_dicom_folder(root_dir)
        if isinstance(mhd_file, np.ndarray):
            mhd_file, origin, spacing = process_dicom_folder(root_dir)
            image_sitk = sitk.GetImageFromArray(mhd_file)
            sitk.WriteImage(image_sitk, output_file_path + '/'+ mhd_file_name +'.mhd')
            volume = sitk.ReadImage( output_file_path +  '/'+ mhd_file_name+'.mhd') # read and cast to float32
            volume = sitk.Cast(volume, sitk.sitkInt16)
            volume.SetSpacing(spacing)
            volume.SetOrigin(origin)
            sitk.WriteImage(volume, output_file_path +'/'+ mhd_file_name +'.mhd')
            # Use glob to retrieve a list of .mhd file paths within the input folder
            mhd_file_paths = glob.glob(os.path.join(output_file_path, '*.mhd'))

    elif len(mhd_files) > 0 and len(dicom_files)==0:
        print("Segmenting .mhd file...")
        mhd_file_paths = glob.glob(os.path.join(root_dir, '*.mhd'))

    elif len(mhd_files) > 0 and len(dicom_files) > 0:
        print("Folder contains both .mhd and .dcm files. Please upload only one scan at a time.")
        sys.exit()

    # Normalize voxel size for each file, segment lungs, and save as new file
    for file_path in mhd_file_paths:
        ct_scan_resampled = resample_volume(file_path,new_spacing = [voxel_size, voxel_size, voxel_size])
        file_name = os.path.basename(file_path)  # Get the file name
        #output_file_path = os.path.join(output_folder_path, file_name)
        #sitk.WriteImage(ct_scan_resampled, output_file_path)
        ct_scan_resampled = np.array(sitk.GetArrayFromImage(ct_scan_resampled), dtype=np.float32)

        min_value = ct_scan_resampled.min()
        max_value = ct_scan_resampled.max()
        ct_scan_resampled = 6000 * ((ct_scan_resampled - min_value) / (max_value - min_value)) - 3000

        unet_segmented_volume = segmented_inferer_volume(ct_scan_resampled)
        unet_trimmed_volume = remove_blank_slices(unet_segmented_volume)
        min_value = np.min(unet_trimmed_volume)

        # Define the target shape
        target_shape = (int(cube_length), int(cube_length), int(cube_length))

        # Calculate the required padding for each dimension
        pad_depth = max(target_shape[0] - unet_trimmed_volume.shape[0], 0)
        pad_height = max(target_shape[1] - unet_trimmed_volume.shape[1], 0)
        pad_width = max(target_shape[2] - unet_trimmed_volume.shape[2], 0)

        # Calculate the padding amounts for each side of each dimension
        pad_depth_before = pad_depth // 2
        pad_depth_after = pad_depth - pad_depth_before

        pad_height_before = pad_height // 2
        pad_height_after = pad_height - pad_height_before

        pad_width_before = pad_width // 2
        pad_width_after = pad_width - pad_width_before

        # Pad the array with the minimum value
        padded_data = np.pad(unet_trimmed_volume, ((pad_depth_before, pad_depth_after), (pad_height_before, pad_height_after), (pad_width_before, pad_width_after)), mode='constant', constant_values=min_value)

        #normalize all values to be between 0 and 1
        padded_data_norm = (padded_data-np.min(padded_data))/(np.max(padded_data)-np.min(padded_data))

        image_sitk = sitk.GetImageFromArray(padded_data_norm)
        #image_sitk = sitk.Cast(image_sitk, sitk.sitkInt16)
        sitk.WriteImage(image_sitk, output_file_path + '/' + mhd_file_name+'_segmented.mhd')
        

        print('done')





def predict(input_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Model())
    with open(os.path.join("model", "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model.to(device)
    input_ = np.expand_dims(input_, 0)
    with torch.no_grad():
        return model(torch.unsqueeze(torch.from_numpy(input_).to(device), 0))[0][0]



##########################################################################################
##########################################################################################
##########################################################################################


import matplotlib.pyplot as plt
import streamlit as st
import os
import SimpleITK as sitk
import numpy as np

def mhd_to_image(mhd_file):
    # Use SimpleITK to read the mhd file
    sitk_image = sitk.ReadImage(mhd_file)

    # Convert the SimpleITK image to a numpy array
    np_image = sitk.GetArrayFromImage(sitk_image)

    return np_image

st.title("Lung Segmentation")


mhd_file = st.file_uploader("Please upload a .mhd file", type=["mhd"])
raw_file = st.file_uploader("Please upload a .raw file", type=["raw"])

# After both files have been uploaded, read and process them
if mhd_file and raw_file:
    # Put a delay for 2 seconds
    time.sleep(1)
    # with open(os.path.join("tmp", mhd_file.name), 'bw+') as f:
    #     f.write(mhd_file.getvalue())
    # with open(os.path.join("tmp", raw_file.name), 'bw+') as f:
    #     f.write(raw_file.getvalue())

    # np_image = mhd_to_image(os.path.join("tmp", mhd_file.name))
    np_image = mhd_to_image("3.000000-1.25MM CHEST BONE PLUS-58248.mhd")
    prediction = predict(np_image)
    
    # Display the image slice
    plt.imshow(np_image[128], cmap='gray')
    st.pyplot(plt)
    st.success(f'Model prediction: {prediction} confidence')