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

def segment_single_scan(mhd_file_path, raw_file_path, output_file_path, voxel_size=2):
    # Initialize lungmask inferer
    inferer = LMInferer()

    cube_length = 512 / voxel_size

    # Check if the input file is a DICOM file
    if mhd_file_path.endswith(".dcm"):
        print("Segmenting .dcm file...")
        mhd_file, origin, spacing = process_dicom_file(mhd_file_path, raw_file_path)
        image_sitk = sitk.GetImageFromArray(mhd_file)
        mhd_file_name = os.path.splitext(os.path.basename(mhd_file_path))[0]
        sitk.WriteImage(image_sitk, os.path.join(output_file_path, mhd_file_name + '.mhd'))
        volume = sitk.ReadImage(os.path.join(output_file_path, mhd_file_name + '.mhd')) # read and cast to float32
        volume = sitk.Cast(volume, sitk.sitkInt16)
        volume.SetSpacing(spacing)
        volume.SetOrigin(origin)
        sitk.WriteImage(volume, os.path.join(output_file_path, mhd_file_name + '.mhd'))
        mhd_file_paths = [os.path.join(output_file_path, mhd_file_name + '.mhd')]

    # Check if the input file is an MHD file
    elif mhd_file_path.endswith(".mhd"):
        print("Segmenting .mhd file...")
        mhd_file_paths = [mhd_file_path]

    else:
        print("Unsupported file type. Please upload a .dcm or .mhd file.")
        return

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


# Layout for logo and title
col1, col2 = st.columns([1, 6])  # Adjust the ratio if needed

# Display the logo in the left column
col1.image("fastvision_logo.png", width=250)  # Adjust width as necessary
# st.markdown('<hr style="color:#2074cc; background-color:#2074cc; height:1px; border:none;"/>', unsafe_allow_html=True)

# Display the title in the right column
# col2.title("FastVision.ai")

# Create columns for file uploaders
col_file1, col_file2 = st.columns(2)

# Place the file uploaders in their respective columns
# Create columns for file uploaders and spacing
col_file1, col_space, col_file2 = st.columns([1, 0.1, 1])  # Adjust the middle value to increase or decrease the spacing

# Place the file uploaders in their respective columns
mhd_file = col_file1.file_uploader("Please drag and drop a .mhd file", type=["mhd"])

# For clarity, you can put an empty space in the middle column, although it's optional
col_space.empty()

raw_file = col_file2.file_uploader("Please drag and drop a .raw file", type=["raw"])



# After both files have been uploaded, read and process them
if mhd_file and raw_file:
    # Put a delay for 2 seconds
    time.sleep(1)

    with open(os.path.join(mhd_file.name), 'bw+') as f:
        f.write(mhd_file.getvalue())
    with open(os.path.join(raw_file.name), 'bw+') as f:
        f.write(raw_file.getvalue())

    np_image = mhd_to_image("./Radiogenomics/user_output_test/test_2mm_segmented.mhd")
    prediction = predict(np_image)
    
    plt.imshow(np_image[128], cmap='gray')
    st.pyplot(plt)

    if prediction > .5:
        prediction_message = "Nodule Detected ({:.2f}% Confidence)".format(prediction * 100)
        st.success(prediction_message)
    else:
        prediction_message = "No Nodule Detected ({:.2f}% Confidence)".format((prediction) * 100)
        st.error(prediction_message)

    # Display the question and buttons inline using markdown
    feedback_prompt = st.markdown("""How did our model perform compared to your prediction?""")

    # Buttons using Streamlit's native method (hidden visually but detectable programmatically)
    good_pred = st.button("👍")
    bad_pred = st.button("👎")


    # Check if either button is clicked
    if good_pred or bad_pred:
        feedback = st.text_area("Thank you! Please provide your feedback which will help us improve our model.:")

        # Optional: Add a submit button for feedback
        submit_feedback = st.button("Submit")
        
        if submit_feedback:
            # Handle the feedback, e.g., save it to a database or file
            st.write("Thank you for your feedback!")