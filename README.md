# FastVision.ai: Tunable ML Solution for Lung Cancer Screening

## Abstract

Medical Imaging, specifically Computerized Tomography (CT) scanning, is one of the main modalities for Lung Cancer screening, performed to detect Lung Cancer early. However, even after individuals have gone through the diagnostic test, it takes weeks or months to get the results because there are few specialized doctors capable of identifying the patches/growth visually and are able to differentiate between benign vs malignant tumors. Given the aggressive nature of this pathologic condition, this delay can make the patient outcome far worse if time progresses. In the recent years there have been many groups using Deep Learning techniques to be able to detect Cancerous growth in lungs of Screening individuals. However, these are closely controlled studies and even small differences between data sources (based on different imaging sites, or the hardware used to collect these images, image processing technique and storage format etc) makes the images different enough that a Model trained on one data set from these studies, cannot be used to make predictions from images in the real world.

Based on reviewing recent literature, it seems that there are some ML modeling strategies that have consistently generated good lung cancer models, such as use of CNNs, Object recognition etc. Therefore there is a potential of standardizing across various datasets. In this project we are going to develop a ML based product/solution, that consists of a pre-trained model (we will design and train this model to be a source/dataset agnostic), that a customer can download onto their own secure data infrastructure and use their own data sets to train. Then they can use this new model to make predictions of new incoming lung screening images from their own clinics. The end deliverable of this project will be a web based UI, that allows personnel from the customer site, to interact sufficiently with the model to change certain properties of the model in order to optimize it. There will also be a chatbot (we will use LLM to optimize the chatbot) that helps guide the customer to optimize and use this plug-n-play model. Once the customer/hospital has an optimized ML model, they will then use this tool to automatically classify whether an individual has cancer or not. This will allow hospitals to “own” and manage their own Lung Cancer ML model, which would help solve problems around incompatibility of images generated from different sources as well as other overarching issues such as patient information privacy, data security etc, which are some of the main barriers to adoption of this technology at hospitals.

## Contents of Repo

- EDA
  - Lung_segmentation_demo
- Modeling
  - EDA_and_Baseline_Model (CNN with ResNet50, ImageNet transfer learning)
  - Lung Segmentation Pipeline Model
  - 3D CNN (Draft)
  - Sagemaker Scaffolding Code
- Streamlit
  - App.py
  - requirements.txt
  - run-local.sh
  - Dockerfile

## How to Use FastVision.ai

### 1. Clone the Repo
git clone https://github.com/romendiratta/FastVision.ai.git
cd FastVision.ai

### 2. Install Dependencies
TODO: Add installation instructions for any specific dependencies here"

### 3. Download Pre-trained Model
Visit our website at https://fastvision.ai/models to download the pre-trained model for lung cancer screening. Save the model file in the models directory.

### 4. Train the Model
TODO: Add instructions to train the model with custom datasets"

### 5. Run the Web UI
TODO: Add instructions to run the web-based user interface.

### 6. Optimize Model with Chatbot
TODO: # Add instructions for using the chatbot to optimize the model.

## Contributing

We welcome contributions from the community. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

MIT License (Replace this with the actual license your project uses)

## Contact

For any inquiries, please contact rmendiratta@ischool.berkeley.edu








