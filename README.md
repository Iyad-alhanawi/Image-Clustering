# Image Clustering using VGG16 and KMeans

This Python script utilizes a pre-trained VGG16 model for feature extraction and the KMeans clustering algorithm to cluster images based on extracted features. Images are organized into separate folders corresponding to the identified clusters.

## Overview

The script automates the process of image clustering by leveraging deep learning for feature extraction and clustering algorithms for grouping similar images together. The steps involved include:

1. **Feature Extraction**:
   - Utilizes the VGG16 model to extract high-level features from images.
  
2. **Resizing**:
   - The images are resized to the required dimensions (224x224 pixels) to match the input size expected by the VGG16 model.

3. **Preprocessing**:
   - Preprocessing steps are applied to the images to ensure they are in the correct format for feature extraction. This includes normalization and reshaping.

4. **Clustering**:
   - Applies the KMeans algorithm to cluster images based on their extracted features.

5. **Organizing Images**:
   - Moves images to individual folders based on the clusters they belong to.
