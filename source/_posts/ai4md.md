---
categories: Machine Learning
title: AI for Medicine
date: 2020-04-22 15:37:42
tags: Machine Learning
---

最近COURSERA上新开了一门课:[AI for Medicine Specialization](https://www.coursera.org/specializations/ai-for-medicine)

# [AI for Medical Diagnosis](https://www.coursera.org/learn/ai-for-medical-diagnosis/home/welcome)
1. [ChestX-ray8 dataset](https://arxiv.org/abs/1705.02315)

2. Image Preprocessing in Keras
```
# Import data generator from keras
from keras.preprocessing.image import ImageDataGenerator

# Normalize images
image_generator = ImageDataGenerator(
    samplewise_center=True, #Set each sample mean to 0.
    samplewise_std_normalization= True # Divide each input by its standard deviation
)

# Flow from directory with specified batch size and target image size
generator = image_generator.flow_from_dataframe(
        dataframe=train_df,
        directory="nih/images-small/",
        x_col="Image", # features
        y_col= ['Mass'], # labels
        class_mode="raw", # 'Mass' column should be in train_df
        batch_size= 1, # images per batch
        shuffle=False, # shuffle the rows or not
        target_size=(320,320) # width and height of output image
)

# Plot a processed image
sns.set_style("white")
generated_image, label = generator.__getitem__(0)
plt.imshow(generated_image[0], cmap='gray')
plt.colorbar()
plt.title('Raw Chest X Ray Image')
print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height")
print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")

# Include a histogram of the distribution of the pixels
sns.set()
plt.figure(figsize=(10, 7))

# Plot histogram for original iamge
sns.distplot(raw_image.ravel(), 
             label=f'Original Image: mean {np.mean(raw_image):.4f} - Standard Deviation {np.std(raw_image):.4f} \n '
             f'Min pixel value {np.min(raw_image):.4} - Max pixel value {np.max(raw_image):.4}',
             color='blue', 
             kde=False)

# Plot histogram for generated image
sns.distplot(generated_image[0].ravel(), 
             label=f'Generated Image: mean {np.mean(generated_image[0]):.4f} - Standard Deviation {np.std(generated_image[0]):.4f} \n'
             f'Min pixel value {np.min(generated_image[0]):.4} - Max pixel value {np.max(generated_image[0]):.4}', 
             color='red', 
             kde=False)

# Place legends
plt.legend()
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixel')

```

3. Dense net
```
# Import Densenet from Keras
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

# Create the base pre-trained model
base_model = DenseNet121(weights='./nih/densenet.hdf5', include_top=False);

# Print the model summary
base_model.summary()

# The number of output channels
print("The output has 1024 channels")
x = base_model.output

# Add a global spatial average pooling layer
x_pool = GlobalAveragePooling2D()(x)

# Define a set of five class labels to use as an example
labels = ['Emphysema', 
          'Hernia', 
          'Mass', 
          'Pneumonia',  
          'Edema']
n_classes = len(labels)
print(f"In this example, you want your model to identify {n_classes} classes")

# Add a logistic layer the same size as the number of classes you're trying to predict
predictions = Dense(n_classes, activation="sigmoid")(x_pool)
print(f"Predictions have {n_classes} units, one for each class")

# Create an updated model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy')
# (You'll customize the loss function in the assignment!)
```

# Chest X-Ray Medical Diagnosis with Deep Learning
1. Import packages and functions
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from keras.models import load_model

import util
```

2. Load the Datasets
```
train_df = pd.read_csv("nih/train-small.csv")
valid_df = pd.read_csv("nih/valid-small.csv")
test_df = pd.read_csv("nih/test.csv")
```

  - Preventing Data Leakage
  ```
  # UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
  def check_for_leakage(df1, df2, patient_col):
      """
      Return True if there any patients are in both df1 and df2.

      Args:
          df1 (dataframe): dataframe describing first dataset
          df2 (dataframe): dataframe describing second dataset
          patient_col (str): string name of column with patient IDs
      
      Returns:
          leakage (bool): True if there is leakage, otherwise False
      """

      ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
      
      df1_patients_unique = list(set(df1[patient_col].tolist()))
      df2_patients_unique = list(set(df2[patient_col].tolist()))
      
      patients_in_both_groups = list(set(df1_patients_unique+df2_patients_unique))

      # leakage contains true if there is patient overlap, otherwise false.
      leakage = False # boolean (true if there is at least 1 patient in both groups)
      if len(patients_in_both_groups)<len(df1_patients_unique)+len(df2_patients_unique):
          leakage = True
      
      ### END CODE HERE ###
      
      return leakage
  ```

  - Preparing Images
  ```
  def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator
    
  def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 320, target_h = 320):
    """
    Return generator for validation set and test test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=IMAGE_DIR, 
        x_col="Image", 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator

  IMAGE_DIR = "nih/images-small/"
  train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
  valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)

  x, y = train_generator.__getitem__(0)
  plt.imshow(x[0]);
    ```
3. Model Development
  - Addressing class imbalance
  ```
  # UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
  def compute_class_freqs(labels):
      """
      Compute positive and negative frequences for each class.

      Args:
          labels (np.array): matrix of labels, size (num_examples, num_classes)
      Returns:
          positive_frequencies (np.array): array of positive frequences for each
                                           class, size (num_classes)
          negative_frequencies (np.array): array of negative frequences for each
                                           class, size (num_classes)
      """
      ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
      
      # total number of patients (rows)
      N = len(labels)
      
      positive_frequencies = sum(labels)/N
      negative_frequencies = 1-positive_frequencies

      ### END CODE HERE ###
      return positive_frequencies, negative_frequencies

  freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
  pos_weights = freq_neg
  neg_weights = freq_pos
  pos_contribution = freq_pos * pos_weights 
  neg_contribution = freq_neg * neg_weights

  # UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
  def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
      """
      Return weighted loss function given negative weights and positive weights.

      Args:
        pos_weights (np.array): array of positive weights for each class, size (num_classes)
        neg_weights (np.array): array of negative weights for each class, size (num_classes)
      
      Returns:
        weighted_loss (function): weighted loss function
      """
      def weighted_loss(y_true, y_pred):
          """
          Return weighted loss value. 

          Args:
              y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
              y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
          Returns:
              loss (Tensor): overall scalar loss summed across all classes
          """
          # initialize loss to zero
          loss = 0.0
          
          ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

          for i in range(len(pos_weights)):
              # for each class, add average weighted loss for that class 
              loss += K.mean(-(pos_weights[i]*y_true[:,i]*K.log(y_pred[:,i]+epsilon)+neg_weights[i]*(1-y_true[:,i])*K.log(1-y_pred[:,i]+epsilon)))
          return loss
      
          ### END CODE HERE ###
      return weighted_loss
  ```
  - DenseNet121
  ```
  # create the base pre-trained model
  base_model = DenseNet121(weights='./nih/densenet.hdf5', include_top=False)

  x = base_model.output

  # add a global spatial average pooling layer
  x = GlobalAveragePooling2D()(x)

  # and a logistic layer
  predictions = Dense(len(labels), activation="sigmoid")(x)

  model = Model(inputs=base_model.input, outputs=predictions)
  model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))
  ```
4. Training
```
history = model.fit_generator(train_generator, 
                              validation_data=valid_generator,
                              steps_per_epoch=100, 
                              validation_steps=25, 
                              epochs = 3)

plt.plot(history.history['loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
plt.show()
```
  - ` model.load_weights("./nih/pretrained_model.h5") `
5. Prediction and Evaluation
```
predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))
auc_rocs = util.get_roc_curve(labels, predicted_vals, test_generator)
```

# ACCURACY
1. ` Accuracy = P(+|disease)P(disease)+P(-|normal)P(normal)`
  - P(+|disease): Sensitivity(true positive rate)
  - P(-|normal): Specificity(true negtive rate)
  - P(disease): Prevalence
  - ` Accuracy = Sensitivity*Prevalence+Specificity*(1-Prevalence)`
2. PPV & NPV
  - PPV: P(disease|+)
  - NPV: P(normal|-)
3. confusion matrix:
                   +                -
  disease   true positive     false negative    → \#(+ and disease)/#disease = sensitvity 
  normal    false positive    true negative     → \#(- and normal)/#normal = specificity
                  ↓                  ↓
  \#(+ and disease)/\#(+) = PPV     \#(- and normal)/\#(-) = NPV

  sensitivity = tp/(tp+fn) 
  specificity = tn/(fp+tn) 
  ppv = tp/(tp+fp)
  npv = tn/(fn+tn)

4. probability
  - ppv = p(pos|pos_p) = p(pos_p|pos)×p(pos)/p(pos_p)
    - sensitivity = p(pos_p|pos)
    - prevalence = p(pos)
      - p(pos_p) = truePos+falsePos 
        - truePos = p(pos_p|pos)×p(pos) = sensitivity×prevalence
        - falsePos = p(pos_p|neg)×p(neg)
          - p(pos_p|neg) = 1-specificity
          - p(neg) = 1-prevalence
  - ppv = sentsitivity×prevalence/(sensitivity×prevalence+(1-specificity)×(1-prevalence))
