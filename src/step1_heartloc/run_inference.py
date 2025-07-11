"""
  ----------------------------------------
     HeartLoc - DeepCAC pipeline step1
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 3.x
  ----------------------------------------
  
"""

import os
import sys
import tables
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.ndimage import measurements
from . import heartloc_model  # Updated import to use relative import

def save_png(patientID, output_dir_png, img, msk, pred):
  maskIndicesMsk = np.where(msk != 0)
  if len(maskIndicesMsk) == 0:
    trueBB = [np.min(maskIndicesMsk[0]), np.max(maskIndicesMsk[0]),
              np.min(maskIndicesMsk[1]), np.max(maskIndicesMsk[1]),
              np.min(maskIndicesMsk[2]), np.max(maskIndicesMsk[2])]
    cen = [trueBB[0] + (trueBB[1] - trueBB[0]) / 2,
           trueBB[2] + (trueBB[3] - trueBB[2]) / 2,
           trueBB[4] + (trueBB[5] - trueBB[4]) / 2]
  else:
    cen = [int(len(img) / 2), int(len(img) / 2), int(len(img) / 2)]

  pred[pred > 0.5] = 1
  pred[pred < 1] = 0

  fig, ax = plt.subplots(2, 3, figsize=(32, 16))
  ax[0, 0].imshow(img[cen[0], :, :], cmap='gray')
  ax[0, 1].imshow(img[:, cen[1], :], cmap='gray')
  ax[0, 2].imshow(img[:, :, cen[2]], cmap='gray')

  ax[0, 0].imshow(msk[cen[0], :, :], cmap='jet', alpha=0.4)
  ax[0, 1].imshow(msk[:, cen[1], :], cmap='jet', alpha=0.4)
  ax[0, 2].imshow(msk[:, :, cen[2]], cmap='jet', alpha=0.4)

  ax[1, 0].imshow(img[cen[0], :, :], cmap='gray')
  ax[1, 1].imshow(img[:, cen[1], :], cmap='gray')
  ax[1, 2].imshow(img[:, :, cen[2]], cmap='gray')

  ax[1, 0].imshow(pred[cen[0], :, :], cmap='jet', alpha=0.4)
  ax[1, 1].imshow(pred[:, cen[1], :], cmap='jet', alpha=0.4)
  ax[1, 2].imshow(pred[:, :, cen[2]], cmap='jet', alpha=0.4)

  fileName = os.path.join(output_dir_png, patientID + '_' + ".png")
  plt.savefig(fileName)
  plt.close(fig)


def save_prediction(output_path, patientID, img, msk, pred):
  """Save prediction results in a structured format"""
  # Save each array separately with clear naming
  np.savez_compressed(
    output_path,
    patient_id=np.array([patientID]),  # Save as a single-element array
    image=img,
    mask=msk,
    prediction=pred
  )


def test(model, dataDir, output_dir_npy, output_dir_png, pkl_file,
         test_file, weights_file, mgpu, has_manual_seg, png):
    
  # Try to load the saved model directly first
  try:
    saved_model = tf.keras.models.load_model(weights_file, custom_objects={
        'dice_coef_loss': heartloc_model.dice_coef_loss,
        'dice_coef': heartloc_model.dice_coef
    })
    model = saved_model
  except:
    print("Could not load as full model, trying to load weights only...")
    try:
      model.load_weights(weights_file, by_name=True)
    except:
      print("Failed to load weights. Please check if the weights file is compatible.")
      return

  testFileHdf5 = tables.open_file(os.path.join(dataDir, test_file), "r")
  with open(os.path.join(dataDir, pkl_file), 'rb') as f:  # Updated pickle loading
    pklData = pickle.load(f)

  # Get data in one list for further processing
  testDataRaw = []
  num_test_imgs = len(testFileHdf5.root.ID)
  
  # Convert all pkl keys to strings if they're bytes
  if any(isinstance(k, bytes) for k in pklData.keys()):
    pklData = {k.decode('utf-8') if isinstance(k, bytes) else k: v for k, v in pklData.items()}
  
  for i in range(num_test_imgs):
    patientID = testFileHdf5.root.ID[i]
    # Convert bytes to string if necessary
    if isinstance(patientID, bytes):
      patientID = patientID.decode('utf-8')
      
    img = testFileHdf5.root.img[i]
    if has_manual_seg:
      msk = testFileHdf5.root.msk[i]
    else:  # Create empty dummy has_manual_seg with same size as the image
      sizeImg = len(img)
      msk = np.zeros((sizeImg, sizeImg, sizeImg), dtype=np.float64)
    
    if patientID not in pklData:
      print(f'Patient {patientID} not found in pkl data')
      continue
      
    zDif = pklData[patientID][6][2]
    testDataRaw.append([patientID, img, msk, zDif])

  if not testDataRaw:
    print("Error: No valid test data found. Please check patient IDs and pkl data.")
    testFileHdf5.close()
    return

  numData = len(testDataRaw)
  size = len(testDataRaw[0][1])
  imgsTrue = np.zeros((numData, size, size, size), dtype=np.float64)
  msksTrue = np.zeros((numData, size, size, size), dtype=np.float64)

  try:
    for i in range(0, len(testDataRaw) + 1, mgpu):  # Updated xrange to range
      imgTest = np.zeros((4, size, size, size), dtype=np.float64)

      for j in range(mgpu):
        # If the number of test images is not mod 4 == 0, just redo the last file severall times
        patientIndex = min(len(testDataRaw) - 1, i + j)
        patientID = testDataRaw[patientIndex][0]
        print(f'Processing patient {patientID}')
        # Store data for score calculation
        imgsTrue[patientIndex, :, :, :] = testDataRaw[patientIndex][1]
        msksTrue[patientIndex, :, :, :] = testDataRaw[patientIndex][2]
        imgTest[j, :, :, :] = testDataRaw[patientIndex][1]

      msksPred = model.predict(imgTest[:, :, :, :, np.newaxis])

      for j in range(mgpu):
        patientIndex = min(len(testDataRaw) - 1, i + j)
        patientID = testDataRaw[patientIndex][0]
        output_path = os.path.join(output_dir_npy, patientID + '_pred.npz')
        save_prediction(
          output_path,
          patientID,
          imgsTrue[patientIndex],
          msksTrue[patientIndex],
          msksPred[j, :, :, :, 0]
        )

      if png:
        for j in range(mgpu):
          patientIndex = min(len(testDataRaw) - 1, i + j)
          patientID = testDataRaw[patientIndex][0]
          save_png(patientID, output_dir_png, imgsTrue[patientIndex], msksTrue[patientIndex], msksPred[j, :, :, :, 0])
  finally:
    testFileHdf5.close()


def run_inference(model_output_dir_path, model_input_dir_path, model_weights_dir_path,
                  crop_size, export_png, model_down_steps, extended, has_manual_seg, weights_file_name):

  print("\nDeep Learning model inference using 4xGPUs:")
  
  mgpu = 4

  output_dir_npy = os.path.join(model_output_dir_path, 'npy')
  output_dir_png = os.path.join(model_output_dir_path, 'png')
  if not os.path.exists(output_dir_npy):
    os.mkdir(output_dir_npy)
  if export_png and not os.path.exists(output_dir_png):
    os.mkdir(output_dir_png)

  test_file = "step1_test_data.h5"
  pkl_file = "step1_downsample_results.pkl"

  weights_file = os.path.join(model_weights_dir_path, weights_file_name)

  print(f'Loading saved model from "{weights_file}"')
  
  # If crop_size is a list, use the first element
  if isinstance(crop_size, list):
    cube_size = crop_size[0]
  else:
    cube_size = crop_size
  
  input_shape = (cube_size, cube_size, cube_size, 1)
  model = heartloc_model.get_unet_3d(down_steps = model_down_steps,
                                     input_shape = input_shape,
                                     mgpu = mgpu,
                                     ext = extended)

  test(model, model_input_dir_path, output_dir_npy, output_dir_png,
       pkl_file, test_file, weights_file, mgpu, has_manual_seg, export_png)
