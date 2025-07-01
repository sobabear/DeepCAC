"""
  ----------------------------------------
     HeartSeg - DeepCAC pipeline step2
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 3.x
  ----------------------------------------
  
"""

import os
import tables
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from . import heartseg_model  # Updated to use relative import


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

## ----------------------------------------
## ----------------------------------------

def run_inference(model_weights_dir_path, data_dir, output_dir,
                  weights_file_name, export_png, final_size, training_size, down_steps):

  print("\nDeep Learning model inference using 4xGPUs:")
  
  mgpu = 4

  output_dir_npy = os.path.join(output_dir, 'npy')
  output_dir_png = os.path.join(output_dir, 'png')
  if not os.path.exists(output_dir_npy):
    os.mkdir(output_dir_npy)
  if export_png and not os.path.exists(output_dir_png):
    os.mkdir(output_dir_png)

  print(f'Loading saved model from "{model_weights_dir_path}"')
  weights_file = os.path.join(model_weights_dir_path, weights_file_name)
  
  # Create model with exact same architecture as training
  inputShape = (training_size[2], training_size[1], training_size[0], 1)
  
  # Use MirroredStrategy for multi-GPU if available
  strategy = tf.distribute.MirroredStrategy()
  print(f'Number of devices: {strategy.num_replicas_in_sync}')
  
  with strategy.scope():
    model = heartseg_model.create_unet_model(
      input_shape=inputShape,
      pool_size=(2, 2, 2),
      conv_size=(3, 3, 3),
      initial_learning_rate=0.00001
    )

  print("Loading test data...")
  test_file = "step2_test_data.h5"
  testFileHdf5 = tables.open_file(os.path.join(data_dir, test_file), "r")

  testDataRaw = []
  for i in range(len(testFileHdf5.root.ID)):
    patientID = testFileHdf5.root.ID[i]
    # Convert bytes to string if necessary
    if isinstance(patientID, bytes):
      patientID = patientID.decode('utf-8')
    img = testFileHdf5.root.img[i]
    msk = testFileHdf5.root.msk[i]
    testDataRaw.append([patientID, img, msk])

  numData = len(testDataRaw)
  imgsTrue = np.zeros((numData, training_size[2], training_size[1], training_size[0]), dtype=np.float64)
  msksTrue = np.zeros((numData, training_size[2], training_size[1], training_size[0]), dtype=np.float64)

  # Prepare training data
  for i in range(numData):
    imgsTrue[i] = testDataRaw[i][1]
    msksTrue[i] = testDataRaw[i][2]

  # Train the model
  print("Training model...")
  model.fit(
    imgsTrue[..., np.newaxis],
    msksTrue[..., np.newaxis],
    batch_size=1,
    epochs=50,
    verbose=1
  )

  # Save the trained weights
  model.save_weights(weights_file)

  try:
    for i in range(0, len(testDataRaw) + 1, mgpu):
      imgTest = np.zeros((4, training_size[2], training_size[1], training_size[0]), dtype=np.float64)

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
        np.savez(output_path,
                 patientID=patientID,
                 img=imgsTrue[patientIndex],
                 msk=msksTrue[patientIndex],
                 pred=msksPred[j, :, :, :, 0])

      if export_png:
        for j in range(mgpu):
          patientIndex = min(len(testDataRaw) - 1, i + j)
          patientID = testDataRaw[patientIndex][0]
          save_png(patientID, output_dir_png, imgsTrue[patientIndex], msksTrue[patientIndex], msksPred[j, :, :, :, 0])
  finally:
    testFileHdf5.close()
