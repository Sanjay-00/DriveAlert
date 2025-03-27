# DriveAlert

DriveAlert is a system that can automatically detect a driver drowsiness in a real-time video stream, using computer vision and neural network (deep learning) algorithms.

The classification is based on blinks, yawns, current time, and travel duration. When the driver appears to be drowsy, the system will play an alarm in the car and send a warning email to an emergency contact.

## Folder Structure

The main folder contains `Data` and `Scripts` directories, a `README` file, and an example video.

`Data` folder includes the alarm sound, email warning message, shape predictor, and background and logo images. `Dataset` contains classified images (train, validation and test), and `Model` includes summary and output, confusion matrix and scores, loss and accuracy plots, and the yawn detection model itself.

`Scripts` folder includes the following Python scripts:

The first GUI page is `start_page.py`, which reads the user details and starts the program. The second page is `drowsiness_classification.py`, which detects the driver drowsiness and displays the video frames.

The blink and yawn detection functions are in `blink_score.py` and `yawn_score.py` respectively. The counter thresholds are calculated in `thresholds.py`. The sound and email functions are listed in `drowsiness_alert.py`.

The yawn classification model was built with `convolutional_neural_network_model.py`, and its scores were evaluated using `convolutional_neural_network_metrics.py`.

To start the app, run `start_page.py`.



Epoch 1/50
loss: 1.0784 - accuracy: 0.5098 - val_loss: 0.6843 - val_accuracy: 0.6558
Epoch 2/50
loss: 0.6868 - accuracy: 0.5727 - val_loss: 0.6115 - val_accuracy: 0.6605
Epoch 3/50
loss: 0.6675 - accuracy: 0.5678 - val_loss: 0.6221 - val_accuracy: 0.5395
Epoch 4/50
loss: 0.6073 - accuracy: 0.5943 - val_loss: 0.5611 - val_accuracy: 0.6837
Epoch 5/50
loss: 0.5686 - accuracy: 0.6768 - val_loss: 0.5584 - val_accuracy: 0.6605
Epoch 6/50
loss: 0.5691 - accuracy: 0.6572 - val_loss: 0.5433 - val_accuracy: 0.6605
Epoch 7/50
loss: 0.5510 - accuracy: 0.6798 - val_loss: 0.5279 - val_accuracy: 0.6837
Epoch 8/50
loss: 0.5446 - accuracy: 0.6945 - val_loss: 0.6382 - val_accuracy: 0.6465
Epoch 9/50
loss: 0.5303 - accuracy: 0.6935 - val_loss: 0.5057 - val_accuracy: 0.7163
Epoch 10/50
loss: 0.5204 - accuracy: 0.7092 - val_loss: 0.5051 - val_accuracy: 0.7302
Epoch 11/50
loss: 0.5096 - accuracy: 0.7141 - val_loss: 0.4807 - val_accuracy: 0.7535
Epoch 12/50
loss: 0.5044 - accuracy: 0.7230 - val_loss: 0.4760 - val_accuracy: 0.7023
Epoch 13/50
loss: 0.4962 - accuracy: 0.7210 - val_loss: 0.4479 - val_accuracy: 0.7674
Epoch 14/50
loss: 0.4804 - accuracy: 0.7583 - val_loss: 0.4598 - val_accuracy: 0.7395
Epoch 15/50
loss: 0.4773 - accuracy: 0.7642 - val_loss: 0.4533 - val_accuracy: 0.7581
Epoch 16/50
loss: 0.4749 - accuracy: 0.7613 - val_loss: 0.4689 - val_accuracy: 0.7535
Epoch 17/50
loss: 0.4424 - accuracy: 0.7868 - val_loss: 0.4184 - val_accuracy: 0.7674
Epoch 18/50
loss: 0.4347 - accuracy: 0.7967 - val_loss: 0.4023 - val_accuracy: 0.7953
Epoch 19/50
loss: 0.4092 - accuracy: 0.8075 - val_loss: 0.4693 - val_accuracy: 0.7767
Epoch 20/50
loss: 0.4115 - accuracy: 0.7927 - val_loss: 0.3878 - val_accuracy: 0.7953
Epoch 21/50
loss: 0.4012 - accuracy: 0.8075 - val_loss: 0.4118 - val_accuracy: 0.8186
Epoch 22/50
loss: 0.3778 - accuracy: 0.8153 - val_loss: 0.3389 - val_accuracy: 0.8512
Epoch 23/50
loss: 0.3502 - accuracy: 0.8350 - val_loss: 0.3190 - val_accuracy: 0.8558
Epoch 24/50
loss: 0.3337 - accuracy: 0.8458 - val_loss: 0.3100 - val_accuracy: 0.8651
Epoch 25/50
loss: 0.2968 - accuracy: 0.8772 - val_loss: 0.3596 - val_accuracy: 0.8465
Epoch 27/50
loss: 0.2837 - accuracy: 0.8831 - val_loss: 0.2930 - val_accuracy: 0.8837
Epoch 28/50
loss: 0.2680 - accuracy: 0.8900 - val_loss: 0.2599 - val_accuracy: 0.8884
Epoch 29/50
loss: 0.2637 - accuracy: 0.8841 - val_loss: 0.2576 - val_accuracy: 0.8930
Epoch 30/50
loss: 0.2517 - accuracy: 0.8969 - val_loss: 0.2383 - val_accuracy: 0.8977
Epoch 31/50
loss: 0.2427 - accuracy: 0.8978 - val_loss: 0.2696 - val_accuracy: 0.8930
Epoch 32/50
loss: 0.2597 - accuracy: 0.8910 - val_loss: 0.2763 - val_accuracy: 0.8837
Epoch 33/50
loss: 0.2607 - accuracy: 0.8811 - val_loss: 0.2657 - val_accuracy: 0.8791
Epoch 34/50
loss: 0.2142 - accuracy: 0.9077 - val_loss: 0.2231 - val_accuracy: 0.9023
Epoch 35/50
loss: 0.2039 - accuracy: 0.9293 - val_loss: 0.1907 - val_accuracy: 0.9302
Epoch 36/50
loss: 0.1823 - accuracy: 0.9263 - val_loss: 0.2086 - val_accuracy: 0.9116
Epoch 37/50
loss: 0.1684 - accuracy: 0.9352 - val_loss: 0.1818 - val_accuracy: 0.9256
Epoch 38/50
loss: 0.1781 - accuracy: 0.9253 - val_loss: 0.2581 - val_accuracy: 0.9023
Epoch 39/50
loss: 0.2025 - accuracy: 0.9204 - val_loss: 0.1925 - val_accuracy: 0.9209
Epoch 40/50
loss: 0.1603 - accuracy: 0.9401 - val_loss: 0.1622 - val_accuracy: 0.9256
Epoch 41/50
loss: 0.1522 - accuracy: 0.9430 - val_loss: 0.1842 - val_accuracy: 0.9302
Epoch 42/50
loss: 0.1439 - accuracy: 0.9450 - val_loss: 0.1987 - val_accuracy: 0.9163
Epoch 43/50
loss: 0.1409 - accuracy: 0.9391 - val_loss: 0.1981 - val_accuracy: 0.9163
Epoch 44/50
loss: 0.1219 - accuracy: 0.9538 - val_loss: 0.2209 - val_accuracy: 0.9163
Epoch 45/50
loss: 0.1365 - accuracy: 0.9528 - val_loss: 0.1855 - val_accuracy: 0.9349
Epoch 46/50
loss: 0.1308 - accuracy: 0.9528 - val_loss: 0.1994 - val_accuracy: 0.9302
Epoch 47/50
loss: 0.1346 - accuracy: 0.9519 - val_loss: 0.2014 - val_accuracy: 0.9256
Epoch 48/50
loss: 0.1299 - accuracy: 0.9509 - val_loss: 0.2076 - val_accuracy: 0.9302
Epoch 49/50
loss: 0.1573 - accuracy: 0.9411 - val_loss: 0.1886 - val_accuracy: 0.9488
Epoch 50/50
loss: 0.1163 - accuracy: 0.9509 - val_loss: 0.2158 - val_accuracy: 0.9302
