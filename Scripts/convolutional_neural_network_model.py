
# # Neural network model for the yawn classification


# # import packages
# from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import ModelCheckpoint
# from keras.models import Sequential
# import matplotlib.pyplot as plt
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # dataset paths
# train_data_path = "Data/Dataset/train"
# validation_data_path = "Data/Dataset/validation"

# # define an image generator to format images before using them
# train_data_generator = ImageDataGenerator(horizontal_flip=True, rescale=1./255, zoom_range=0.2, validation_split=0.1)
# validation_data_generator = ImageDataGenerator(rescale=1./255)

# # pre-process images using the image generator
# train_set = train_data_generator.flow_from_directory(train_data_path, target_size=(256, 256), batch_size=128, color_mode='grayscale', class_mode='categorical')
# validation_set = validation_data_generator.flow_from_directory(validation_data_path, target_size=(256, 256), batch_size=128, color_mode='grayscale', class_mode='categorical')

# # model layers building: create a sequence of layers by adding one layer at a time until the network architecture is satisfying
# model = Sequential()

# # The activation function decides whether a neuron should be activated or not, by calculating weighted sum and further adding bias with it
# # Max-Pooling is used to reduce the spatial dimensions of an model_output.txt volume

# # the first layer (input): learning 32 filters
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256, 256, 1), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # the second layer: learning 64 filters
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # the third layer: learning 128 filters
# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # flatten the multi-dimensional input tensors into a single dimension
# model.add(Flatten())

# # add dense layer, which is a connected deeply neural network layer
# model.add(Dense(64, activation='relu'))
# classes = 2
# model.add(Dense(classes, activation='softmax'))


# # summarize the model and capture the summary in a string
# summary_str = []
# model.summary(print_fn=lambda x: summary_str.append(x))

# # # Write the summary to a file
# # with open("Data/Model/model_summary5.txt", "w") as file:
# #     file.write("\n".join(summary_str))

# with open("Data/Model/summary.txt", "w", encoding="utf-8") as file:
#     file.write("\n".join(summary_str))

# with open("Data/Model/summary5.txt", "w",encoding="utf-8") as file:
#     model.summary(print_fn=lambda x: file.write(x + '\n'))

# # configure the learning process before training the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # define the model path
# model_path = "Data/Model/yawn_detection.h5"

# # modelCheckpoint: a callback object that can perform actions at various stages of the training, and can monitor either the accuracy or the loss
# checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_format='h5')
# # checkpoint = ModelCheckpoint(model_path, monitor='val_loss',save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

# # define the number of epochs, which is an arbitrary cutoff, used to separate training into distinct phases, which is useful for logging and periodic evaluation
# num_epochs = 50

# # train the model
# history = model.fit_generator(train_set, epochs=num_epochs, validation_data=validation_set, callbacks=callbacks_list)

# # plot loss and accuracy
# plt.figure(figsize=(20, 10))
# plt.suptitle('Optimizer : Adam', fontsize=10)

# plt.subplot(1, 2, 1)
# plt.ylabel('Loss', fontsize=16)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend(loc='upper right')

# plt.subplot(1, 2, 2)
# plt.ylabel('Accuracy', fontsize=16)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.legend(loc='lower right')

# # # save and show plots
# plt.savefig("Data/Model/loss_and_accuracy_plot33.png")
# plt.show()
# # # Neural network model for the yawn classification

# # # Neural network model for the yawn classification

# # # import packages
# # # from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# # # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # # from keras.callbacks import ModelCheckpoint
# # # from keras.models import Sequential
# # # import matplotlib.pyplot as plt

# # # # dataset paths
# # # train_data_path = "Data/Dataset/train"
# # # validation_data_path = "Data/Dataset/validation"

# # # # define an image generator to format images before using them
# # # train_data_generator = ImageDataGenerator(horizontal_flip=True, rescale=1./255, zoom_range=0.2, validation_split=0.1)
# # # validation_data_generator = ImageDataGenerator(rescale=1./255)

# # # # pre-process images using the image generator
# # # train_set = train_data_generator.flow_from_directory(train_data_path, target_size=(256, 256), batch_size=128, color_mode='grayscale', class_mode='categorical')
# # # validation_set = validation_data_generator.flow_from_directory(validation_data_path, target_size=(256, 256), batch_size=128, color_mode='grayscale', class_mode='categorical')

# # # # model layers building: create a sequence of layers by adding one layer at a time until the network architecture is satisfying
# # # model = Sequential()

# # # # The activation function decides whether a neuron should be activated or not, by calculating weighted sum and further adding bias with it
# # # # Max-Pooling is used to reduce the spatial dimensions of an model_output.txt volume

# # # # the first layer (input): learning 32 filters
# # # model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256, 256, 1), activation='relu'))
# # # model.add(MaxPooling2D(pool_size=(2, 2)))

# # # # the second layer: learning 64 filters
# # # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# # # model.add(MaxPooling2D(pool_size=(2, 2)))

# # # # the third layer: learning 128 filters
# # # model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# # # model.add(MaxPooling2D(pool_size=(2, 2)))

# # # # flatten the multi-dimensional input tensors into a single dimension
# # # model.add(Flatten())

# # # # add dense layer, which is a connected deeply neural network layer
# # # model.add(Dense(64, activation='relu'))
# # # classes = 2
# # # model.add(Dense(classes, activation='softmax'))

# # # # configure the learning process before training the model
# # # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # # # define the model path
# # # model_path = "Data/Model/yawn_detection3.keras"  # Change to .keras extension

# # # # modelCheckpoint: a callback object that can perform actions at various stages of the training, and can monitor either the accuracy or the loss
# # # checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')  # Removed save_format argument
# # # callbacks_list = [checkpoint]

# # # # define the number of epochs, which is an arbitrary cutoff, used to separate training into distinct phases, which is useful for logging and periodic evaluation
# # # num_epochs = 50

# # # # train the model
# # # history = model.fit(train_set, epochs=num_epochs, validation_data=validation_set, callbacks=callbacks_list)  # Use fit instead of fit_generator

# # # # plot loss and accuracy
# # # plt.figure(figsize=(20, 10))
# # # plt.suptitle('Optimizer : Adam', fontsize=10)

# # # plt.subplot(1, 2, 1)
# # # plt.ylabel('Loss', fontsize=16)
# # # plt.plot(history.history['loss'], label='Training Loss')
# # # plt.plot(history.history['val_loss'], label='Validation Loss')
# # # plt.legend(loc='upper right')

# # # plt.subplot(1, 2, 2)
# # # plt.ylabel('Accuracy', fontsize=16)
# # # plt.plot(history.history['accuracy'], label='Training Accuracy')
# # # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# # # plt.legend(loc='lower right')

# # # # save and show plots
# # # plt.savefig("Data/Model/loss_and_accuracy_plot2.png")
# # # plt.show()




# # Neural network model for the yawn classification using Transfer Learning with MobileNetV2

# # Import necessary packages
# # from keras.layers import Dense, Flatten, GlobalAveragePooling2D
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from keras.callbacks import ModelCheckpoint
# # from keras.models import Model
# # from tensorflow.keras.applications import MobileNetV2
# # import matplotlib.pyplot as plt

# # # Dataset paths
# # train_data_path = "Data/Dataset/train"
# # validation_data_path = "Data/Dataset/validation"

# # # Define an image generator to format images before using them
# # train_data_generator = ImageDataGenerator(horizontal_flip=True, rescale=1./255, zoom_range=0.2, validation_split=0.1)
# # validation_data_generator = ImageDataGenerator(rescale=1./255)

# # # Pre-process images using the image generator
# # train_set = train_data_generator.flow_from_directory(train_data_path, target_size=(224, 224), batch_size=128, color_mode='rgb', class_mode='categorical')
# # validation_set = validation_data_generator.flow_from_directory(validation_data_path, target_size=(224, 224), batch_size=128, color_mode='rgb', class_mode='categorical')

# # # Load the MobileNetV2 model pre-trained on ImageNet
# # base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # # Freeze the layers of MobileNetV2 to retain the learned features from ImageNet
# # base_model.trainable = False

# # # Create the model architecture by adding custom layers on top
# # model = Model(inputs=base_model.input, outputs=base_model.output)

# # # Add custom layers on top of MobileNetV2
# # x = GlobalAveragePooling2D()(model.output)  # Use GlobalAveragePooling2D instead of Flatten for better performance
# # x = Dense(128, activation='relu')(x)
# # classes = 2
# # x = Dense(classes, activation='softmax')(x)

# # # Final model
# # model = Model(inputs=base_model.input, outputs=x)

# # # Summarize the model and capture the summary in a string
# # summary_str = []
# # model.summary(print_fn=lambda x: summary_str.append(x))

# # # Write the summary to a file
# # with open("Data/Model/model_summary2.txt", "w") as file:
# #     file.write("\n".join(summary_str))

# # # Configure the learning process before training the model
# # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # # Define the model path
# # model_path = "Data/Model/yawn_detection2.h5"

# # # ModelCheckpoint: a callback object to monitor validation accuracy and save the best model
# # checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# # callbacks_list = [checkpoint]

# # # Define the number of epochs
# # num_epochs = 50

# # # Train the model
# # history = model.fit(train_set, epochs=num_epochs, validation_data=validation_set, callbacks=callbacks_list)

# # # Plot loss and accuracy
# # plt.figure(figsize=(20, 10))
# # plt.suptitle('Optimizer : Adam', fontsize=10)

# # plt.subplot(1, 2, 1)
# # plt.ylabel('Loss', fontsize=16)
# # plt.plot(history.history['loss'], label='Training Loss')
# # plt.plot(history.history['val_loss'], label='Validation Loss')
# # plt.legend(loc='upper right')

# # plt.subplot(1, 2, 2)
# # plt.ylabel('Accuracy', fontsize=16)
# # plt.plot(history.history['accuracy'], label='Training Accuracy')
# # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# # plt.legend(loc='lower right')

# # # Save and show plots
# # plt.savefig("Data/Model/loss_and_accuracy_plot2.png")
# # plt.show()

# Neural network model for the yawn classification


# # import packages
# from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import ModelCheckpoint
# from keras.models import Sequential
# import matplotlib.pyplot as plt

# # dataset paths
# train_data_path = "Data/Dataset/train"
# validation_data_path = "Data/Dataset/validation"

# # define an image generator to format images before using them
# train_data_generator = ImageDataGenerator(horizontal_flip=True, rescale=1./255, zoom_range=0.2, validation_split=0.1)
# validation_data_generator = ImageDataGenerator(rescale=1./255)

# # pre-process images using the image generator
# train_set = train_data_generator.flow_from_directory(train_data_path, target_size=(256, 256), batch_size=128, color_mode='grayscale', class_mode='categorical')
# validation_set = validation_data_generator.flow_from_directory(validation_data_path, target_size=(256, 256), batch_size=128, color_mode='grayscale', class_mode='categorical')

# # model layers building: create a sequence of layers by adding one layer at a time until the network architecture is satisfying
# model = Sequential()

# # The activation function decides whether a neuron should be activated or not, by calculating weighted sum and further adding bias with it
# # Max-Pooling is used to reduce the spatial dimensions of an model_output.txt volume

# # the first layer (input): learning 32 filters
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=(256, 256, 1), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # the second layer: learning 64 filters
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # the third layer: learning 128 filters
# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # flatten the multi-dimensional input tensors into a single dimension
# model.add(Flatten())

# # add dense layer, which is a connected deeply neural network layer
# model.add(Dense(64, activation='relu'))
# classes = 2
# model.add(Dense(classes, activation='softmax'))

# # summarize the model
# with open("Data/Model/summary.txt", "w",encoding="utf-8") as file:
#     model.summary(print_fn=lambda x: file.write(x + '\n'))

# # configure the learning process before training the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # define the model path
# model_path = "Data/Model/yawn_detection.weights.h5"

# # modelCheckpoint: a callback object that can perform actions at various stages of the training, and can monitor either the accuracy or the loss
# checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
# # checkpoint = ModelCheckpoint(model_path, monitor='val_loss',save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

# # define the number of epochs, which is an arbitrary cutoff, used to separate training into distinct phases, which is useful for logging and periodic evaluation
# num_epochs = 50

# # train the model
# history = model.fit(train_set, epochs=num_epochs, validation_data=validation_set, callbacks=callbacks_list)

# # plot loss and accuracy
# plt.figure(figsize=(20, 10))
# plt.suptitle('Optimizer : Adam', fontsize=10)

# plt.subplot(1, 2, 1)
# plt.ylabel('Loss', fontsize=16)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend(loc='upper right')

# plt.subplot(1, 2, 2)
# plt.ylabel('Accuracy', fontsize=16)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.legend(loc='lower right')

# # save and show plots
# plt.savefig("Data/Model/loss_and_accuracy_plot5.png")
# plt.show()
















# Import packages
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Dataset paths
train_data_path = "Data/Datasett/train"
val_data_path = "Data/Datasett/validation"  # Path to separate validation dataset

# Define image generators
train_data_generator = ImageDataGenerator(horizontal_flip=True, rescale=1./255, zoom_range=0.2)

# Separate generator for validation (only rescaling)
val_data_generator = ImageDataGenerator(rescale=1./255)

# Load training set
train_set = train_data_generator.flow_from_directory(
    train_data_path, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical'
)

# Load validation set from separate dataset
validation_set = val_data_generator.flow_from_directory(
    val_data_path, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical'
)

# Load MobileNetV2 as base model (exclude top layers)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze first 100 layers to retain learned features
for layer in base_model.layers[:100]:  
    layer.trainable = False  

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduce dimensions
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)  # Prevent overfitting
x = Dense(64, activation='relu')(x)

# Dynamically detect number of classes
num_classes = len(train_set.class_indices)
x = Dense(num_classes, activation='softmax')(x)  # Output layer

# Define the final model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Model checkpoint to save best model
model_path = "Data/Model/yawn_detection_mobilenet2.keras"
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=False)
callbacks_list = [checkpoint]

# Define number of epochs
num_epochs = 30

# Train the model
history = model.fit(train_set, epochs=num_epochs, validation_data=validation_set, callbacks=callbacks_list)

# Plot loss and accuracy
plt.figure(figsize=(20, 10))
plt.suptitle('Transfer Learning with MobileNetV2', fontsize=10)

plt.subplot(1, 2, 1)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')

# Save and show plots
plt.savefig("Data/Model/loss_and_accuracy_mobilenet2.png")
plt.show()
