import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

shapes = ["AEI","L","O","CDGLMSTXYZ","FV","QW","BMP","U","E","R","TH","ChJSh","SMILE","FROWN","TOUT","OFFSET"]
training_dir = './data/'
image_size = (100, 100)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=.2
        )
validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=.2
        )



train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size = image_size,
        subset="training",
        batch_size=32,
        class_mode='sparse',
        seed=42,shuffle=True)
validation_generator = validation_datagen.flow_from_directory(
        training_dir,
        target_size=image_size,
        batch_size=32,
        class_mode='sparse',
        subset="validation",
        seed=42)

test_optimizers = [
    'sgd',
    'adam',
    'adamax'
]
models = []

# Build models (on 3 different optimizers) and add them to the array of models
for opt in test_optimizers:
    # Print which model it is working on
    print(f'\n\nModel with {opt} optimizer and sparse_categorical_crossentropy loss function:')

    # Build the model
    model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
                include_top=True, weights=None, input_tensor=None,
                input_shape=(100,100,3), pooling='max', classes=len(shapes),
                classifier_activation='softmax')

    # Use early stopping with default parameters
    early_stopping = EarlyStopping()

    # Compile the model
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Fit the model to the training data and validate in on the validation set
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=[early_stopping]
    )

    # Add model to the list
    models.append(model)

test_dir = '/content/'

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        classes=['test'],
        target_size=image_size,
        class_mode='sparse',
        shuffle=False)

model_predictions = []
for i in range(len(models)):
    probabilities = models[i].predict(test_generator)
    predictions = [np.argmax(probas) for probas in probabilities]
    model_predictions.append(predictions)




print('                                  precision   recall   f1-score   count')
precisions = []
recalls = []
f1_scores = []

# Print out report (for best model) with class names instead of nunbers
for i in range(len(working_report)):
    # Class Name
    if (i) % 5 == 0:
        print(class_names[int(working_report[i])].ljust(34), end='')
    
    # Precision
    if (i - 1) % 5 == 0:
        print(str(int(float(working_report[i])*100)).rjust(8), end='%')
        precisions.append(float(working_report[i]))

    # Recall
    if (i - 2) % 5 == 0:
        print(str(int(float(working_report[i])*100)).rjust(8), end='%')
        recalls.append(float(working_report[i]))
    
    # F1-score
    if (i - 3) % 5 == 0:
        print(str(int(float(working_report[i])*100)).rjust(8), end='%')
        f1_scores.append(float(working_report[i]))
    
    # Count
    if (i - 4) % 5 == 0:
        print(working_report[i].rjust(10), end='\n')

print(f'Average Precision: {round(100*(sum(precisions)/len(precisions)))}%')
print(f'Average Recall:    {round(100*(sum(recalls)/len(recalls)))}%')
print(f'Average F1-score:  {round(100*(sum(f1_scores)/len(f1_scores)))}%')