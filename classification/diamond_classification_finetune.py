import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os




#Features for training the model
IMG_SIZE = 380
BATCH_SIZE = 32
EPOCHS_INITIAL = 5
EPOCHS_FINE = 5

TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'

#Training the data generator (inspired by datagen paper)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    brightness_range=[0.5, 1.5], 
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

#Validation of data generator
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)



#FINETUNING THE MODEL
def build_finetune_model(num_classes):

    base_model = EfficientNetB3(weights='imagenet', include_top=False, 
                                input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

    base_model.trainable = False


    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

num_classes = train_generator.num_classes
model = build_finetune_model(num_classes)


#Compiling for the Initial Training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])




print("Starting training")
history_initial = model.fit(
    train_generator,
    epochs=EPOCHS_INITIAL,
    validation_data=validation_generator
)


#Fine-Tuning the Model
print("Starting fine-tune")

for layer in model.layers[-20:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    epochs=EPOCHS_FINE,
    validation_data=validation_generator
)


#Final Check and Checking the Accuracy of Model
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=1)
print(f"Model Accuracy: {val_accuracy * 100:.2f}%")
