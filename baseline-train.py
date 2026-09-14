"""
Epoch: 1 Loss: 2.8424022 Accuracy: 0.12244392 Val Loss: 2.7537029 Val Accuracy: 0.1558551
Epoch: 2 Loss: 2.3752065 Accuracy: 0.24056391 Val Loss: 2.8517673 Val Accuracy: 0.14532435
Epoch: 3 Loss: 2.0518649 Accuracy: 0.3460499 Val Loss: 2.6445508 Val Accuracy: 0.16933446
Epoch: 4 Loss: 1.7996686 Accuracy: 0.43056932 Val Loss: 2.7865498 Val Accuracy: 0.15711878
Test Loss: 1.7745596 Test Accuracy: 0.4466645
"""

import tensorflow as tf

# Load YCB training dataset
train_dataset_path = (
    "/Volumes/AIR_DISK/training-data/object-detection-YCB/"
    "classification_dataset_split/train")
validation_dataset_path = (
    "/Volumes/AIR_DISK/training-data/object-detection-YCB/"
    "classification_dataset_split/validation"
)
test_dataset_path =  (
    "/Volumes/AIR_DISK/training-data/object-detection-YCB/"
    "classification_test_dataset"
)

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dataset_path,
    image_size = (320,320), # (height,width)
    batch_size = 16,
    shuffle = True
) # The shape of x_batch is (batch_size,320,320,3), where 3 represents RGB 3 colors

class_names = train_data.class_names

validation_data = tf.keras.utils.image_dataset_from_directory(
    validation_dataset_path,
    image_size=(320, 320),
    batch_size=16,
    shuffle=False,
    class_names=class_names
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dataset_path,
    image_size = (320,320), # (height,width)
    batch_size = 16,
    shuffle=False,
    class_names=class_names
)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.2),
])

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(320,320,3)),
    
    # Augmentation
    data_augmentation,
    
    # Normalization
    tf.keras.layers.Rescaling(1.0/255.0),
    
    # Define CNN layers
    # image_from_directory() does not need to expand the dimension since
    # it has RGB dimension. CNN expand this dimension to "filter" size
    # CNN 1
    tf.keras.layers.Conv2D(
        filters = 32,
        kernel_size = 3,
        padding = "same",
        kernel_initializer = tf.keras.initializers.HeNormal(seed=1),
        use_bias = True
    ), # (16,320，320，32)
    
    tf.keras.layers.MaxPooling2D(
        pool_size = 2
    ), # (16,160,160,32)
    
    # CNN 2
    tf.keras.layers.Conv2D(
        filters = 64,
        kernel_size = 3,
        padding = "same",
        kernel_initializer = tf.keras.initializers.HeNormal(seed=1),
        use_bias = True
    ), # (16,160，160，64)

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPooling2D(
        pool_size = 2
    ), # (16,80,80,64)
    
    # CNN 3
    tf.keras.layers.Conv2D(
        filters = 128,
        kernel_size = 3,
        padding = "same",
        kernel_initializer = tf.keras.initializers.HeNormal(seed=1),
        use_bias = True
    ), # (16,80，80，128)

    tf.keras.layers.MaxPooling2D(
        pool_size = 2
    ), # (16,40,40,128)
    
    # Global_pool
    # (16,128)
    tf.keras.layers.GlobalAveragePooling2D(),
    
    # Dense 1
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=1)
    ),
    # Dropout
    tf.keras.layers.Dropout(0.5),
    
    # Dense 2
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=1)
    ),
    # Dropout
    tf.keras.layers.Dropout(0.3),
        
    # Output
    tf.keras.layers.Dense(
        units=len(class_names),
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1)
    )
])

# Loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits = True
)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Training loop
for epoch in range(4):
    total_loss = tf.constant(0.0,dtype=tf.float32)
    total = 0
    correct = 0
    
    for x_batch,y_batch in train_data:
        with tf.GradientTape() as tape:
            y_pre = model(x_batch, training=True)
            loss = loss_fn(
                y_batch,y_pre
            )
        
        gradients = tape.gradient(
            loss,
            model.trainable_variables
        )
        optimizer.apply_gradients(
            zip(gradients,model.trainable_variables)
        )
        
        pred = tf.argmax(
            y_pre,
            axis=1,
            output_type = tf.int32
            )

        correct += tf.reduce_sum(
            tf.cast(pred == y_batch,tf.float32)
        )
        total+=tf.size(y_batch)
        total_loss+=loss
                
    accuracy = correct/tf.cast(total,tf.float32)
    
    # Validation
    val_loss = tf.constant(
        0.0,
        dtype=tf.float32
    )

    val_correct = 0
    val_total = 0

    for x_batch, y_batch in validation_data:

        y_pre = model(
            x_batch,
            training=False
        )

        loss = loss_fn(
            y_batch,
            y_pre
        )

        pred = tf.argmax(
            y_pre,
            axis=1,
            output_type=tf.int32
        )

        val_correct += tf.reduce_sum(
            tf.cast(
                pred == y_batch,
                tf.float32
            )
        )

        val_total += tf.size(y_batch)
        val_loss += loss

    val_accuracy = val_correct / tf.cast(
        val_total,
        tf.float32
    )
    
    print(
        "Epoch:",epoch + 1,
        "Loss:",total_loss.numpy() / len(train_data),
        "Accuracy:",accuracy.numpy(),
        "Val Loss:", val_loss.numpy() / len(validation_data),
        "Val Accuracy:", val_accuracy.numpy()
    )
    
# Test loop
test_loss = tf.constant(0.0,dtype=tf.float32)
correct = 0
total = 0

for x_batch, y_batch in test_data:
    
    y_pre = model(x_batch,training=False)
    loss = loss_fn(
        y_batch,
        y_pre
    )
        
    pred = tf.argmax(
        y_pre,
        axis=1,
        output_type = tf.int32
    )
    correct += tf.reduce_sum(
        tf.cast(pred == y_batch,tf.float32)
    )
    total+=tf.size(y_batch)
    test_loss+=loss
        
test_accuracy = correct/tf.cast(total,tf.float32)
        
print(
    "Test Loss:",test_loss.numpy() / len(test_data),
    "Test Accuracy:",test_accuracy.numpy()
)

model.save("ycb_CNNmodel.keras")
