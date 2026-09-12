"""

"""

import tensorflow as tf

# Load YCB training dataset
train_dataset_path = "/Volumes/AIR_DISK/training-data/object-detection-YCB/classification_dataset_split/train"
test_dataset_path =  "/Volumes/AIR_DISK/training-data/object-detection-YCB/classification_dataset_split/test"
validation_dataset_path = "/Volumes/AIR_DISK/training-data/object-detection-YCB/classification_dataset_split/validation"


train_data = tf.keras.utils.image_dataset_from_directory(
    train_dataset_path,
    image_size = (320,320), # (height,width)
    batch_size = 32,
    shuffle = True
) # The shape of x_batch is (32,320,320,3), where 3 represents RGB 3 colors

class_names = train_data.class_names

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dataset_path,
    image_size = (320,320), # (height,width)
    batch_size = 32,
    shuffle=False,
    class_names=class_names
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    validation_dataset_path,
    image_size=(320, 320),
    batch_size=32,
    shuffle=False,
    class_names=class_names
)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(320,320,3)),
    
    # Normalization
    tf.keras.layers.Rescaling(1.0/255.0),
    
    # Define CNN layers
    # image_from_directory() does not need to expand the dimension since
    # it has RGB dimension. CNN expand this dimension to "filter" size
    # CNN 1
    tf.keras.layers.Conv2D(
        filters = 32,
        kernel_size = 3,
        activation = "relu",
        padding = "same",
        kernel_initializer = tf.keras.initializers.HeNormal(seed=1),
        use_bias = True
    ), # (32,320，320，32)
    
    tf.keras.layers.MaxPooling2D(
        pool_size = 2
    ), # (32,160,160,32)
    
    # CNN 2
    tf.keras.layers.Conv2D(
        filters = 64,
        kernel_size = 3,
        activation = "relu",
        padding = "same",
        kernel_initializer = tf.keras.initializers.HeNormal(seed=1),
        use_bias = True
    ), # (32,160，160，64)

    tf.keras.layers.MaxPooling2D(
        pool_size = 2
    ), # (32,80,80,64)
    
    # CNN 3
    tf.keras.layers.Conv2D(
        filters = 128,
        kernel_size = 3,
        activation = "relu",
        padding = "same",
        kernel_initializer = tf.keras.initializers.HeNormal(seed=1),
        use_bias = True
    ), # (32,80，80，128)

    tf.keras.layers.MaxPooling2D(
        pool_size = 2
    ), # (32,40,40,128)
    
    # Global_pool
    # (32,128)
    tf.keras.layers.GlobalAveragePooling2D(),
    
    # Dense 1
    tf.keras.layers.Dense(
        units=64,
        activation="relu",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=1)
    ),
    
    # Dense 2
    tf.keras.layers.Dense(
        units=32,
        activation="relu",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=1)
    ),
        
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
