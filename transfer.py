"""

"""

import tensorflow as tf

IMAGE_SIZE = 320
BATCH_SIZE = 16
EPOCHS = 6

train_dataset_path = (
    "/Volumes/AIR_DISK/training-data/object-detection-YCB/"
    "classification_dataset_split/train"
)

validation_dataset_path = (
    "/Volumes/AIR_DISK/training-data/object-detection-YCB/"
    "classification_dataset_split/validation"
)

test_dataset_path = (
    "/Volumes/AIR_DISK/training-data/object-detection-YCB/"
    "classification_test_dataset"
)

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dataset_path,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = train_data.class_names

validation_data = tf.keras.utils.image_dataset_from_directory(
    validation_dataset_path,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=class_names
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dataset_path,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=class_names
)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.2),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(
    units=128,
    activation="relu",
    kernel_initializer=tf.keras.initializers.HeNormal(seed=1)
)(x)

x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(
    units=64,
    activation="relu",
    kernel_initializer=tf.keras.initializers.HeNormal(seed=1)
)(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    units=len(class_names),
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=1)
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

for epoch in range(EPOCHS):

    total_loss = tf.constant(
        0.0,
        dtype=tf.float32
    )

    total = 0

    correct = tf.constant(
        0.0,
        dtype=tf.float32
    )

    for x_batch, y_batch in train_data:

        with tf.GradientTape() as tape:

            y_pre = model(
                x_batch,
                training=True
            )

            loss = loss_fn(
                y_batch,
                y_pre
            )

        gradients = tape.gradient(
            loss,
            model.trainable_variables
        )

        optimizer.apply_gradients(
            zip(
                gradients,
                model.trainable_variables
            )
        )

        pred = tf.argmax(
            y_pre,
            axis=1,
            output_type=tf.int32
        )

        correct += tf.reduce_sum(
            tf.cast(
                pred == y_batch,
                tf.float32
            )
        )

        total += tf.size(y_batch)

        total_loss += loss

    train_accuracy = correct / tf.cast(
        total,
        tf.float32
    )

    train_loss = (
        total_loss.numpy()
        /
        len(train_data)
    )

    val_loss = tf.constant(
        0.0,
        dtype=tf.float32
    )

    val_correct = tf.constant(
        0.0,
        dtype=tf.float32
    )

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

    val_accuracy = (
        val_correct
        /
        tf.cast(
            val_total,
            tf.float32
        )
    )

    val_loss_average = (
        val_loss.numpy()
        /
        len(validation_data)
    )

    print(
        "Epoch:", epoch + 1,
        "Loss:", train_loss,
        "Accuracy:", train_accuracy.numpy(),
        "Val Loss:", val_loss_average,
        "Val Accuracy:", val_accuracy.numpy()
    )

test_loss = tf.constant(
    0.0,
    dtype=tf.float32
)

correct = tf.constant(
    0.0,
    dtype=tf.float32
)

total = 0

for x_batch, y_batch in test_data:

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

    correct += tf.reduce_sum(
        tf.cast(
            pred == y_batch,
            tf.float32
        )
    )

    total += tf.size(y_batch)

    test_loss += loss

test_accuracy = (
    correct
    /
    tf.cast(
        total,
        tf.float32
    )
)

test_loss_average = (
    test_loss.numpy()
    /
    len(test_data)
)

print(
    "Test Loss:", test_loss_average,
    "Test Accuracy:", test_accuracy.numpy()
)

model.save(
    "ycb_MobileNetV2_model.keras"
)