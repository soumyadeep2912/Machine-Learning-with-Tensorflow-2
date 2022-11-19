import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

DIMS = (512, 512)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def pre_process_image(image):
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image


def tensor_to_image(tensor):
    tensor_shape = tf.shape(tensor)
    number_elem_shape = tf.shape(tensor_shape)
    if number_elem_shape > 3:
        assert tensor_shape[0] == 1
        tensor = tensor[0]
    return tf.keras.preprocessing.image.array_to_img(tensor)


content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

layer_names = content_layers + style_layers


def load_img(path_to_img):
    image = tf.io.read_file(path_to_img)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, DIMS)
    image = tf.cast(image, tf.uint8)
    return tf.expand_dims(image, axis=0)


def load_images(content_path, style_path):
    content_image = load_img("{}".format(content_path))
    style_image = load_img("{}".format(style_path))
    return content_image, style_image


def clip_image_values(image, min_value=0.0, max_value=255.0):
    return tf.clip_by_value(image, clip_value_min=min_value, clip_value_max=max_value)


def vgg_model(layer_names):
    vgg = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model(inputs=vgg.inputs, outputs=outputs)
    return model


def content_loss(features, targets):
    return 0.5*tf.reduce_mean(tf.square(features - targets))


def style_loss(features, targets):
    return tf.reduce_mean(tf.square(features - targets))


def total_loss(style_targets, style_outputs, content_targets, content_outputs, style_weight, content_weight):
    style_losses = tf.add_n([style_loss(style_output, style_target)
                             for style_output, style_target in zip(style_outputs, style_targets)])
    style_losses *= style_weight/len(style_layers)
    content_losses = tf.add_n([content_loss(content_output, content_target)
                               for content_output, content_target in zip(content_outputs, content_targets)])
    content_losses *= content_weight/len(content_layers)
    loss = style_losses + content_losses
    return loss


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


def style_image_features(image):
    preprocessed_style_image = pre_process_image(image)
    vgg = vgg_model(style_layers)
    style_outputs = vgg(preprocessed_style_image)
    style_outputs = style_outputs[:len(style_layers)]
    gram_style_features = [gram_matrix(style_layer)
                           for style_layer in style_outputs]
    return gram_style_features


def content_image_features(image):
    preprocessed_content_image = pre_process_image(image)
    vgg = vgg_model(content_layers)
    content_outputs = vgg(preprocessed_content_image)
    return content_outputs


def calc_gradients(image, style_targets, content_targets, style_weight, content_weight, with_regularization=False):
    with tf.GradientTape() as tape:
        style_features = style_image_features(image)
        content_features = content_image_features(image)
        loss = total_loss(style_targets, style_features, content_targets,
                          content_features, style_weight, content_weight)
    gradients = tape.gradient(loss, image)
    return gradients


def update_image_with_style(image, style_targets, content_targets, style_weight,
                            content_weight, optimizer):
    gradients = calc_gradients(image, style_targets, content_targets,
                               style_weight, content_weight)

    optimizer.apply_gradients([(gradients, image)])
    image.assign(clip_image_values(image, min_value=0.0, max_value=255.0))


def fit_style_transfer(style_image, content_image, style_weight=1e-2, content_weight=1e-4,
                       optimizer='adam', epochs=1, steps_per_epoch=1):
    images = []
    step = 0
    style_targets = style_image_features(style_image)
    content_targets = content_image_features(content_image)
    generated_image = tf.cast(content_image, dtype=tf.float32)
    generated_image = tf.Variable(generated_image)
    images.append(content_image)

    for n in range(epochs):
        for m in tqdm.tqdm(range(steps_per_epoch)):
            step += 1
            update_image_with_style(generated_image, style_targets, content_targets,
                                    style_weight, content_weight, optimizer)
            print(".", end='')
            if (m + 1) % 10 == 0:
                images.append(generated_image)
        display_image = tensor_to_image(generated_image)
        images.append(generated_image)
        print("Train step: {}".format(step))
    generated_image = tf.cast(generated_image, dtype=tf.uint8)
    return generated_image, images


if __name__ == '__main__':
    content_path = tf.keras.utils.get_file(
        'content_image.jpg', 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/MLColabImages/dog1.jpg')
    style_path = tf.keras.utils.get_file(
        'style_image.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
    style_weight = 0.05
    content_weight = 0.75
    content_image, style_image = load_images(content_path, style_path)

    plt.imsave('style.jpg', style_image[0].numpy())
    # plt.imshow(style_image[0])
    # plt.show()

    plt.imsave('content.jpg', content_image[0].numpy())
    # plt.imshow(content_image[0])
    # plt.show()

    # define optimizer. learning rate decreases per epoch.
    adam = tf.optimizers.Adam(
        tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=80.0, decay_steps=100, decay_rate=0.80
        )
    )

    # start the neural style transfer
    stylized_image, display_images = fit_style_transfer(style_image=style_image, content_image=content_image,
                                                        style_weight=style_weight, content_weight=content_weight,
                                                        optimizer=adam, epochs=10, steps_per_epoch=100)
    plt.imsave('stylized.jpg', stylized_image[0].numpy())
    # plt.imshow(stylized_image[0])
    # plt.show()

    # for ind, elem in enumerate(display_images):
    #     plt.imsave('stylized_display_'+str(ind)+'.jpg', elem[0].numpy())
        # plt.imshow(elem[0])
        # plt.show()
