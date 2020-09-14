import glob
import os

import cv2
import numpy as np

image_width = 256
image_height = 256


def gamma_transform(image, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(image, gamma_table)


def sfdjy_random_gamma_transform(image, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(image, gamma)


def sfdjy_rotate(image, label, angle):
    M_rotate = cv2.getRotationMatrix2D((image_width / 2, image_height / 2), angle, 1)
    image = cv2.warpAffine(image, M_rotate, (image_width, image_height))
    label = cv2.warpAffine(label, M_rotate, (image_width, image_height))
    return image, label


def sfdjy_blur(image):
    image = cv2.blur(image, (3, 3))
    return image


def sfdjy_add_noise(image):
    for i in range(200):
        temp_x = np.random.randint(0, image.shape[0])
        temp_y = np.random.randint(0, image.shape[1])
        image[temp_x][temp_y] = 255
    return image


def sfdjy_data_augment(image, label):
    if np.random.random() < 0.1:
        image, label = sfdjy_rotate(image, label, 90)

    if np.random.random() < 0.1:
        image, label = sfdjy_rotate(image, label, 180)

    if np.random.random() < 0.1:
        image, label = sfdjy_rotate(image, label, 270)

    if np.random.random() < 0.1:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)

    if np.random.random() < 0.1:
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)

    if np.random.random() < 0.25:
        image = sfdjy_random_gamma_transform(image, 1.0)

    if np.random.random() < 0.25:
        image = sfdjy_blur(image)

    if np.random.random() < 0.25:
        image = sfdjy_add_noise(image)

    return image, label


image_dir = './sfdjy_train_data/images/'
label_dir = './sfdjy_train_data/labels/'

count = 10000

sfdjy_images_paths = glob.glob('./sfdjy_data/images/' + '*.jpg')
print('sfdjy_images_paths length is ', len(sfdjy_images_paths))
images_paths = sfdjy_images_paths[10:30]
print('images_paths length is ', len(images_paths))

# for image_path in images_paths:
#     _, file_name = os.path.split(image_path)
#     image_name, _ = os.path.splitext(file_name)
#     label_path = './sfdjy_data/labels/' + image_name + '_label.png'

#     image = cv2.imread(image_path)
#     label = cv2.imread(label_path)

#     image_width_new = np.int(np.ceil(image.shape[1] / image_width)) * image_width
#     image_height_new = np.int(np.ceil(image.shape[0] / image_height)) * image_height

#     image = cv2.resize(image, (image_width_new, image_height_new))
#     label = cv2.resize(label, (image_width_new, image_height_new))

#     label_new = np.zeros((image_height_new, image_width_new), dtype=np.uint8)
#     for m in range(label_new.shape[0]):
#         for n in range(label_new.shape[1]):
#             if (np.array(label[m, n]) == (0, 0, 0)).all():
#                 label_new[m, n] = 0
#             if (np.array(label[m, n]) == (255, 0, 0)).all():
#                 label_new[m, n] = 1
#             if (np.array(label[m, n]) == (0, 255, 0)).all():
#                 label_new[m, n] = 2
#             if (np.array(label[m, n]) == (0, 255, 255)).all():
#                 label_new[m, n] = 3
#             if (np.array(label[m, n]) == (255, 255, 0)).all():
#                 label_new[m, n] = 4
#             if (np.array(label[m, n]) == (0, 0, 255)).all():
#                 label_new[m, n] = 5

#     for i in range(image_height_new // image_height):
#         for j in range(image_width_new // image_width):
#             image_result = image[i * image_height:(i + 1) * image_height, j * image_width:(j + 1) * image_width]
#             label_result = label_new[i * image_height:(i + 1) * image_height, j * image_width:(j + 1) * image_width]

#             print(count)
#             print(label_result.shape)

#             image_result, label_result = sfdjy_data_augment(image_result, label_result)

#             cv2.imwrite(image_dir + str(count) + '.jpg', image_result)
#             cv2.imwrite(label_dir + str(count) + '.png', label_result)
#             count = count + 1

count = 0

test_images_paths = sfdjy_images_paths[100:120]
print('images_paths length is ', len(test_images_paths))

for image_path in test_images_paths:
    _, file_name = os.path.split(image_path)
    image_name, _ = os.path.splitext(file_name)
    label_path = './sfdjy_data/labels/' + image_name + '_label.png'

    image = cv2.imread(image_path)
    label = cv2.imread(label_path)

    image_width_new = np.int(np.ceil(image.shape[1] / image_width)) * image_width
    image_height_new = np.int(np.ceil(image.shape[0] / image_height)) * image_height

    image = cv2.resize(image, (image_width_new, image_height_new))
    label = cv2.resize(label, (image_width_new, image_height_new))

    label_new = np.zeros((image_height_new, image_width_new), dtype=np.uint8)
    for m in range(label_new.shape[0]):
        for n in range(label_new.shape[1]):
            if (np.array(label[m, n]) == (0, 0, 0)).all():
                label_new[m, n] = 0
            if (np.array(label[m, n]) == (255, 0, 0)).all():
                label_new[m, n] = 1
            if (np.array(label[m, n]) == (0, 255, 0)).all():
                label_new[m, n] = 2
            if (np.array(label[m, n]) == (0, 255, 255)).all():
                label_new[m, n] = 3
            if (np.array(label[m, n]) == (255, 255, 0)).all():
                label_new[m, n] = 4
            if (np.array(label[m, n]) == (0, 0, 255)).all():
                label_new[m, n] = 5

    for i in range(image_height_new // image_height):
        for j in range(image_width_new // image_width):
            image_result = image[i * image_height:(i + 1) * image_height, j * image_width:(j + 1) * image_width]
            label_result = label_new[i * image_height:(i + 1) * image_height, j * image_width:(j + 1) * image_width]

            print(count)
            print(label_result.shape)

            cv2.imwrite('./sfdjy_test_data/images/' + str(count) + '.jpg', image_result)
            cv2.imwrite('./sfdjy_test_data/labels/' + str(count) + '.png', label_result)
            count = count + 1
