# -*- coding:utf-8 -*-
# @Author: Jone Chiang
# @Date  : 2019/12/27 14:40
# @File  : tf_0x006
import pathlib
import random
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)
print(data_root)

for item in data_root.iterdir():
    print(item)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print(image_count)

attributions = (data_root/'LICENSE.txt').open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)

def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    image_rel = str(image_rel).replace('\\', '/')
    return 'Image (CC BY 2.0)' + '-'.join(attributions[image_rel].split(' - ')[:-1])

for n in range(3):
    image_path = random.choice(all_image_paths)
    # img = Image.open(image_path)
    # img.show()
    print(caption_image(image_path))

# 确定每张图片的标签
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)

# 为每个标签分配索引
label_to_index = dict((name, index) for index, name in enumerate(label_names))
print(label_to_index)

# 创建一个列表，包含每个文件的标签索引
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

# 加载和格式化图片
img_path = all_image_paths[0]
img_raw = tf.io.read_file(img_path)
print(repr(img_raw)[:100]+'...')

# 将它解码为图像Tensor
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

image_path = all_image_paths[0]

label = all_image_labels[0]
plt.subplot(121)
plt.imshow(load_and_preprocess_image(image_path))
plt.grid(False)
plt.xlabel(caption_image(image_path))
plt.title(label_names[label].title())

# image = Image.open(image_path)
# plt.subplot(122)
# plt.imshow(image)
# plt.xlabel(caption_image(image_path))
# plt.title(label_names[label].title())

plt.show()

# 构建一个tf.data.Dataset
path_ds = tf.data.Dataset.from_tensor_slices(all_image_labels)

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

plt.figure(figsize=(8, 8))
for n, image in enumerate(image_ds.take(4)):
    plt.subplot(2, 2, n+1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(caption_image(all_image_paths[n]))
    plt.show()

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
for label in label_ds.take(4):
    print(label_names[label.numpy()])

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)
