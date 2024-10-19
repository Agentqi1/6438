import random
import os
import shutil
import cv2
from tqdm import tqdm
import albumentations as A

dog_dir = './datasets/train/dog/'
aug_dog_dir = './datasets/train_aug/dog/'
cat_dir = './datasets/train/cat/'
aug_cat_dir = './datasets/train_aug/cat/'
os.makedirs(aug_dog_dir, exist_ok=True)
os.makedirs(aug_cat_dir, exist_ok=True)

random.seed(10)

transform = A.Compose([
    A.ChannelShuffle(),     # 随机重新排列RGB
    A.HueSaturationValue(),  # 随机调整图像的HSV
    A.RandomRotate90(),     # 随机将图像旋转90度的倍数
    A.ShiftScaleRotate(shift_limit=0.075, scale_limit=0.45, rotate_limit=45, p=0.75),  # 平移、缩放和旋转变换
    A.Blur(blur_limit=5, p=0.5),  # 模糊
    A.RandomBrightnessContrast(p=0.5),  # 亮度和对比度
    A.HorizontalFlip(p=0.5),  # 水平翻转
    A.VerticalFlip(p=0.5)  # 垂直翻转
])

for file in tqdm(os.listdir(dog_dir)):
    image = cv2.imread(os.path.join(dog_dir, file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented_image = transform(image=image)['image']
    cv2.imwrite(os.path.join(aug_dog_dir, 'aug_'+file), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
    shutil.copy(os.path.join(dog_dir, file), os.path.join(aug_dog_dir, file))

for file in tqdm(os.listdir(cat_dir)):
    image = cv2.imread(os.path.join(cat_dir, file))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented_image = transform(image=image)['image']
    cv2.imwrite(os.path.join(aug_cat_dir, 'aug_'+file), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
    shutil.copy(os.path.join(cat_dir, file), os.path.join(aug_cat_dir, file))

print('CAT training data after augmentation ———', len(os.listdir('datasets/train_aug/cat/')))
print('DOG training data after augmentation ———', len(os.listdir('datasets/train_aug/dog/')))
