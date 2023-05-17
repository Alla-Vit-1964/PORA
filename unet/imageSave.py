import os
from PIL import Image
import uuid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import asyncio
from concurrent.futures import ThreadPoolExecutor

from __init__ import get_DIR_IMG, get_DIR_IMG_MASK, get_DIR_IMG_INPUT, get_DIR_IMG_RESULT


def getNameImage(path=get_DIR_IMG(), startName="img", bitmap_format='png'):
    return os.path.join(path, f'{startName}-{str(uuid.uuid4())}.{bitmap_format}')


def saveImagePIL(img, startName="img"):
    img = Image.fromarray((np.uint8(img * 255)))
    img.save(
        fp=getNameImage(startName=startName),
        bitmap_format='png'
    )


def saveImagesPLT(train_x, train_y, count_images=None):
    print('Сохранение изображений')
    if count_images is None:
        count_images = len(train_x)
    else:
        count_images = 1
    for k in tqdm(range(count_images)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].set_axis_off()
        axes[0].imshow(train_x[k])
        axes[1].set_axis_off()
        axes[1].imshow(train_y[k].squeeze())

        # Сохраняем результат (маску + изображение)
        mask_file = os.path.join(get_DIR_IMG_RESULT(), f'result_{k}.png')
        plt.savefig(mask_file)
        plt.close(fig)

        # Сохраняем изображение
        input_file = os.path.join(get_DIR_IMG_INPUT(), f'input_{k}.png')
        plt.imshow(train_x[k])
        plt.savefig(input_file)
        plt.close()

        # Сохраняем маску
        result_file = os.path.join(get_DIR_IMG_MASK(), f'mask_{k}.png')
        plt.imshow(train_x[k])
        plt.imshow(train_y[k].squeeze(), alpha=0.5)
        plt.savefig(result_file)
        plt.close()
