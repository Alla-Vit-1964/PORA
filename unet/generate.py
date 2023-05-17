import asyncio

import numpy as np
from tqdm import tqdm

from __init__ import ellipse, plt, get_DIR_IMG_RESULT, exists_data_obj
from imageSave import getNameImage, saveImagesPLT

from lossAndAccuracyFunctions import bce_dice_loss, dice_loss, dice_coef, my_iou_metric
from save_data import save_data, get_data
from utilities import check_ellipse_intersection


def load_get_custom_objects():
    from keras.utils import get_custom_objects

    get_custom_objects().update({ 'bce_dice_loss': bce_dice_loss })
    get_custom_objects().update({ 'dice_loss': dice_loss })
    get_custom_objects().update({ 'dice_coef': dice_coef })
    get_custom_objects().update({ 'my_iou_metric': my_iou_metric })


def next_pair(w_size, radius_max, radius_min, img_l, img_h, num_ellipses=15):
    img = img_h.copy()
    msk = np.zeros((w_size, w_size, 1), dtype='float32')

    coords_use = []
    for i in range(num_ellipses):
        while True:
            # r,c - координаты центра эллипса
            r = np.random.sample() * (w_size - 2 * radius_max) + radius_max
            c = np.random.sample() * (w_size - 2 * radius_max) + radius_max
            # большой и малый радиусы эллипса
            r_radius = np.random.sample() * (radius_max - radius_min) + radius_min
            c_radius = np.random.sample() * (radius_max - radius_min) + radius_min
            regenerated = False
            for coord in coords_use:
                if check_ellipse_intersection(
                        x1=coord['x'],
                        y1=coord['y'],
                        a1=coord['a'],
                        b1=coord['b'],
                        x2=r,
                        y2=c,
                        a2=r_radius,
                        b2=c_radius,
                ):
                    regenerated = True
                    break
            if not regenerated:
                coords_use.append({
                    'x': r,
                    'y': c,
                    'a': r_radius,
                    'b': c_radius,
                })
                break
        rot = np.random.sample() * 360  # наклон эллипса
        rr, cc = ellipse(
            r, c,
            r_radius, c_radius,
            rotation=np.deg2rad(rot),
            shape=img_l.shape
        )  # получаем все точки эллипса

        # красим пиксели эллипса в шум от 0.0 до 0.5
        img[rr, cc] = img_l[rr, cc]
        msk[rr, cc] = 1.  # красим пиксели маски эллипса

    return img, msk


async def generate(train_num, train_x, train_y, w_size, radius_max, radius_min, img_l, img_h, save_data_with_gen=False):
    print('Генерация всех img train')
    name_next_pair = 'next_pair'
    if exists_data_obj(name_next_pair) and save_data_with_gen:
        task_save = asyncio.create_task(get_data(name_next_pair))
        train_x, train_y = await task_save
    else:
        for k in tqdm(range(train_num)):  # генерация всех img train
            img, msk = next_pair(
                w_size,
                radius_max,
                radius_min,
                img_l,
                img_h
            )
            train_x[k] = img
            train_y[k] = msk
        if save_data_with_gen:
            task_save = asyncio.create_task(save_data(name_next_pair, (train_x, train_y)))
            await task_save
    if save_data_with_gen:
        saveImagesPLT(
            train_x, train_y
        )
    else:
        saveImagesPLT(
            train_x, train_y, 10
        )

    return train_x, train_y
