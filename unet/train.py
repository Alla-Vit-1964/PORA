import asyncio
import math
import os.path
import numpy as np

from UnetBuildModel import build_model
from imageSave import saveImagePIL
from lossAndAccuracyFunctions import bce_dice_loss, my_iou_metric

from __init__ import get_DIR_DATA, Adam, Model, Input, exists_data_obj

from generate import load_get_custom_objects, generate
from save_data import get_data, save_data

from PIL import Image, ImageDraw
import random
from utilities import check_circle_intersection


async def main():
    load_get_custom_objects()
    w_size = 256
    train_num = 8192
    train_x = np.zeros((train_num, w_size, w_size, 3), dtype='float32')
    train_y = np.zeros((train_num, w_size, w_size, 1), dtype='float32')

    img_l = np.random.sample((w_size, w_size, 3)) * 0.5
    img_h = np.random.sample((w_size, w_size, 3)) * 0.5 + 0.5
    radius_min = 10
    radius_max = 30
    result = []
    name_train_data = 'train_data'
    saveImagePIL(img_l, 'Background__img_l')
    saveImagePIL(img_h, 'Background__img_h')

    if exists_data_obj(name_train_data):
        task_load = asyncio.create_task(get_data(name_train_data))
        train_x, train_y = await task_load
    else:
        task = asyncio.create_task(generate(
            train_num,
            train_x,
            train_y,
            w_size,
            radius_max,
            radius_min,
            img_l,
            img_h))
        train_x, train_y = await task
        task_save = asyncio.create_task(save_data(name_train_data, (train_x, train_y)))
        await task_save

    input_layer = Input((w_size, w_size, 3))
    output_layer = build_model(input_layer, 16)
    model = Model(input_layer, output_layer)
    model.compile(loss=bce_dice_loss, optimizer=Adam(learning_rate=1e-3), metrics=[my_iou_metric])
    model.save_weights(os.path.join(get_DIR_DATA(), 'keras.weights'))

    while True:
        history = model.fit(train_x, train_y,
                            batch_size=32,
                            epochs=1,
                            verbose=1,
                            validation_split=0.1
                            )
        if history.history['my_iou_metric'][0] > 0.75:
            break


if __name__ == '__main__':
    asyncio.run(main())
