import asyncio
import os.path
import pickle
from __init__ import get_DIR_DATA_OBJ


async def save_data(name, data):
    print(f'Сохранение {name}')
    loop = asyncio.get_running_loop()
    with open(os.path.join(get_DIR_DATA_OBJ(), name + '.pickle'), 'wb') as f:
        await loop.run_in_executor(None, pickle.dump, data, f)


async def get_data(name):
    print(f'Загрузка {name}')
    loop = asyncio.get_running_loop()
    with open(os.path.join(get_DIR_DATA_OBJ(), name + '.pickle'), 'rb') as f:
        result = await loop.run_in_executor(None, pickle.load, f)
    return result
