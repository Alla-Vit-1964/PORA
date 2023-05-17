import math


def check_ellipse_intersection(x1, y1, a1, b1, x2, y2, a2, b2, overlap_percent=100):
    """
    Проверяет, пересекаются ли два эллипса.

    :param x1: x-координата центра эллипса 1
    :param y1: y-координата центра эллипса 1
    :param a1: длина большой оси эллипса 1
    :param b1: длина малой оси эллипса 1
    :param x2: x-координата центра эллипса 2
    :param y2: y-координата центра эллипса2
    :param a2: длина большой оси эллипса 2
    :param b2: длина малой оси эллипса 2
    :param overlap_percent: процент пересечения, от 0 до 100
    :return: True, если два эллипса пересекаются на указанном уровне, False в противном случае
    """
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx * dx + dy * dy)
    return dist < (a1 / 2 + a2 / 2) * overlap_percent/100 and dist < (b1 / 2 + b2 / 2) * overlap_percent/100

def check_circle_intersection(x1, y1, r1, x2, y2, r2, overlap_percent=100):
    """
    Проверяет, пересекаются ли два круга в указанном процентном соотношении.

    :param x1: x-координата центра первого круга
    :param y1: y-координата центра первого круга
    :param r1: радиус первого круга
    :param x2: x-координата центра второго круга
    :param y2: y-координата центра второго круга
    :param r2: радиус второго круга
    :param overlap_percent: процент площади одного круга, который может перекрываться с другим
    :return: True, если два круга пересекаются в указанном процентном соотношении, False в противном случае
    """
    distance_between_centers = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    total_radius = r1 + r2
    overlap_radius = total_radius * overlap_percent / 100
    return distance_between_centers <= overlap_radius
