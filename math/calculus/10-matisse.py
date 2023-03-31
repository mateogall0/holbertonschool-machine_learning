#!/usr/bin/env python3
"""calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Calculate the derivative of a polynomial"""
    result = []
    if type(poly) != list:
        return None
    for idx, i in enumerate(poly):
        if type(i) != int:
            return None
        if idx == 0:
            continue
        result.append(i * idx)
    if len(result) == 0:
        return None
    return result
