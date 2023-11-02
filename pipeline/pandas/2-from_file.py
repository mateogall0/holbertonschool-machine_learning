#!/usr/bin/env python3

import pandas as pd

def from_file(filename, delimiter):
    df = pd.read_csv(filename, delimiter)
    return df
