#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from sys import argv

from libs import equations, model, settings

if __name__ == "__main__":
    if len(argv) != 2:
        print(f"\nusage: {argv[0]} <EQUATION>")
        exit()

    tf.keras.backend.set_floatx(settings.DTYPE)
    eq = getattr(equations, argv[1])()
    model = model.DeepBSDE(eq)
    model.train()

