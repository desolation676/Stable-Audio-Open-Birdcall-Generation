import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KS = [44, 88, 16, 32, 100]
ratios = [1, 2]
re = 5
CS = [0.01, 0.1, 1.0, 10.0]