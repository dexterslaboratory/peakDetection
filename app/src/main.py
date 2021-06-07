import json
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from numpy import ndarray


def plot_timeline(time: ndarray, interactions: ndarray, peaks):
    plt.plot(time, interactions)
    plt.scatter(time[peaks], interactions[peaks], c='red', s=10, linewidth=0)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig("plot_1_public.png", dpi=100)


def initialize():
    ts = json.load(open('../resources/timeSeries/1/public/sample.json'))
    interactions: ndarray = np.array(list(map(lambda entry: entry['total'], ts)))
    timestamp: ndarray = np.array(list(map(lambda entry: entry['time'], ts)))
    peaks: ndarray = sig.argrelextrema(
        interactions,
        comparator=np.greater,
        order=5
    )[0]
    plot_timeline(timestamp, interactions, peaks)


if __name__ == '__main__':
    initialize()
