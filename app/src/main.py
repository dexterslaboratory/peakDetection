import json
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from numpy import ndarray
from datetime import datetime


def plot_timeline(times: ndarray, interactions: ndarray, peaks: ndarray, spans: ndarray):
    fig = plt.gcf()
    plt.gca().set_facecolor((0.14, 0.14, 0.15))
    fig.set_size_inches(18.5, 10.5)
    plt.plot(times, interactions)
    plt.xticks(rotation=90)
    dates_to_display = list()
    for time in times:
        dates_to_display.append(datetime.fromtimestamp(time).strftime("%Y-%m-%d"))
    x_axis_tick_freq = 24
    ticks_to_mark = times[np.arange(0, len(times), x_axis_tick_freq)]
    tick_labels_to_display = dates_to_display[1::x_axis_tick_freq]
    plt.xticks(ticks=ticks_to_mark, labels=tick_labels_to_display)
    plot_peaks(times, interactions, peaks)
    plot_spans(times, interactions, spans)
    plt.legend()
    plt.xlabel("Time ->")
    plt.ylabel("Interactions ->")
    plt.title("Peak and trough detection using argrelextrema algorithm")
    plt.savefig('../resources/output/public_1.png', dpi=100)


def plot_peaks(time: ndarray, interactions: ndarray, peaks: ndarray):
    plt.scatter(time[peaks], interactions[peaks], c='red', s=10, linewidth=0, label="Peaks")


def plot_spans(time: ndarray, interactions: ndarray, spans: ndarray):
    plt.scatter(time[spans], interactions[spans], c='yellow', s=10, linewidth=0, label="Troughs")


def initialize():
    ts = json.load(open('../resources/timeSeries/1/public/sample.json'))
    interactions: ndarray = np.array(list(map(lambda entry: entry['total'], ts)))
    inverted_interactions: ndarray = np.array(list(map(lambda entry: entry['total'] * -1, ts)))
    timestamp: ndarray = np.array(list(map(lambda entry: entry['time'] / 1000, ts)))
    peaks: ndarray = sig.argrelextrema(
        interactions,
        comparator=np.greater,
        order=5
    )[0]
    spans: ndarray = sig.argrelextrema(
        inverted_interactions,
        comparator=np.greater,
        order=3
    )[0]
    plot_timeline(timestamp, interactions, peaks, spans)


if __name__ == '__main__':
    initialize()
