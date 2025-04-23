# algorithms.py
import numpy as np
import pyabf
from scipy.signal import peak_widths


class Algorithms:
    @staticmethod
    def run_kmeans(path, data, params):

        from sklearn.cluster import KMeans
        abf = pyabf.ABF(path)
        points = abf.sweepY  # 当前扫描的数据点

        inverted_signal = -points
        segments = []
        valid_peaks = []
        results = peak_widths(inverted_signal, data, rel_height=0.5)
        for i, peak in enumerate(data):
            left = int(results[2][i])  # left_ip
            right = int(results[3][i])  # right_ip
            if right - left < 5:  # 可加一个过滤条件
                continue
            raw_segment = points[left:right]
            valid_peaks.append(peak)

            # 插值为41长度
            interpolated = np.interp(np.linspace(0, len(raw_segment) - 1, 101),
                                     np.arange(len(raw_segment)),
                                     raw_segment)
            segments.append(interpolated)

        valid_peaks = np.array(valid_peaks)


        kmeans = KMeans(n_clusters=params['n_clusters'], n_init=params['n_init'],max_iter=params['max_iter'],random_state=params['random_state']).fit(segments)
        labels = kmeans.labels_

        return labels

    @staticmethod
    def run_dbscan(data, params):
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        labels = dbscan.fit_predict(data)
        return labels
