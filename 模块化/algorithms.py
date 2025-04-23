# algorithms.py
class Algorithms:
    @staticmethod
    def run_kmeans(data, params):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=params['n_clusters'], max_iter=params['max_iter'], n_init=10)
        labels = kmeans.fit_predict(data)
        return labels

    @staticmethod
    def run_dbscan(data, params):
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        labels = dbscan.fit_predict(data)
        return labels