import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans as model

from common import polar_in_cartesian, to_polar


class Model:
    def __init__(self) -> None:
        pass 

    def run(self, dos: np.ndarray, energy: np.ndarray) -> tuple[ndarray, ndarray]:
        x, y = polar_in_cartesian(*to_polar(dos, energy))
        data = np.stack((x, y), axis=1)

        kmeans = model(n_clusters=2, n_init='auto')
        kmeans.fit(data)
        labels = kmeans.labels_

        min_index = dos.argmin()
        max_index = dos.argmax()
        min_label = labels[min_index]
        max_label = labels[max_index]
        
        select_index = np.where(max_label == labels)
        assert len(select_index) != 0, "No cluster found"
        select_index = select_index[0]
        
        return dos[select_index], energy[select_index]
