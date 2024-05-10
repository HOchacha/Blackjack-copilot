from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
from yellowbrick.cluster import KElbowVisualizer


data = np.array([(5,42),(35,2.4),(9.3,50),(99.1,24.2)])
                
scaler = MinMaxScaler()
data_scale = scaler.fit_transform(data)


k = 3

model = KMeans(n_clusters = k, random_state = 10)

visualizer = KElbowVisualizer(model, k=(1,4))
visualizer.fit(data_scale)

predict = visualizer.fit_predict(data_scale)

print(predict)