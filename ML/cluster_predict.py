from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

def cluster(data):
    model = KMeans(random_state = 10)
    visualizer = KElbowVisualizer(model, k=(1,4))
    scaler = MinMaxScaler()
    data_scale = scaler.fit_transform(data)
    visualizer.fit(data_scale)
    predict = visualizer.fit_predict(data_scale)
    return predict

def main():
    result = cluster([(5,42),(35,2.4),(9.3,50),(99.1,24.2)])
    print(result)

if __name__=="__main__":
    main()
