def get_distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def get_distance2(xy1, xy2):
    return get_distance(xy1[0], xy1[1], xy2[0], xy2[1])

def predict(data):
    n = len(data)

    if n == 0:
        return []
    
    if n == 1:
        return [0]

    ret = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            distance = get_distance2(data[i], data[j])

            # if i and j is in the same cluster
            if distance < 0.3:
                ret[j] = ret[i]
            else:
                ret[j] = ret[i] + 1
    
    return ret


