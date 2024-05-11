def get_distance3(width, height):
    return (width**2 + height**2)**0.5

def get_distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def get_distance2(xy1, xy2):
    return get_distance(xy1[0], xy1[1], xy2[0], xy2[1])

def predict(data, height, width):
    n = len(data)

    if n == 0:
        return []
    
    if n == 1:
        return [0]
    
    standard = min(width, height) * 0.22

    # -1 means it is not belong to any cluster yet
    ret = [-1] * n
    uniqueid = 0

    for i in range(n):

        # if i has no cluster, assign unique cluster
        if ret[i] == -1:
            ret[i] = uniqueid
            uniqueid += 1

        for j in range(i+1, n):
            distance = get_distance2(data[i], data[j])

            # if i and j should be in the same cluster
            if distance < standard:
                if ret[j] != -1:
                    ret[i] = ret[j]
                else:
                    ret[j] = ret[i]
    
    return ret


