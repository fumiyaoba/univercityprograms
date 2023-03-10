def list_sum(lists):
    list = [0]*len(lists[0])
    for i in range(len(lists)):
        list = [x + y for (x, y) in zip(list, lists[i])]
    return list