def search_list(items, target):
    for i in range(0, len(items)-1):
        if items[i] == target:
            return i
    return -1

