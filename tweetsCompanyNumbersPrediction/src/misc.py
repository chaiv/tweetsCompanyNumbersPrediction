def pad_dict_lists(data):
    # Find largest list size for each key
    max_sizes = {}
    for key in data:
        max_size = 0
        for value in data[key]:
            if isinstance(value, list):
                max_size = max(max_size, len(value))
        max_sizes[key] = max_size
    
    # Pad non-list values to largest list size with element value
    for i in range(len(data[list(data.keys())[0]])):
        for key in data:
            if isinstance(data[key][i], list):
                continue
            max_size = max_sizes[key]
            padded_value = [data[key][i]] * max_size
            data[key][i] = padded_value
    
    return data


data = {'a': [1, 2], 'b': [[10, 11], [12, 13, 14]], 'c': [15, 16, 17]}
print(pad_dict_lists(data))
 

