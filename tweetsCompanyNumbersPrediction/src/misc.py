def transform_nested_list(input_list):
    output_list = []
    
    for item in input_list:
        if isinstance(item, list):
            flattened_item = [x for sublist in item for x in sublist]
        else:
            flattened_item = [item]
        
        repeated_item = []
        repeat_count = 5 // len(flattened_item)
        for elem in flattened_item:
            repeated_item.extend([elem] * repeat_count)
        
        remaining = 5 - len(repeated_item)
        if remaining > 0:
            repeated_item.extend([flattened_item[-1]] * remaining)
        
        output_list.append(repeated_item[:5])
    
    return output_list

# Test the function
input_list = [['a1', 'a2'], [['b1', 'b2'], ['b3', 'b4', 'b5']], 'c1']
output_list = transform_nested_list(input_list)

print(output_list)