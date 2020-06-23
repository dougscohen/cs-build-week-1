
import csv

def read_csv(filename):
    
    data = []
    
    with open(filename) as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            data.append(row)
            
    return data


def convert_str_col_to_float(data, column):
    for row in data:
        row[column] = float(row[column].strip())
            
    # return data
    
def convert_str_col_to_int(data, column):
    for row in data:
        row[column] = int(row[column].strip())
            
    # return data


def encode_strings(data, column):
	class_values = [row[column] for row in data]
	unique_classes = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique_classes):
		lookup[value] = i
		# print('[%s] => %d' % (value, i))
	for row in data:
		row[column] = lookup[row[column]]
	return lookup


# function to find most frequent element in a list 
def most_frequent(List): 
    return max(set(List), key = List.count) 

# data = read_csv('iris.csv')
# print(data)
# convert_str_cols_to_float(data)
# convert_str_column_to_int(data)
# print(data)