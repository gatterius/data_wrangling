import pandas as pd
import numpy as np  # to process numeric arrays
import scipy.optimize as op  # to perform linear sum assignment
from demo import util_functions
import jellyfish  # for distance functions
from fuzzywuzzy import fuzz  # for distance functions

# calculate the distance between two given strings
def get_distance(string_a, string_b):
    # similarity scores given by edit distance functions are reversed  to turn them into distances,
    # since linear sum assignment finds combination with lowest possible sum
    lev = 1 - fuzz.ratio(string_a, string_b) / 100  # given value is normalized in range 1-100, not in 0-1
    jar = 1 - jellyfish.jaro_distance(string_a, string_b)
    jw = 1 - jellyfish.jaro_winkler(string_a, string_b)
    # calculate mean value of all distances
    score = (lev + jar + jw) / 3
    return score


# function which mathes 2 given datasets based on the attribute names and attribute datatype using the edit
# distance functions and linear sum assignment
# Linear sum assignment is a combinatorial optimization algorithm that solves the assignment problem. In a given
# matrix of certain scores, it aims to find a combination of elements of each column with the unique row number
# (once the element from a certain row is used, no other elements from this row can be chosen) which gives the minimum
# possible sum.
# The main idea of this function is to use different distance measures (Levenstein, Jaro and Jaro-Winkler) to calculate
# the mean distance for each pair of attributes in source and target schema. Once the distance matrix
# (where each element is the distance between corresponding pair of attributes) is calculated, it is transformed
# into a square matrix (adding rows/columns of zeros) and linear sum assigment is used to find the best candidate
# pairs. Once matrix coordinates of best elements are found, the respective attribute names are extracted and
# used to form match dictionary.
def schema_match_datatypes(source_schema, target_schema):
    distance_matrix = []
    # define the distance which will be used to discard pairs with different datatypes form matching
    infinite_distance = 10.0
    # extract attribute names and their datatypes
    source_attributes = list(source_schema.keys())
    target_attributes = list(target_schema.keys())
    source_types = list(source_schema.values())
    target_types = list(target_schema.values())
    # compute the distance matrix
    # iterate through all source and target attributes
    for targ in range(0, len(target_attributes)):
        distances = []
        for sour in range(0, len(source_attributes)):
            distances.append(get_distance(source_attributes[sour], target_attributes[targ]))
        distance_matrix.append(distances)

    # compare the data types of source and target schema, putting the results
    # (true if they same and false if they are different) in the comparison matrix,
    # "marking" the distances in the original dataset
    datatype_comparison_matrix = []
    for targ in range(0, len(target_attributes)):
        matrix_row = []
        for sour in range(0, len(source_attributes)):
            if source_types[sour] == target_types[targ]:
                matrix_row.append(True)
            else:
                matrix_row.append(False)
        datatype_comparison_matrix.append(matrix_row)

    # change the distance values for pairs which were marked in the previous loop (which have different datatypes)
    for targ in range(0, len(target_attributes)):
        for sour in range(0, len(source_attributes)):
            if not datatype_comparison_matrix[targ][sour]:
                distance_matrix[targ][sour] += infinite_distance


    cost = np.array(distance_matrix)
    cost = util_functions.pad_matrix(cost)
    [rows, columns] = op.linear_sum_assignment(cost)
    cols = list(columns)

    # cols now contains the position of elements which give best match
    # now there elements are extracted and a dictionary of matches is made where key is source attribute and value
    # is target attribute

    final_match = dict()
    for i in range(0, len(target_schema)):
        key = source_attributes[cols[i]]
        val = list(target_schema.keys())[i]
        final_match[key] = val
    return final_match


def rename_columns(datasets, schemas, rename_dictionaries, datasets_numbers):
    for i in datasets_numbers:
        dataset = datasets[i]
        names_list = rename_dictionaries[i]
        schema = schemas[i]['schema']
        dataset = dataset.rename(columns=names_list)
        datasets[i] = dataset
        for key, val in names_list.items():
            schema[val] = schema.pop(key)
        schemas[i]['schema'] = schema

    return datasets, schemas


# function which iterates through the given list of source schemata, matching each one to the target schema,
# and returns the dictionary of best matches
def match_schemas(schema_list, target_schema):
    match_list = []
    for a in range(0, len(schema_list)):
        schema = schema_list[a]['schema']  # extract the schema of the current dataset
        match = schema_match_datatypes(schema, target_schema)  # find the matches of this schema to the target
        match_list.append(match)

    return match_list



