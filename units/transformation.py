import pandas as pd
import re
from word2number import w2n  # to convert words into numbers
import numpy as np


# function which deletes all redundant symbols from the numeric columns
def clean_numeric(dataset, target_schema):
    # iterate through all columns in the dataset
    for i in range(0, len(dataset.columns)):
        dataset_attr_name = dataset.columns[i]
        if dataset_attr_name in target_schema.keys() and target_schema[dataset_attr_name] == 'int':
            # find all numeric words and replace them with numbers
            try:
                dataset[dataset_attr_name] = w2n.word_to_num(str(dataset[dataset_attr_name]))
            except ValueError:
                pass
            # remove all symbols except digits
            column = dataset[dataset_attr_name]
            column = column.reset_index(drop=True)
            column = find_regex(column , '\d+[,.]?\d*')
            column = replace_symbol(column , ',', '')
            dataset[dataset_attr_name] = column

    return dataset


# function which finds all numeric words and replaces them with corresponding numbers, deleting the rest of the symbols
def replace_numeric_words(dataset, target_schema):
    # iterate through all columns in the dataset
    for i in range(0, len(dataset.columns)):
        dataset_attr_name = dataset.columns[i]
        if target_schema[dataset_attr_name] == 'int':
            # find all numeric words and replace them with numbers
            try:
                dataset[dataset_attr_name] = w2n.word_to_num(str(dataset[dataset_attr_name]))
            except ValueError:
                pass

    return dataset


def replace_symbol(column, old_symbol, new_symbol):
    for i in range(0, len(column)):
        val = column[i]
        val = str(val).replace(old_symbol, new_symbol)
        column[i] = val
    return column


# function which matches column values with given regex and replaces original values with found matches
def find_regex(column, regex):
    # iterate through all rows in dataset
    for i in range(0, len(column)):
        # compile and use regex on dataset value
        reg = re.compile(regex)
        val = column[i]
        val = str(val)
        r = reg.search(val)
        # if no matches were found, replace value with null
        if r is None:
            column[i] = np.nan
        # else replace value with found match
        else:
            val = r.group()
            # val = val.replace(',', '')
            column[i] = val
    return column


# function which matches column values with given regex and replaces original values with given values
# (not with the matches)
def replace_regex(column, regex, replace_value):
    # iterate through all rows in dataset
    for i in range(0, len(column)):
        # compile and use regex on dataset value
        reg = re.compile(regex)
        val = column[i]
        val = str(val)
        r = reg.search(val)
        # if regex was matched, replace dataset value with given replace value
        if r is not None:
            str(column[i]).replace(r.group(),replace_value)

    return column


# remove all redundant space or line breaks from the dataset columns
def clean_spaces(dataset):
    for i in range(0, len(dataset.columns)):
        # remove any spaces which have more then one
        column_name = dataset.columns[i]
        column = dataset[column_name]
        column = column.reset_index(drop=True)
        column = replace_regex(column, '  +', ' ')
        # remove all line breaks
        column = replace_regex(column, '\n', ' ',)
        dataset[column_name] = column
    return dataset


##########################
# The following transformation functions were inspired by the functionality of Trifacta Wrangler, described in the
# Trifacta paper (


# use a list of different regexes in the given order, replacing the matches with given values
# (False if match should be output),
# e.g. regex_list=['\d+', '\s+\d+'], replace_list=['aaa', False], order=[1,0]
def reorder_regex_matches(column, regex_list, replace_list, order):
    # iterate through all rows in dataset
    for i in range(0, len(column)):
        match_list = []
        # match all the regexes from the given regex list
        for regex in regex_list:
            # compile and use regex on dataset value
            reg = re.compile(regex)
            val = column[i]
            val = str(val)
            r = reg.search(val)
            # if match is found, replace value with found match
            if r is not None:
                val = r.group()
            # add value to the new value list
            match_list.append(val)

        # order the values in the order given as input
        val_list = ''
        for a in range(0, len(match_list)):
            if replace_list[order[a]]:
                val_list += replace_list[order[a]]
            else:
                val_list += str(match_list[order[a]])
        column[i] = val_list

    return column


# split column into multiple columns by using given split symbol
def split_column_bysymbol(dataset, column, split_symbol):
    # extract column values
    column_data = dataset[column]
    new_columns = []
    # create new columns, creating their values by splitting the original column
    for i in range(0, len(column_data)):
        new_row = str(column_data[i]).split(split_symbol)
        new_columns.append(new_row)

    # create a dataframe with new columns
    new_df = pd.DataFrame(new_columns)
    col_names = []
    for i in range(0, new_df.shape[1]):
        col_names.append(str(column) + str(i))
    new_df.columns = col_names
    # replace old column in the dataset with the new columns
    dataset = dataset.drop(columns=[column])
    dataset = dataset.concat(dataset, new_df)

    return dataset


# split column into multple by using given set of regex and their order; each match corresponding to a regex from
# list is written into separated column
def split_column_byregex(dataset, column, regex_list, order):
    # extract column values
    column_data = dataset[column]
    new_columns = []
    for i in range(0, len(column_data)):
        match_list = []
        new_row = []
        # for each regex from the list, match the column values
        for regex in regex_list:
            # compile regex and match it to the column value
            reg = re.compile(regex)
            val = column_data[i]
            val = str(val)
            r = reg.search(val)
            if r is not None:
                val = r.group()
            # append new value to the new column
            match_list.append(val)
        # reorder data in column according to the given order
        for a in range(0, len(match_list)):
            new_row.append(str(match_list[order[a]]))
        new_columns.append(new_row)
        # create a dataframe with new columns
        new_df = pd.DataFrame(new_columns)
        col_names = []
        for i in range(0, new_df.shape[1]):
            col_names.append(str(column) + str(i))
        new_df.columns = col_names
        # replace old column in the dataset with the new columns
        dataset = dataset.drop(columns=[column])
        dataset = dataset.concat(dataset, new_df)

    return dataset


# merge 2 columns into 1, naming it with the header of left column
def merge_columns(dataset, column_1, column_2, glue_symbol):
    # iterate through all value fo both column
    for i in range(0, len(dataset[column_1])):
        dataset[column_1][i] += dataset[column_2][i]
    # remove the right column
    dataset = dataset.drop(columns=[column_2])

    return dataset
