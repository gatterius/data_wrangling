import pandas as pd
import numpy as np


# function which perform UNION on the given dataset, creating one datasets out of multiple
def union_data(datasets, datasets_composition, match_list, target_schema):
    # create an empty dataset of target schema
    unionized_data = pd.DataFrame(columns=target_schema.keys())
    for a in datasets_composition['main']:
        # extract column names from the dataset which correspond to discovered matches
        # (keys here are the keys of a dictionary)
        source_columns = list(match_list[a].values())
        # extract all data values from these columns
        data = datasets[a].loc[:, source_columns]
        # rename attributes according to the target schema
        # data = data.rename(columns=match_list[a])
        # append extracted column to unionized dataset
        unionized_data = unionized_data.append(data, ignore_index=True)

    return unionized_data


# using the found on the matching step matches, append all datasets into one, renaming their columns according to the
# names of attributes in target schema;
# add any additional datasets, appending them into one and then joining it with the main data on given attribute
# using the given type of join (INNER/OUTER)
def union_and_join(datasets, datasets_composition, match_list, target_schema, add_data_cols, join_type, join_attribute):

    # UNION all main datasets into one of target schema
    unionized_data = union_data(datasets, datasets_composition, match_list, target_schema)

    # UNION all additional datasets into one
    add_data_list = []
    for i in datasets_composition['additional']:
        add_data_list.append(datasets[i])
    unionized_add_data = add_data_list[0]
    for i in range(1, len(add_data_list)):
        unionized_add_data = unionized_add_data.append(add_data_list[1])
    # choose needed column from additional data, using given column names
    unionized_add_data = unionized_add_data.loc[:,add_data_cols]
    # JOIN the main data and additional data, using the given type of JOIN
    merged_data = pd.merge(unionized_data, unionized_add_data, on=join_attribute, how=join_type)
    # fill in emerged null values with numpy null
    merged_data = merged_data.fillna(np.nan)

    return merged_data


def union_datasets(datasets, datasets_composition, match_list, target_schema):
    # UNION all main datasets into one of target schema
    unionized_data = union_data(datasets, datasets_composition, match_list, target_schema)

    return unionized_data


def join_additional_data(unionized_data, datasets, datasets_composition, add_data_cols, join_type, join_attribute):
    # UNION all additional datasets into one
    add_data_list = []
    for i in datasets_composition['additional']:
        add_data_list.append(datasets[i])
    unionized_add_data = add_data_list[0]
    for i in range(1, len(add_data_list)):
        unionized_add_data = unionized_add_data.append(add_data_list[1])
    # choose needed column from additional data, using given column names
    unionized_add_data = unionized_add_data.loc[:, add_data_cols]
    # JOIN the main data and additional data, using the given type of JOIN
    merged_data = pd.merge(unionized_data, unionized_add_data, on=join_attribute, how=join_type)
    # fill in emerged null values with numpy null
    merged_data = merged_data.fillna(np.nan)

    return merged_data