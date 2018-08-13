from scipy.stats import mode
import numpy as np
import util_functions


# function which utilizes the given data context dataset to improve and curate the original dataset
# it iterates through all the columns and rows of given dataset, doing following:
# 1. if the column of dataset is string, its value is a substring of corresponding column in data context, replace it
# with the corresponding value from data context
# 2. if the column dataset is numeric, remove all the redundant symbols
# 3. if field of dataset is empty, find a corresponding field in data context and fill the field in using found value
def use_reference(dataset, reference_data, target_schema, repaired_columns):
    # iterate through all columns in the dataset
    columns = dataset.columns
    col_number = len(columns)
    for outer_counter in range(0, col_number):
        # extract the name of current column
        dataset_attr_name = dataset.columns[outer_counter]

        # repair only original data
        if dataset_attr_name in repaired_columns and dataset_attr_name in reference_data.columns and \
                'added' not in target_schema[dataset_attr_name] and 'id' not in target_schema[dataset_attr_name]:
        # if dataset_attr_name in repaired_columns:
            print(str(dataset_attr_name))
            # iterate through all values in current column
            for dataset_row_counter in range(0, dataset.shape[0]):
                # print information in terminal to understand the algorithm execution
                # (not needed for the actual repair process)
                if dataset_row_counter % 50 == 0:
                    print("counter = " + str(dataset_row_counter))

                # normalize values using same columns in dataset and context (replace main dataset value with value from
                # reference data if latter is a substring of former); this is done to remove all redundant information
                # only perform if column has string type
                if target_schema[dataset_attr_name] == 'str':
                    # iterate through all values of column with the same name in the reference data
                    for reference_row_counter in range(0, reference_data.shape[0]):
                        # if reference value is a substring of value in the main dataset
                        dataset_value = dataset[dataset_attr_name].iloc[dataset_row_counter]
                        reference_value = reference_data[dataset_attr_name].iloc[reference_row_counter]
                        if reference_value in dataset_value:
                            # extract reference value and replace value in the main dataset with it
                            new_value = reference_data[dataset_attr_name].iloc[reference_row_counter]
                            dataset[dataset_attr_name].iloc[dataset_row_counter] = new_value
                            break

                # fill in empty fields in the main dataset
                # if main dataset field is not defined or has null in it
                if not dataset[dataset_attr_name].iloc[dataset_row_counter] or \
                        dataset[dataset_attr_name].iloc[dataset_row_counter] == np.nan or \
                        'null' in dataset[dataset_attr_name].iloc[dataset_row_counter]:
                    # iterate through all columns in data context
                    for reference_column_counter in range(0, len(reference_data.columns)):
                        # extract current reference column name
                        context_attr_name = reference_data.columns[reference_column_counter]
                        # iterate through all values in column
                        for reference_row_counter in range(0, reference_data.shape[0]):
                            # if values in the columns with same names are same, or reference data value
                            # is a substring of main data value
                            if ((dataset[context_attr_name].iloc[dataset_row_counter] == reference_data[context_attr_name].iloc[reference_row_counter]) or \
                                    (target_schema[context_attr_name] == 'str' and
                                     dataset[context_attr_name].iloc[dataset_row_counter] in reference_data[context_attr_name].iloc[reference_row_counter])) and \
                                    dataset_attr_name in reference_data.columns:
                                # replace empty main data value with founf corresponding reference data value
                                dataset[dataset_attr_name].iloc[dataset_row_counter] = reference_data[dataset_attr_name].iloc[reference_row_counter]
                                break
    return dataset


# use the given functional dependencies (externally discovered) to detect and correct violations
# FD list format: [['A', 'B'], ['C', 'D']] for FDs A -> B and C -> D
def replace_fd_violation_threshold(dataset, fd_list, threshold):
    # iterate though all FDs in list
    for fd in fd_list:
        # extract all unique values from the RHS column
        unique_vals = dataset[fd[0]].unique()
        for unique_val in unique_vals:
            # extract values of LHS column
            corr_vals = list(dataset.loc[dataset[fd[0]] == unique_val, fd[1]])
            # find most frequent LHS values
            mode_val = mode(corr_vals)
            value_number = len(corr_vals)
            # if its frequency is above given threshold, replace all other value in LHS column with it
            if mode_val[1]/value_number >= threshold:
                dataset.loc[dataset[fd[0]] == unique_val, fd[1]] = mode_val[0]

    return dataset

# use the conditional FDs extracted from the file to detect and correct violations
def use_cfds(dataset, cfds):
    # extract dataset column names
    col_names = list(dataset.columns)
    # iterate through all rows in the dataset
    for dataset_counter in range(0, dataset.shape[0]):
        # iterate through all CFDs in the list
        for cfd_counter in range(0, len(cfds)):
            # extract values from current rows
            data_row = dataset.iloc[dataset_counter]
            # exract CFD from the list
            cfd = cfds[cfd_counter]

            if_num = 0
            # calculate the number of CFD's RHS rule "hits"
            for key, val in cfd['lhs'].items():
                if key in col_names and (data_row[key] == val or val in data_row[key]):
                    if_num += 1
            # if all parts of CFD's RHS are present, replace the column corresponding to the LHS part of CFD
            # with appropriate values
            if if_num == len(cfd['lhs']):
                for key, val in cfd['rhs'].items():
                    if key in dataset.columns:
                        data_row['key'] = val
    return dataset


# Data repair function
# use reference data and given target schema with the data types of columns to update dataset - clean, replace noisy
# values, clean numeric columns or fill in empty fields by iterating through reference data
def repair_with_reference(merged_data, datasets, datasets_composition, data_context_cols, target_schema, non_repaired_columns):
    # UNION all reference datasets into one
    reference_data_list = []
    for i in datasets_composition['reference']:
        reference_data_list.append(datasets[i])
    unionized_reference_data = reference_data_list[0]
    for i in range(1, len(reference_data_list)):
        unionized_reference_data = unionized_reference_data.append(reference_data_list[1])
    # choose needed reference column using given column names
    reference_data = unionized_reference_data.loc[:, data_context_cols]
    # repair the main data
    merged_data = use_reference(merged_data, reference_data, target_schema, non_repaired_columns)

    return merged_data


# Data repair function
# use given FDs and conditional FDs to find the violations and to replace the violating values with the correct ones
def repair_with_fds(merged_data, fds, cfds_file):
    # repair data using FDs
    merged_data = replace_fd_violation_threshold(merged_data, fds, 0.6)
    # exract CFDs from the files
    extracted_cfds = util_functions.extract_cfds(cfds_file)
    # repair data using CFDs
    merged_data = use_cfds(merged_data, extracted_cfds)

    return merged_data