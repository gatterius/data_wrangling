# import all needed libraries

import pandas as pd # for general data processing
from demo import util_functions


# load data and create meta data
def load_data(main_dir, context_dir, main_data_names, additional_data_names, reference_data_names, pk_list, fk_list):
    datasets = []
    schemas = []
    dataset_list_composition = {}
    dataset_list_composition_counter = 0

    # load main datasets and create metadata for them
    dataset_list_composition['main'] = []
    for i in range(0, len(main_data_names)):
        # create a new metadata object with the name of dataset
        dataset_schema = {}
        dataset_schema['name'] = main_data_names[i]
        # load dataset from file
        loaded_dataset = pd.read_csv(main_dir + main_data_names[i] + ".csv")
        # extract the datatypes which are stored in the first row (right after column names)
        # and then remove this row from dataframe
        loaded_schema = loaded_dataset.iloc[0].to_dict()
        loaded_dataset = loaded_dataset.drop([0])
        # add labeled nulls, null calculation data and provenance
        dataset, null_list = add_labeled_nulls(loaded_dataset, main_data_names[i])
        dataset, loaded_schema = add_provenance(dataset, loaded_schema, main_data_names[i])
        # compose metadata file from created parts
        dataset_schema['schema'] = loaded_schema
        dataset_schema['PK'] = pk_list[i]
        dataset_schema['FK'] = list(fk_list.values())[i]
        dataset_schema['null_list'] = null_list
        # add dataset to the datasets list and schema to schemas list
        datasets.append(dataset)
        schemas.append(dataset_schema)
        # add information about a main dataset to the composition list
        dataset_list_composition['main'].append(dataset_list_composition_counter)
        dataset_list_composition_counter += 1

    # load additional datasets
    dataset_list_composition['additional'] = []
    for i in range(0, len(additional_data_names)):
        # load dataset and add it to the datasets list
        loaded_dataset = pd.read_csv(context_dir + additional_data_names[i] + ".csv")
        datasets.append(loaded_dataset)
        # add information about an addiotional dataset to the composition list
        dataset_list_composition['additional'].append(dataset_list_composition_counter)
        dataset_list_composition_counter += 1

    # load reference datasets
    dataset_list_composition['reference'] = []
    for i in range(0, len(reference_data_names)):
        # load dataset and add it to the datasets list
        loaded_dataset = pd.read_csv(context_dir + reference_data_names[i] + ".csv")
        datasets.append(loaded_dataset)
        # add information about a referenec  dataset to the composition list
        dataset_list_composition['reference'].append(dataset_list_composition_counter)
        dataset_list_composition_counter += 1

    return datasets, dataset_list_composition, schemas


# function which adds labeled nulls to the given dataset
def add_labeled_nulls(dataset, name):
    # compute the boolean matrix of the same size as dataset: if dataset element is null, assign True
    # to the corresponding element, not null - assign False
    isnull_matrix = dataset.isnull()
    # call function to compute the nulls in each dataset column
    null_count_list = count_nulls(isnull_matrix)
    count = 0
    # iterate through each column in dataset and each value in the column
    for col in dataset.columns:
        for i in range(0, dataset.shape[0]):
            # extract element from isnull matrix
            val = isnull_matrix[col].iloc[i]
            # if corresponding element in isnull matrix is null, replace the dataset element with labeled null
            if val:
                dataset[col].iloc[i] = name + '_null_' + str(count)
                count += 1
    return dataset, null_count_list


# function which adds provenance (in fact, primamry key) column to the dataset, and adds information
# about it to the schema
def add_provenance(dataset, schema, name):
    prov_column = []
    # create a list of unique values (consist of dataset name + counter) of the same length as dataset row number
    for i in range(0, dataset.shape[0]):
        prov_column.append(str(name) + '_' + str(i))
    # add this list as new column
    dataset['provenance'] = prov_column
    # add metadata to the schema
    schema['provenance'] = 'int-id'

    return dataset, schema


# function which calculates the number of nulls in each column based on the given isnull matrix
def count_nulls(isnull_matrix):
    null_count_list = {}
    # iterate through each column and each element of column in isnull matrix
    for column in isnull_matrix.columns:
        null_count_list[column] = []
        null_count_list[column].append(0)
        for i in range(0, isnull_matrix.shape[1]):
            # if element is True, add 1 to the null counter for this column
            if isnull_matrix[column].iloc[i]:
                null_count_list[column][0] += 1
        # add ratio information
        null_count_list[column].append(null_count_list[column][0] / isnull_matrix.shape[1])

    return null_count_list

# function which prints the metadata information
def print_metadata(schemas):
    for schema in schemas:
        print('Dataset name: ' + schema['name'])
        print('Dataset schema:')
        print(schema['schema'])
        print('Dataset primary key: ' + schema['PK'])
        print('Dataset foreign keys: ')
        print(schema['FK'])
        print('Dataset column null counts (total nulls in column/nulls to total values ratio): ')
        print(schema['null_list'])
        print('\n')

#########################################################################


