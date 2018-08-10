import pandas as pd
from units import profiling
from units import matching
from units import mapping
from units import repair
from units import transformation
from units import deduplication


target_schema = {'street_name': 'str',
                 'price': 'int',
                 'postcode': 'str',
                 'bedroom_number': 'int',
                 'provenance': 'int-id'}

# define directories and file names
folder_main = "demo/Data/real_estate/main_data/"
folder_context = "demo/Data/real_estate/context_data/"
folder_res = "demo/Data/results/"

main_data_names = ['onthemarket', 'belvoir', 'agent', 'rightmove', 'zoopla', 'cotswoldlettings', 'jpknight',
                   'oxlets', 'trinity']
additional_data_names = ['deprivation-data_pc_man', 'deprivation-data_pc_ox']
reference_data_names = ['openaddresses_m_ox_reducted']


pk_list = []
fk_list = {}
# since no pk and fk data is provided
for name in main_data_names:
    pk_list.append('provenance_' + name)
    fk_list[name] = []



# used columns from additional data
add_data_columns = ['postcode', 'crimerank']

# FDs
#fd_list = [['street_name', 'postcode']]
fd_list = []
cfds = pd.read_csv('demo/Data/real_estate/context_data/cfds.csv')

# deduplication parameters
pca_components_number = 6
mean_shift_quantile = 0.25
distance_threshold = 0.995

data_context_columns = ['postcode', 'street_name']
repaired_columns = ['street_name']

def wrangle_data(order, join, repair_type):

    if order == 0:
        target_schema = {'street_name': 'str',
                         'price': 'int',
                         'postcode': 'str',
                         'bedroom_number': 'int',
                         'provenance': 'int-id'}

        datasets, datasets_composition, schemas = profiling.load_data(folder_main, folder_context, main_data_names,
                                                                      additional_data_names,
                                                                      reference_data_names, pk_list, fk_list)
        match_list = matching.match_schemas(schemas, target_schema)
        datasets, schemas = matching.rename_columns(datasets, schemas, match_list, datasets_composition['main'])
        data = mapping.union_and_join(datasets, datasets_composition, match_list, target_schema, add_data_columns,
                                      join, 'postcode')
        # data = data[['provenance', 'postcode', 'price', 'street_name', 'bedroom_number', 'crimerank']]
        target_schema['crimerank'] = 'int-added'

    if order == 1: # normal
        target_schema = {'street_name': 'str',
                         'price': 'int',
                         'postcode': 'str',
                         'bedroom_number': 'int',
                         'provenance': 'int-id'}

        datasets, datasets_composition, schemas = profiling.load_data(folder_main, folder_context, main_data_names,
                                                                      additional_data_names,
                                                                      reference_data_names, pk_list, fk_list)
        match_list = matching.match_schemas(schemas, target_schema)
        datasets, schemas = matching.rename_columns(datasets, schemas, match_list, datasets_composition['main'])
        data = mapping.union_and_join(datasets, datasets_composition, match_list, target_schema, add_data_columns,
                                            join, 'postcode')
        # data = data[['provenance', 'postcode', 'price', 'street_name', 'bedroom_number', 'crimerank']]
        target_schema['crimerank'] = 'int-added'
        if repair_type == 'ref':
            data = repair.repair_with_reference(data, datasets, datasets_composition, data_context_columns,
                                                     target_schema, repaired_columns)
        else:
            data = repair.repair_with_fds(data, fd_list, cfds)
        data = transformation.clean_spaces(data)
        data = transformation.clean_numeric(data, target_schema)
        data = deduplication.deduplicate(data, distance_threshold, pca_components_number, mean_shift_quantile)

    if order == 2: # repair after union
        target_schema = {'street_name': 'str',
                         'price': 'int',
                         'postcode': 'str',
                         'bedroom_number': 'int',
                         'provenance': 'int-id'}

        datasets, datasets_composition, schemas = profiling.load_data(folder_main, folder_context, main_data_names,
                                                                      additional_data_names,
                                                                      reference_data_names, pk_list, fk_list)
        match_list = matching.match_schemas(schemas, target_schema)
        datasets, schemas = matching.rename_columns(datasets, schemas, match_list, datasets_composition['main'])
        data = mapping.union_data(datasets, datasets_composition, match_list, target_schema)
        if repair_type == 'ref':
            data = repair.repair_with_reference(data, datasets, datasets_composition, data_context_columns, target_schema, repaired_columns)
        else:
            data = repair.repair_with_fds(data, fd_list, cfds)
        data = transformation.clean_spaces(data)
        data = transformation.clean_numeric(data, target_schema)
        data = mapping.join_additional_data(data, datasets, datasets_composition, add_data_columns, join, 'postcode')
        # data = data[['provenance', 'postcode', 'price', 'street_name', 'bedroom_number', 'crimerank']]
        target_schema['crimerank'] = 'int-added'
        data = deduplication.deduplicate(data, distance_threshold, pca_components_number, mean_shift_quantile)

    if order == 3: # map after repair
        target_schema = {'street_name': 'str',
                         'price': 'int',
                         'postcode': 'str',
                         'bedroom_number': 'int',
                         'provenance': 'int-id'}

        datasets, datasets_composition, schemas = profiling.load_data(folder_main, folder_context, main_data_names,
                                                                      additional_data_names,
                                                                      reference_data_names, pk_list, fk_list)
        match_list = matching.match_schemas(schemas, target_schema)
        datasets, schemas = matching.rename_columns(datasets, schemas, match_list, datasets_composition['main'])
        for i in datasets_composition['main']:
            dataset = datasets[i]
            schema = schemas[i]['schema']
            if repair_type == 'ref':
                data = repair.repair_with_reference(dataset, datasets, datasets_composition, data_context_columns, schema, repaired_columns)
            else:
                data = repair.repair_with_fds(dataset, fd_list, cfds)
            data = transformation.clean_spaces(data)
            data = transformation.clean_numeric(data, target_schema)
            datasets[i] = data
        data = mapping.union_and_join(datasets, datasets_composition, match_list, target_schema, add_data_columns, join, 'postcode')
        # data = data[['provenance', 'postcode', 'price', 'street_name', 'bedroom_number', 'crimerank']]
        target_schema['crimerank'] = 'int-added'
        data = deduplication.deduplicate(data, distance_threshold, pca_components_number, mean_shift_quantile)



# data = wrangle_data(0,'inner','ref')

# no repair
results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(0,'inner','ref')
result = str(result)
results_file.write('No repair, inner\n')
results_file.write(result + '\n')
results_file.close()

results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(0,'left','ref')
result = str(result)
results_file.write('No repair, left\n')
results_file.write(result + '\n')
results_file.close()

# normal

results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(1,'inner','ref')
result = str(result)
results_file.write('Normal order, inner, reference\n')
results_file.write(result + '\n')
results_file.close()

results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(1,'inner','fds')
result = str(result)
results_file.write('Normal order, inner, CFDs\n')
results_file.write(result + '\n')
results_file.close()

results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(1,'left','ref')
result = str(result)
results_file.write('Normal order, outer, reference\n')
results_file.write(result + '\n')
results_file.close()

results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(1,'left','fds')
result = str(result)
results_file.write('Normal order, outer, CFDs\n')
results_file.write(result + '\n')
results_file.close()

# repair after union

results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(2,'inner','ref')
result = str(result)
results_file.write('Repair after union, inner, reference\n')
results_file.write(result + '\n')
results_file.close()

results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(2,'inner','fds')
result = str(result)
results_file.write('Repair after union, inner, CFDs\n')
results_file.write(result + '\n')
results_file.close()

results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(2,'left','ref')
result = str(result)
results_file.write('Repair after union, outer, reference\n')
results_file.write(result + '\n')
results_file.close()

results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(2,'left','fds')
result = str(result)
results_file.write('Repair after union, outer, CFDs\n')
results_file.write(result + '\n')
results_file.close()

# map after repair

results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(3,'inner','ref')
result = str(result)
results_file.write('Map after repair, inner, reference\n')
results_file.write(result + '\n')
results_file.close()

results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(3,'inner','fds')
result = str(result)
results_file.write('Map after repair, inner, CFDs\n')
results_file.write(result + '\n')
results_file.close()

results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(3,'left','ref')
result = str(result)
results_file.write('Map after repair, outer, reference\n')
results_file.write(result + '\n')
results_file.close()

results_file = open('demo/Data/time_eval_results.txt', 'a')
result = %timeit -o wrangle_data(3,'left','fds')
result = str(result)
results_file.write('Map after repair, outer, CFDs\n')
results_file.write(result + '\n')
results_file.close()