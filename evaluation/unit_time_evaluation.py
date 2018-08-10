import pandas as pd
from demo import profiling
from demo import matching
from demo import mapping
from demo import repair
from demo import transformation
from demo import deduplication
from demo import gt_comparison

#############################
# meta information to be used in any workflow

# define target schema
target_schema = {'street_name': 'str',
                 'price': 'int',
                 'postcode': 'str',
                 'bedroom_number': 'int',
                 'provenance': 'int-id'}

# define directories and file names
folder_main = "Data/real_estate/main_data/"
folder_context = "Data/real_estate/context_data/"
folder_res = "Data/results/"

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
cfds = pd.read_csv('Data/real_estate/context_data/cfds.csv')

# deduplication parameters
pca_components_number = 6
mean_shift_quantile = 0.25
distance_threshold = 0.995

###################################################
# normal order workflows

# profiling
result = %timeit -o profiling.load_data(folder_main, folder_context, main_data_names, additional_data_names, reference_data_names, pk_list, fk_list)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Data profiling\n')
results_file.write(result + '\n')
results_file.close()
datasets, datasets_composition, schemas = profiling.load_data(folder_main, folder_context, main_data_names, additional_data_names, reference_data_names, pk_list, fk_list)

# matching
result = %timeit -o matching.match_schemas(schemas, target_schema)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Data matching\n')
results_file.write(result + '\n')
results_file.close()

match_list = matching.match_schemas(schemas, target_schema)
datasets, schemas = matching.rename_columns(datasets, schemas, match_list, datasets_composition['main'])



# join for no repair and normal repair

result = %timeit -o mapping.union_and_join(datasets, datasets_composition, match_list, target_schema, add_data_columns, 'inner', 'postcode')
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Data mapping (union and inner join)\n')
results_file.write(result + '\n')
results_file.close()

inner_data = mapping.union_and_join(datasets, datasets_composition, match_list, target_schema, add_data_columns, 'inner', 'postcode')
inner_data = inner_data[['provenance','postcode', 'price', 'street_name', 'bedroom_number', 'crimerank']]


result = %timeit -o mapping.union_and_join(datasets, datasets_composition, match_list, target_schema, add_data_columns, 'left', 'postcode')
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Data mapping (union and left outer join)\n')
results_file.write(result + '\n')
results_file.close()

outer_data = mapping.union_and_join(datasets, datasets_composition, match_list, target_schema, add_data_columns, 'left', 'postcode')
outer_data = outer_data[['provenance','postcode', 'price', 'street_name', 'bedroom_number', 'crimerank']]



target_schema['crimerank'] = 'int-added'

# repair data

data_context_columns = ['postcode', 'street_name']
repaired_columns = ['street_name']

result = %timeit -o repair.repair_with_reference(inner_data, datasets, datasets_composition, data_context_columns, target_schema, repaired_columns)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Data repair (normal, inner join, reference)\n')
results_file.write(result + '\n')
results_file.close()
inner_reference_data = repair.repair_with_reference(inner_data, datasets, datasets_composition, data_context_columns, target_schema, repaired_columns)

result = %timeit -o repair.repair_with_reference(outer_data, datasets, datasets_composition, data_context_columns, target_schema, repaired_columns)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Data repair (normal, outer join, reference)\n')
results_file.write(result + '\n')
results_file.close()
outer_reference_data = repair.repair_with_reference(outer_data, datasets, datasets_composition, data_context_columns, target_schema, repaired_columns)

result = %timeit -o repair.repair_with_fds(inner_data, fd_list, cfds)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Data repair (normal, inner join, CFDs)\n')
results_file.write(result + '\n')
results_file.close()
inner_fds_data = repair.repair_with_fds(inner_data, fd_list, cfds)


result = %timeit -o repair.repair_with_fds(outer_data, fd_list, cfds)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Data repair (normal, outer join, CFDs)\n')
results_file.write(result + '\n')
results_file.close()

outer_fds_data = repair.repair_with_fds(outer_data, fd_list, cfds)

# transform

def transform(tr_data):
    tr_data = transformation.clean_spaces(tr_data)
    tr_data = transformation.clean_numeric(tr_data)

result = %timeit -o transform(inner_reference_data)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Transformation (normal, inner join, reference)\n')
results_file.write(result + '\n')
results_file.close()

inner_reference_data = transformation.clean_spaces(inner_reference_data)
inner_reference_data = transformation.clean_numeric(inner_reference_data, target_schema)


result = %timeit -o transform(outer_reference_data)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Transformation (normal, outer join, reference)\n')
results_file.write(result + '\n')
results_file.close()

outer_reference_data = transformation.clean_spaces(outer_reference_data)
outer_reference_data = transformation.clean_numeric(outer_reference_data, target_schema)


result = %timeit -o transform(inner_fds_data)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Transformation (normal, inner join, CFDs, spaces)\n')
results_file.write(result + '\n')
results_file.close()

inner_fds_data = transformation.clean_spaces(inner_fds_data)
inner_fds_data = transformation.clean_numeric(inner_fds_data, target_schema)

result = %timeit -o transform(outer_fds_data)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Transformation (normal, outer join, CFDs, spaces)\n')
results_file.write(result + '\n')
results_file.close()

outer_fds_data = transformation.clean_spaces(outer_fds_data)
outer_fds_data = transformation.clean_numeric(outer_fds_data, target_schema)

# deduplicate

result = %timeit -o deduplication.deduplicate(inner_reference_data, distance_threshold, pca_components_number, mean_shift_quantile)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Deduplication (normal, inner join, reference)\n')
results_file.write(result + '\n')
results_file.close()
inner_reference_data = deduplication.deduplicate(inner_reference_data, distance_threshold, pca_components_number, mean_shift_quantile)

result = %timeit -o deduplication.deduplicate(outer_reference_data , distance_threshold, pca_components_number, mean_shift_quantile)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Deduplication (normal, outer join, reference)\n')
results_file.write(result + '\n')
results_file.close()
outer_reference_data = deduplication.deduplicate(outer_reference_data , distance_threshold, pca_components_number, mean_shift_quantile)

result = %timeit -o deduplication.deduplicate(inner_fds_data, distance_threshold, pca_components_number, mean_shift_quantile)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Deduplication (normal, inner join, CFDs)\n')
results_file.write(result + '\n')
results_file.close()
inner_fds_data = deduplication.deduplicate(inner_fds_data, distance_threshold, pca_components_number, mean_shift_quantile)

result = %timeit -o deduplication.deduplicate(outer_fds_data, distance_threshold, pca_components_number, mean_shift_quantile)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Deduplication (normal, outer join, CFDs)\n')
results_file.write(result + '\n')
results_file.close()
outer_fds_data = deduplication.deduplicate(outer_fds_data, distance_threshold, pca_components_number, mean_shift_quantile)



match_list = matching.match_schemas(schemas, target_schema)
datasets, schemas = matching.rename_columns(datasets, schemas, match_list, datasets_composition['main'])

results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('\n\nRepairing original data\n')
results_file.close()

data_context_columns = ['postcode', 'street_name']
for i in datasets_composition['main']:
    dataset = datasets[i]
    schema = schemas[i]['schema']

    result = % timeit -o repair.repair_with_reference(dataset, datasets, datasets_composition, data_context_columns, schema, dataset.columns)
    result = str(result)
    results_file = open('demo/Data/unit_time_results.txt', 'a')
    results_file.write('Repair (reference)\n')
    results_file.write(result + '\n')
    results_file.close()
    reference_data = repair.repair_with_reference(dataset, datasets, datasets_composition, data_context_columns, schema, dataset.columns)

    result = % timeit -o transformation.clean_spaces(reference_data)
    result = str(result)
    results_file = open('demo/Data/unit_time_results.txt', 'a')
    results_file.write('Transform (reference, spaces)\n')
    results_file.write(result + '\n')
    results_file.close()
    reference_data = transformation.clean_spaces(reference_data)

    result = % timeit -o transformation.clean_numeric(reference_data, target_schema)
    result = str(result)
    results_file = open('demo/Data/unit_time_results.txt', 'a')
    results_file.write('Transform (reference, numeric)\n')
    results_file.write(result + '\n')
    results_file.close()
    reference_data = transformation.clean_numeric(reference_data, target_schema)
    datasets[i] = reference_data


match_list = matching.match_schemas(schemas, target_schema)
datasets, schemas = matching.rename_columns(datasets, schemas, match_list, datasets_composition['main'])



for i in datasets_composition['main']:
    dataset = datasets[i]

    result = % timeit -o repair.repair_with_fds(dataset, fd_list, cfds)
    result = str(result)
    results_file = open('demo/Data/unit_time_results.txt', 'a')
    results_file.write('Repair (CFDs)\n')
    results_file.write(result + '\n')
    results_file.close()
    fds_data = repair.repair_with_fds(dataset, fd_list, cfds)

    result = % timeit -o transformation.clean_spaces(fds_data)
    result = str(result)
    results_file = open('demo/Data/unit_time_results.txt', 'a')
    results_file.write('Transform (CFDs, spaces)\n')
    results_file.write(result + '\n')
    results_file.close()
    fds_data = transformation.clean_spaces(fds_data)

    result = % timeit -o transformation.clean_numeric(fds_data, target_schema)
    result = str(result)
    results_file = open('demo/Data/unit_time_results.txt', 'a')
    results_file.write('Transform (CFDs, numeric)\n')
    results_file.write(result + '\n')
    results_file.close()
    fds_data = transformation.clean_numeric(fds_data, target_schema)
    datasets[i] = fds_data


def repair_original_ref(datasets):
    for i in datasets_composition['main']:
        dataset = datasets[i]
        schema = schemas[i]['schema']
        reference_data = repair.repair_with_reference(dataset, datasets, datasets_composition, data_context_columns, schema, dataset.columns)
        datasets[i] = reference_data


def repair_original_fds(datasets):
    for i in datasets_composition['main']:
        dataset = datasets[i]
        fds_data = repair.repair_with_fds(dataset, fd_list, cfds)
        datasets[i] = fds_data


def transform_original(datasets):
    for i in datasets_composition['main']:
        dataset = datasets[i]
        fds_data = transformation.clean_spaces(dataset)
        datasets[i] = fds_data


result = % timeit -o repair_original_ref(datasets)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Repair original (ref)\n')
results_file.write(result + '\n')
results_file.close()

result = % timeit -o repair_original_fds(datasets)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Repair original (CFDs)\n')
results_file.write(result + '\n')
results_file.close()

result = % timeit -o transform_original(datasets)
result = str(result)
results_file = open('demo/Data/unit_time_results.txt', 'a')
results_file.write('Transform original \n')
results_file.write(result + '\n')
results_file.close()
