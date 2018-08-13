import pandas as pd
import profiling
import matching
import mapping
import repair
import transformation
import deduplication

#############################
# meta information to be used in any workflow

# define target schema
target_schema = {'street_name': 'str',
                 'price': 'int',
                 'postcode': 'str',
                 'bedroom_number': 'int',
                 'provenance': 'int-id'}

# define directories and file names
folder_main = "data/real_estate/main_data/"
folder_context = "data/real_estate/context_data/"
folder_res = "data/results/"

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
cfds = pd.read_csv('data/real_estate/context_data/cfds.csv')

# deduplication parameters
pca_components_number = 6
mean_shift_quantile = 0.25
distance_threshold = 0.995

# evaluation parameters
strict_threshold = 0.9
strict_runs = 1
pk_attribute = 'provenance'
groundtruth_inner = pd.read_csv('data/groundtruth_inner.csv')
groundtruth_outer = pd.read_csv('data/groundtruth_outer.csv')
###################################################
# normal order workflows

# join data

datasets, datasets_composition, schemas = profiling.load_data(folder_main, folder_context, main_data_names, additional_data_names,
                                                              reference_data_names, pk_list, fk_list)

match_list = matching.match_schemas(schemas, target_schema)
datasets, schemas = matching.rename_columns(datasets, schemas, match_list, datasets_composition['main'])

# union datasets
unionized_data = mapping.union_data(datasets, datasets_composition, match_list, target_schema)

# join additional data
inner_joined_data = mapping.join_additional_data(unionized_data, datasets, datasets_composition, add_data_columns, 'inner', 'postcode')
inner_joined_data = inner_joined_data[['provenance', 'postcode', 'price', 'street_name', 'bedroom_number', 'crimerank']]

outer_joined_data = mapping.join_additional_data(unionized_data, datasets, datasets_composition, add_data_columns, 'left', 'postcode')
outer_joined_data = outer_joined_data[['provenance', 'postcode', 'price', 'street_name', 'bedroom_number', 'crimerank']]

target_schema['crimerank'] = 'int-added'

# repair data

data_context_columns = ['postcode', 'street_name']
repaired_columns = ['street_name']

inner_reference_data = repair.repair_with_reference(inner_joined_data, datasets, datasets_composition, data_context_columns, target_schema, repaired_columns)
outer_reference_data = repair.repair_with_reference(outer_joined_data, datasets, datasets_composition, data_context_columns, target_schema, repaired_columns)

inner_fds_data = repair.repair_with_fds(inner_joined_data, fd_list, cfds)
outer_fds_data = repair.repair_with_fds(outer_joined_data, fd_list, cfds)

# transform data

inner_reference_data = transformation.clean_spaces(inner_reference_data)
inner_reference_data = transformation.clean_numeric(inner_reference_data, target_schema)

outer_reference_data = transformation.clean_spaces(outer_reference_data)
outer_reference_data = transformation.clean_numeric(outer_reference_data, target_schema)

inner_fds_data = transformation.clean_spaces(inner_fds_data)
inner_fds_data = transformation.clean_numeric(inner_fds_data, target_schema)

outer_fds_data = transformation.clean_spaces(outer_fds_data)
outer_fds_data = transformation.clean_numeric(outer_fds_data, target_schema)

# deduplicate data

inner_reference_data = deduplication.deduplicate(inner_reference_data, distance_threshold, pca_components_number, mean_shift_quantile)
outer_reference_data = deduplication.deduplicate(outer_reference_data , distance_threshold, pca_components_number, mean_shift_quantile)
inner_fds_data = deduplication.deduplicate(inner_fds_data, distance_threshold, pca_components_number, mean_shift_quantile)
outer_fds_data = deduplication.deduplicate(outer_fds_data, distance_threshold, pca_components_number, mean_shift_quantile)

# save data

inner_reference_data.to_csv('data/results/normal_order_inner_reference.csv', sep=',', encoding='utf-8', index=False)
outer_reference_data.to_csv('data/results/normal_order_outer_reference.csv', sep=',', encoding='utf-8', index=False)
inner_fds_data.to_csv('data/results/normal_order_inner_fds.csv', sep=',', encoding='utf-8', index=False)
outer_fds_data.to_csv('data/results/normal_order_outer_fds.csv', sep=',', encoding='utf-8', index=False)



