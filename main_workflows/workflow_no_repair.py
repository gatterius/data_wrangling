import pandas as pd
import profiling
import matching
import mapping
import gt_comparison

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
# profiling.print_metadata(schemas)

match_list = matching.match_schemas(schemas, target_schema)
datasets, schemas = matching.rename_columns(datasets, schemas, match_list, datasets_composition['main'])

inner_data = mapping.union_and_join(datasets, datasets_composition, match_list, target_schema, add_data_columns, 'inner', 'postcode')
inner_data = inner_data[['provenance','postcode', 'price', 'street_name', 'bedroom_number', 'crimerank']]

outer_data = mapping.union_and_join(datasets, datasets_composition, match_list, target_schema, add_data_columns, 'left', 'postcode')
outer_data = outer_data[['provenance','postcode', 'price', 'street_name', 'bedroom_number', 'crimerank']]

target_schema['crimerank'] = 'int-added'


# save data
inner_data = inner_data.astype(str)
outer_data = outer_data.astype(str)
inner_data.to_csv('data/results/no_repair_inner.csv', sep=',', encoding='utf-8', index=False)
outer_data.to_csv('data/results/no_repair_outer.csv', sep=',', encoding='utf-8', index=False)

