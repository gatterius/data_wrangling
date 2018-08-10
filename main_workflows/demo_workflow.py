import pandas as pd
import profiling
import matching
import mapping
import repair
import transformation
import deduplication
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


# FDs
#fd_list = [['street_name', 'postcode']]
fd_list = []
cfds = pd.read_csv('Data/real_estate/context_data/cfds.csv')

# deduplication parameters
pca_components_number = 6
mean_shift_quantile = 0.25
distance_threshold = 0.995

# evaluation parameters
strict_threshold = 0.9
strict_runs = 1
pk_attribute = 'provenance'
groundtruth_inner = pd.read_csv('Data/groundtruth_inner.csv')
groundtruth_outer = pd.read_csv('Data/groundtruth_outer.csv')
###################################################
# normal order workflows

# load data and metadata

datasets, datasets_composition, schemas = profiling.load_data(folder_main, folder_context, main_data_names, additional_data_names,
                                                              reference_data_names, pk_list, fk_list)

# match schemata

# find matches
match_list = matching.match_schemas(schemas, target_schema)
# rename the matched columns
datasets, schemas = matching.rename_columns(datasets, schemas, match_list, datasets_composition['main'])

# map data and reorder columns
add_data_columns = ['postcode', 'crimerank']
mapped_data = mapping.union_and_join(datasets, datasets_composition, match_list, target_schema, add_data_columns, 'inner', 'postcode')
mapped_data = mapped_data[['provenance','postcode', 'price', 'street_name', 'bedroom_number', 'crimerank']]
# adjust data schema
target_schema['crimerank'] = 'int-added'

# repair data

# define needed reference columns and columns which are omitted during reference repair
data_context_columns = ['postcode', 'street_name']
non_repaired_columns = ['provenance', 'postcode', 'crimerank']
# use reference and CFDs to repair data
reference_repaired_data = repair.repair_with_reference(mapped_data, datasets, datasets_composition, data_context_columns, target_schema, non_repaired_columns)
fds_repaired_data = repair.repair_with_fds(mapped_data, fd_list, cfds)

# transform data

transformed_reference_data = transformation.clean_spaces(reference_repaired_data)
transformed_reference_data = transformation.clean_numeric(transformed_reference_data, target_schema)

transformed_fds_data = transformation.clean_spaces(fds_repaired_data)
transformed_fds_data = transformation.clean_numeric(transformed_fds_data, target_schema)

# deduplicate data

deduplicated_reference_data = deduplication.deduplicate(transformed_reference_data, distance_threshold, pca_components_number, mean_shift_quantile)
deduplicated_fds_data = deduplication.deduplicate(transformed_fds_data, distance_threshold, pca_components_number, mean_shift_quantile)
