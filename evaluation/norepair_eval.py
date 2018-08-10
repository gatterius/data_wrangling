import gt_comparison
import pandas as pd

schema = {'provenance': str,
          'postcode': str,
          'price': str,
          'street_name': str,
          'bedroom_number': str,
          'crimerank': str}
ground_truth_inner = pd.read_csv('Data/groundtruth_inner.csv', dtype=schema)
ground_truth_outer = pd.read_csv('Data/groundtruth_outer.csv', dtype=schema)
pk_attribute = 'provenance'
strict_threshold = 0.8
strict_runs = 1
result = ''
output = pd.read_csv('Data/results/no_repair_inner.csv', dtype=schema)
result = gt_comparison.compare_datasets(output, ground_truth_inner, pk_attribute, strict_threshold, strict_runs)