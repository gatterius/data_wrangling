import pandas as pd # for general data processing
from evaluation import gt_comparison

schema = {'provenance': str,
          'postcode': str,
          'price': str,
          'street_name': str,
          'bedroom_number': str,
          'crimerank': str}

ground_truth_inner = pd.read_csv('Data/groundtruth_inner.csv', dtype=schema)
ground_truth_outer = pd.read_csv('Data/groundtruth_outer.csv', dtype=schema)

pk_attribute = 'provenance'
strict_threshold = 0.6
strict_runs = 1

result = ''

# outputs

output_inner_names = ['no_repair_inner.csv',
                      'normal_order_inner_reference.csv',
                      'normal_order_inner_fds.csv',
                      'map_after_repair_inner_reference.csv',
                      'map_after_repair_inner_fds.csv',
                      'repair_after_union_inner_reference.csv',
                      'repair_after_union_inner_fds.csv']
output_outer_names =['no_repair_outer.csv',
                      'normal_order_outer_reference.csv',
                      'normal_order_outer_fds.csv',
                      'map_after_repair_outer_reference.csv',
                      'map_after_repair_outer_fds.csv',
                      'repair_after_union_outer_reference.csv',
                      'repair_after_union_outer_fds.csv']

output_columns = ['metric approach', 'encoding', 'TP', 'FP', 'FN', 'precision', 'recall', 'f-measure']

new_data = []
for i in range(0, len(output_inner_names)):
    output = pd.read_csv('Data/results/' + output_inner_names[i], dtype=schema)

    # use strict metrics
    [x1, x2, x3] = gt_comparison.compare_dataset_roc_strict(output, ground_truth_inner, pk_attribute, strict_threshold, strict_runs)
    precision = x1 / (x1 + x2)
    recall = x1 / (x1 + x3)
    fmeasure = (2 * precision * recall) / (precision + recall)
    new_row = {'metric approach': 'strict',
               'encoding': output_inner_names[i],
               'TP': x1,
               'FP': x2,
               'FN': x3,
               'precision': precision,
               'recall': recall,
               'f-measure': fmeasure}
    new_data.append(new_row)


    # use permissive metrics
    [x1, x2, x3] = gt_comparison.compare_dataset_roc(output, ground_truth_inner, pk_attribute)
    precision = x1 / (x1 + x2)
    recall = x1 / (x1 + x3)
    fmeasure = (2 * precision * recall) / (precision + recall)
    new_row = {'metric approach': 'permissive',
               'encoding': output_inner_names[i],
               'TP': x1,
               'FP': x2,
               'FN': x3,
               'precision': precision,
               'recall': recall,
               'f-measure': fmeasure}
    new_data.append(new_row)

inner_output = pd.DataFrame(new_data, columns=output_columns)

inner_output.to_csv('Data/eval_results_inner' + str(strict_threshold) + '.csv', sep=',', encoding='utf-8', index=False)

new_data = []
for i in range(0, len(output_outer_names)):
    output = pd.read_csv('Data/results/' + output_outer_names[i], dtype=schema)

    # use strict metrics
    [x1, x2, x3] = gt_comparison.compare_dataset_roc_strict(output, ground_truth_outer, pk_attribute, strict_threshold,
                                                            strict_runs)
    precision = x1 / (x1 + x2)
    recall = x1 / (x1 + x3)
    fmeasure = (2 * precision * recall) / (precision + recall)
    new_row = {'metric approach': 'strict',
               'encoding': output_outer_names[i],
               'TP': x1,
               'FP': x2,
               'FN': x3,
               'precision': precision,
               'recall': recall,
               'f-measure': fmeasure}
    new_data.append(new_row)

    # use permissive metrics
    [x1, x2, x3] = gt_comparison.compare_dataset_roc(output, ground_truth_outer, pk_attribute)
    precision = x1 / (x1 + x2)
    recall = x1 / (x1 + x3)
    fmeasure = (2 * precision * recall) / (precision + recall)
    new_row = {'metric approach': 'permissive',
               'encoding': output_outer_names[i],
               'TP': x1,
               'FP': x2,
               'FN': x3,
               'precision': precision,
               'recall': recall,
               'f-measure': fmeasure}
    new_data.append(new_row)

outer_output = pd.DataFrame(new_data, columns=output_columns)

outer_output.to_csv('Data/eval_results_outer' + str(strict_threshold) + '.csv', sep=',', encoding='utf-8', index=False)
