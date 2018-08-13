import numpy as np


# function which implements simple comparison - return True if all attributes of both tuples have identical values,
# and False in all other cases
def compare_tuples(tuple1, tuple2):
    res = False
    count = 0
    attr_num = len(tuple1)
    for i in range(0, attr_num):
        if tuple1[i] == tuple2[i]:
            count += 1
    if count >= attr_num-1:
        res = True
        for i in range(0, attr_num):
            t1 = tuple1[i]
            t2 = tuple2[i]
            if t1 == t2:
                count += 1
    return res


# calc the number of tuples which are present in both datasets, and the number of tuples which are not present only
# in one of the datasets
# format of output: [number of tuples in both, tuples only in dataset1, tuples only in dataset2]
def compare_dataset(dataset1, dataset2):
    count1 = 0
    for a in range(0, dataset1.shape[0]):
        for b in range(0, dataset2.shape[0]):
            if compare_tuples(dataset1.iloc[a], dataset2.iloc[b]):
                count1 += 1
                break
    count2 = dataset1.shape[0] - count1
    count3 = dataset2.shape[0] - count1
    return count1, count2, count3

def is_null(dataset_value):
    if 'null' in dataset_value or 'nan' in dataset_value or dataset_value == np.nan or \
            dataset_value == '' or dataset_value is None:
        return True
    else:
        return False

# compare given data tuples, assigning 1 of 4 label (TP, FP, FN, TN) to each attribute in output tuple
def compare_tuples_roc(output, gt):
    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0
    for i in range(0, len(output)):
        output_val = str(output[i])
        gt_val = str(gt[i])
        # if values are same and they are not nulls, assign TP
        if output_val == gt_val and not is_null(gt_val):
            tp_count += 1
        # if values are same and they are nulls, assign TN
        elif output_val == gt_val and is_null(gt_val):
            tn_count += 1
        # if value is null in output and not null in ground truth, assign FN
        elif is_null(output_val) and not is_null(gt_val):
            fn_count += 1
        # if value is not null in output and null in ground truth, assign FP
        else:
            fp_count += 1

    label_list = [tp_count, tn_count, fp_count, fn_count]
    # res = max(list)
    # res_index = list.index(max(list))
    return label_list


# compare 2 datasets, using defined primary key (pseudo or real) tuplewise
# this function implement more "permissive" approach, where
# 1. tuples are not removed from GT after match, so they can be matched to multiple output tuples
# 2. TP is assigned to the tuple if TP has highest ratio among all 4 labels, no matter what its ratio to total
# number of attributes is
# 3. if TP and any other label have the same amount of occurences, always assign TP to the tuple
def compare_dataset_roc(output, gt, pk_attribute):
    tp_count = 0
    fp_count = 0
    fn_count = 0
    # for each tuple in output dataset
    for i in range(0, output.shape[0]):
        # extract pk value
        output_row = output.iloc[i]
        output_pk = output_row[pk_attribute]
        # extract all tuple in GT that have same pk value
        gt_subset = gt.loc[gt[pk_attribute] == output_pk]
        roc_results = []
        # for each tuple in extracted GT susbet, compute the metrics
        for gt_row_num in range(0, gt_subset.shape[0]):
            roc_results.append(compare_tuples_roc(output_row, gt_subset.iloc[gt_row_num]))
        # if no tuples in GT with same pk value were found, assign FP label
        if not roc_results:
            fp_count += 1
        else:
            # out of all GT sibset tuples, choose the best one (the one that has maximum TP number) and assign final
            # label to output tuple (based on the label which has maximum number)
            roc_results = np.array(roc_results)
            tps = list(roc_results[:,0])
            subset_best_index = tps.index(max(tps))
            match = roc_results[subset_best_index, :]
            match_tp = match[0] + match[1]
            match_fp = match[2]
            match_fn = match[3]
            if (match_tp >= match_fn and match_tp == match_tp) or (match_tp >= match_fp and match_tp == match_fn):
                tp_count += 1
            elif (match_fp > match_tp and match_fp >= match_fn):
                fp_count += 1
            elif (match_fn > match_tp and match_fn >= match_fp):
                fn_count += 1
            else:
                tp_count += 1

    return tp_count, fp_count, fn_count


# compare 2 datasets, using defined primary key (pseudo or real number of runs and TP assignment threshold
# this function implements more "strict" approach, where
# 1. tuples are removed from GT after match, so they can be matched to only one output tuple
# 2. TP is assigned to the tuple if its ratio to total is above certain threshold
# 3. the problem of TP and any other label having the same amount of occurences is resolved in previous point if the
# value of threshold is above 0.5; the usage of any value under it does not align with "restritive" approach as it
# allows too many TPs
def compare_dataset_roc_strict(output, gt_set, pk_attribute, tp_threshold, runs_num):
    label_array = []
    # because GT tuples are removed from further search after successful match, several runs are executed to get
    # averaged results
    for run_counter in range(0, runs_num):
        gt_set = gt_set.reset_index(drop=True)
        gt = gt_set.copy()
        tp_count = 0
        fp_count = 0
        fn_count = 0
        attr_num = len(output.columns)
        # for each tuple in output dataset
        for i in range(0, output.shape[0]):
            # extract pk value
            output_row = output.iloc[i]
            output_pk = output_row[pk_attribute]
            # extract all tuple in GT that have same pk value
            gt_subset = gt.loc[gt[pk_attribute] == output_pk]
            gt_subset_index = gt.index[gt[pk_attribute] == output_pk]
            roc_results = []
            # for each tuple in extracted GT susbet, compute the metrics
            for gt_row_num in range(0, gt_subset.shape[0]):
                roc_results.append(compare_tuples_roc(output_row, gt_subset.iloc[gt_row_num]))
            # if no tuples in GT with same pk value were found, assign FP label
            if not roc_results:
                fp_count += 1
            else:
                # out of all GT sibset tuples, choose the best one (the one that has maximum TP number)
                roc_results = np.array(roc_results)
                tps = list(roc_results[:,0])
                subset_best_index = tps.index(max(tps))
                # get the index of best matching tuple in the whole GT dataset
                gt_best_index = gt_subset_index[subset_best_index]
                # remove matched tuple and reindex GT
                gt = gt.drop(gt.index[gt_best_index])
                gt = gt.reset_index(drop=True)
                match = roc_results[subset_best_index, :]
                match_tp = match[0] + match[1]
                match_fp = match[2]
                match_fn = match[3]
                # assign final label to output tuple - TP if it has ratio above threshold, FN if it has greater number
                # than FP, ot FP if it has greater number than FN
                if match_tp/attr_num >= tp_threshold:
                    tp_count += 1
                elif match_fn > match_fp:
                    fn_count += 1
                else:
                    fp_count += 1
        res_list = tp_count, fp_count, fn_count
        label_array.append(res_list)
    l = len(label_array)
    tp_count = 0
    fp_count = 0
    fn_count = 0

    # calculate the average metrics
    for i in range(0, l):
        tp_count += label_array[i][0]
        fp_count += label_array[i][1]
        fn_count += label_array[i][2]

    tp_count = int(tp_count/l)
    fp_count = int(fp_count/l)
    fn_count = int(fn_count/l)

    return tp_count, fp_count, fn_count


########


def compare_datasets(output, ground_truth, pk_attribute, strict_threshold, strict_runs):

    result = ''

    # use strict metrics
    [x1, x2, x3] = compare_dataset_roc_strict(output, ground_truth, pk_attribute, strict_threshold, strict_runs)

    # print('TP: ' + str(x1) + ', FP: ' + str(x2) + ', FN: ' + str(x3))
    result += 'Strict metrics: \n'
    result += 'TP: ' + str(x1) + ', FP: ' + str(x2) + ', FN: ' + str(x3) + '\n'
    if x1 + x2 != 0 and x1 + x3 != 0:
        precision = x1 / (x1 + x2)
        recall = x1 / (x1 + x3)
        fmeasure = (2 * precision * recall) / (precision + recall)
        # print('precision: ' + str(precision) + ' recall: ' + str(recall) + ' f-measure: ' + str(fmeasure))
        result += 'precision: ' + str(precision) + ' recall: ' + str(recall) + ' f-measure: ' + str(fmeasure) + '\n'

    # use permissive metrics
    [x1, x2, x3] = compare_dataset_roc(output, ground_truth, pk_attribute)

    # print('TP: ' + str(x1) + ', FP: ' + str(x2) + ', FN: ' + str(x3))
    result += 'Permissive metrics: \n'
    result += 'TP: ' + str(x1) + ', FP: ' + str(x2) + ', FN: ' + str(x3) + '\n'
    if x1+x2 != 0 and x1+x3 != 0:
        precision = x1/(x1+x2)
        recall = x1/(x1+x3)
        fmeasure = (2*precision*recall)/(precision+recall)
        # print('precision: ' + str(precision) + ' recall: ' + str(recall) + ' f-measure: ' + str(fmeasure))
        result += 'precision: ' + str(precision) + ' recall: ' + str(recall) + ' f-measure: ' + str(fmeasure) + '\n'

    print(result)

    return result

