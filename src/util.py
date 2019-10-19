

def computeConfMatrix(conf_matrix):
    total_count = conf_matrix.sum()
    correct = 0
    precision = 0.0
    recall = 0.0
    f_score = 0.0
    r_count = 0
    f_count = 0
    for i in range(conf_matrix.shape[0]):
        tp = conf_matrix[i, i]
        tp_fp = conf_matrix.sum(axis=0)[i]
        tp_fn = conf_matrix.sum(axis=1)[i]
        if tp > 0:
            correct += tp
            precision += float(tp) / tp_fp * float(tp_fn) / total_count  # normalized by the portion of true label
            recall += float(tp) / tp_fn
            f_score += float(2 * tp) / (tp_fp + tp_fn)
        if tp_fn > 0:
            r_count += 1
        if tp_fp + tp_fn > 0:
            f_count += 1
    accuracy = float(correct) / total_count
    # precision /= conf_matrix.shape[0]
    recall /= r_count
    f_score /= f_count
    return accuracy, precision, recall, f_score


def outputs(algs, measurements):
    print('    accuracy, precision, recall, f_score')
    for a, m in zip(algs, measurements):
    	print('%s: %.6f, %.6f, %.6f, %.6f' % (a, m[0], m[1], m[2], m[3]))
    print()


def formatFloats(nums, digits):
    res = ''
    for n in nums:
        s = str(n)
        res += s[:s.find('.') + digits + 1] + ', '
    return res[:-2]