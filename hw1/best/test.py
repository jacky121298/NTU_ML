import sys
import pandas as pd
import numpy as np

if __name__ == '__main__':
    test_csv = sys.argv[1]
    np_file = sys.argv[2]
    submit_file  = sys.argv[3]

    data = pd.read_csv(test_csv, header = None, encoding = 'big5').iloc[:, 2:].replace('NR', 0)
    data = data.to_numpy(dtype = float)

    with np.load(np_file) as npf:
        w, mean_x, std_x, del_row = npf['w'], npf['mean_x'], npf['std_x'], npf['del_row']

    for i in range(len(data)):
        if (i % 18) in del_row:
            data[i, :] = 0

    test_x = np.empty([240, 18 * 9], dtype = float)
    for i in range(240):
        test_x[i, :] = data[18 * i : 18 * (i + 1), :].reshape(1, -1)

    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
    
    test_x = np.concatenate( (np.ones([240, 1]), test_x), axis = 1).astype(float)
    ans = np.dot(test_x, w)

    import csv
    with open(submit_file, mode = 'w', newline = '') as sb_file:
        csv_w = csv.writer(sb_file)
        header = ['id', 'value']
        csv_w.writerow(header)
        for i in range(len(ans)):
            row = ['id_' + str(i), ans[i][0]]
            csv_w.writerow(row)