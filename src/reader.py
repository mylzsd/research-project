import os
import pandas as pd


class Reader:
    def __init__(self, random_state, portion=0.5):
        self.random_state = random_state
        self.dir = os.getcwd() + '/data/'
        self.portion = portion

    '''
    portion is for the first half
    '''
    def splitByPortion(self, data, portion):
        part1 = data.sample(frac=portion, random_state=self.random_state)
        part2 = data.loc[~data.index.isin(part1.index), :]
        return (part1, part2)

    # encode boolean categorical column to numerical
    def boolColumn(self, dfs, col, type1, type2):
        for df in dfs:
            for i in df.index:
                var = df.loc[i, col]
                if var == type1:
                    df.loc[i, col] = -1
                elif var == type2:
                    df.loc[i, col] = 1
                else:
                    df.loc[i, col] = 0

    # One-hot encoding with pre-defined missing value
    def cateColumn(self, dfs, col, missing):
        types = dict()
        for df in dfs:
            for i in df.index:
                var = str(df.loc[i, col])
                if var != missing and var not in types:
                    types[var] = col + '_' + var

        for val in types.values():
            for df in dfs:
                df.insert(0, val, 0)

        for df in dfs:
            for i in df.index:
                var = str(df.loc[i, col])
                if var in types:
                    df.loc[i, types[var]] = 1
            df.drop(col, axis = 1, inplace = True)

    # encode ordered categorical column to numerical
    def numeColumn(self, dfs, col, values):
        for df in dfs:
            for i in df.index:
                var = df.loc[i, col]
                df.loc[i, col] = values[var]

    # fill empty numerical column with mean value
    def fillEmpty(self, dfs, col, missing, is_float):
        s = 0
        c = 0
        for df in dfs:
            for i in df.index:
                var = df.loc[i, col]
                if var != missing:
                    s += float(var)
                    c += 1
        if is_float:
            v = s / c
        else:
            v = int(s / c)
        for fd in dfs:
            for i in df.index:
                if df.loc[i, col] == missing:
                    df.loc[i, col] = v


    '''
    (200, 95)
    (26, 95)
    '''
    def audiology(self):
        train = pd.read_csv(self.dir + 'audiology/audiology.data.csv', header = None)
        test = pd.read_csv(self.dir + 'audiology/audiology.test.csv', header = None)
        columns = ['age_gt_60', 'air', 'airBoneGap', 'ar_c', 'ar_u', 
                   'bone', 'boneAbnormal', 'bser', 'history_buzzing', 
                   'history_dizziness', 'history_fluctuating', 'history_fullness', 
                   'history_heredity', 'history_nausea', 'history_noise', 
                   'history_recruitment', 'history_ringing', 'history_roaring', 
                   'history_vomiting', 'late_wave_poor', 'm_at_2k', 'm_cond_lt_1k', 
                   'm_gt_1k', 'm_m_gt_2k', 'm_m_sn', 'm_m_sn_gt_1k', 'm_m_sn_gt_2k', 
                   'm_m_sn_gt_500', 'm_p_sn_gt_2k', 'm_s_gt_500', 'm_s_sn', 
                   'm_s_sn_gt_1k', 'm_s_sn_gt_2k', 'm_s_sn_gt_3k', 'm_s_sn_gt_4k', 
                   'm_sn_2_3k', 'm_sn_gt_1k', 'm_sn_gt_2k', 'm_sn_gt_3k', 
                   'm_sn_gt_4k', 'm_sn_gt_500', 'm_sn_gt_6k', 'm_sn_lt_1k', 
                   'm_sn_lt_2k', 'm_sn_lt_3k', 'middle_wave_poor', 'mod_gt_4k', 
                   'mod_mixed', 'mod_s_mixed', 'mod_s_sn_gt_500', 'mod_sn', 
                   'mod_sn_gt_1k', 'mod_sn_gt_2k', 'mod_sn_gt_3k', 'mod_sn_gt_4k', 
                   'mod_sn_gt_500', 'notch_4k', 'notch_at_4k', 'o_ar_c', 'o_ar_u', 
                   's_sn_gt_1k', 's_sn_gt_2k', 's_sn_gt_4k', 'speech', 
                   'static_normal', 'tymp', 'viith_nerve_signs', 'wave_V_delayed', 
                   'waveform_ItoV_prolonged', 'indentifier', 'class']
        bool_columns = ['age_gt_60', 'airBoneGap', 'boneAbnormal', 'history_buzzing', 
                        'history_dizziness', 'history_fluctuating', 'history_fullness', 
                        'history_heredity', 'history_nausea', 'history_noise', 
                        'history_recruitment', 'history_ringing', 'history_roaring', 
                        'history_vomiting', 'late_wave_poor', 'm_at_2k', 'm_cond_lt_1k', 
                        'm_gt_1k', 'm_m_gt_2k', 'm_m_sn', 'm_m_sn_gt_1k', 'm_m_sn_gt_2k', 
                        'm_m_sn_gt_500', 'm_p_sn_gt_2k', 'm_s_gt_500', 'm_s_sn', 
                        'm_s_sn_gt_1k', 'm_s_sn_gt_2k', 'm_s_sn_gt_3k', 'm_s_sn_gt_4k', 
                        'm_sn_2_3k', 'm_sn_gt_1k', 'm_sn_gt_2k', 'm_sn_gt_3k', 
                        'm_sn_gt_4k', 'm_sn_gt_500', 'm_sn_gt_6k', 'm_sn_lt_1k', 
                        'm_sn_lt_2k', 'm_sn_lt_3k', 'middle_wave_poor', 'mod_gt_4k', 
                        'mod_mixed', 'mod_s_mixed', 'mod_s_sn_gt_500', 'mod_sn', 
                        'mod_sn_gt_1k', 'mod_sn_gt_2k', 'mod_sn_gt_3k', 'mod_sn_gt_4k', 
                        'mod_sn_gt_500', 'notch_4k', 'notch_at_4k', 's_sn_gt_1k', 
                        's_sn_gt_2k', 's_sn_gt_4k', 'static_normal', 'viith_nerve_signs', 
                        'wave_V_delayed', 'waveform_ItoV_prolonged']
        cate_columns = ['air', 'ar_c', 'ar_u', 'bone', 'bser', 
                        'o_ar_c', 'o_ar_u', 'speech', 'tymp']
        train.columns = columns
        test.columns = columns
        # drop id column
        train.drop('indentifier', axis = 1, inplace = True)
        test.drop('indentifier', axis = 1, inplace = True)
        # cast categorical to numerical
        for col in bool_columns:
            self.boolColumn([train, test], col, 'f', 't')
        for col in cate_columns:
            self.cateColumn([train, test], col, '?')
        train = train.infer_objects()
        train_clf, train_mdp = self.splitByPortion(train, self.portion)
        test = test.infer_objects()
        return (train, train_clf, train_mdp, test)


    '''
    (257, 16)
    (29, 16)
    '''
    def breast_cancer(self):
        data = pd.read_csv(self.dir + 'breast-cancer/breast-cancer.data.csv', header = None)
        columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 
                   'deg-malig', 'breast', 'breast-quad', 'irradiat']
        data.columns = columns
        # boolean columns
        self.boolColumn([data], 'node-caps', 'yes', 'no')
        self.boolColumn([data], 'breast', 'left', 'right')
        self.boolColumn([data], 'irradiat', 'yes', 'no')
        # categorical columns
        self.cateColumn([data], 'menopause', '?')
        self.cateColumn([data], 'breast-quad', '?')
        # numerical columns
        age_dict = {'?' : 0, '10-19' : 1, '20-29' : 2, '30-39' : 3, '40-49' : 4, 
                    '50-59' : 5, '60-69': 6, '70-79' : 7, '80-89' : 8, '90-99' : 9}
        self.numeColumn([data], 'age', age_dict)
        tumor_dict = {'?' : 0, '0-4' : 1, '5-9' : 2, '10-14' : 3, '15-19' : 4, 
                      '20-24' : 5, '25-29' : 6, '30-34' : 7, '35-39' : 8, 
                      '40-44' : 9, '45-49' : 10, '50-54' : 11, '55-59' : 12}
        self.numeColumn([data], 'tumor-size', tumor_dict)
        inv_dict = {'?' : 0, '0-2' : 1, '3-5' : 2, '6-8' : 3, '9-11' : 4, '12-14' : 5, 
                    '15-17' : 6, '18-20' : 7, '21-23' : 8, '24-26' : 9, 
                    '27-29' : 10, '30-32' : 11, '33-35' : 12, '36-39' : 13}
        self.numeColumn([data], 'inv-nodes', inv_dict)
        # move class to last column
        col_size = data.shape[1]
        data.insert(col_size - 1, 'Class', data.pop('Class'))
        data = data.infer_objects()
        # split to train & test sets
        train, test = self.splitByPortion(data, 0.9)
        train_clf, train_mdp = self.splitByPortion(train, self.portion)
        return (train, train_clf, train_mdp, test)


    '''
    (629, 10)
    (70, 10)
    '''
    def breast_w(self):
        data = pd.read_csv(self.dir + 'breast-w/bc-w.data.csv', header = None)
        columns = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 
                   'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 
                   'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
        data.columns = columns
        data.drop('id', axis = 1, inplace = True)
        data = data.infer_objects()
        # fill missing value with the average of column
        for col in data.columns:
            self.fillEmpty([data], col, '?', False)

        train, test = self.splitByPortion(data, 0.9)
        train_clf, train_mdp = self.splitByPortion(train, self.portion)
        return (train, train_clf, train_mdp, test)


    '''
    (1326, 22)
    (147, 22)
    '''
    def cmc(self):
        data = pd.read_csv(self.dir + 'cmc/cmc.data.csv', header=None)
        columns = ['W_age', 'W_education', 'H_education', 'Num_children', 'W_religion', 
                   'W_working', 'H_occupation', 'Standard-of-living', 
                   'Media', 'Contraceptive method used']
        cate_columns = ['W_education', 'H_education', 'H_occupation', 'Standard-of-living']
        data.columns = columns
        for col in cate_columns:
            self.cateColumn([data], col, '?')
        data = data.infer_objects()
        train, test = self.splitByPortion(data, 0.9)
        train_clf, train_mdp = self.splitByPortion(train, self.portion)
        return (train, train_clf, train_mdp, test)


    '''
    (27000, 31)
    (3000, 31)
    '''
    def credit_card(self):
        '''
        This part reads original excel file and cast to csv
        data = pd.read_excel(self.dir + 'credit_card/credit_card.xls', header=1)
        data.drop('ID', axis = 1, inplace = True)
        data.rename(columns={'default payment next month': 'class'}, inplace=True)
        self.boolColumn([data], 'SEX', 1, 2)
        cate_columns = ['EDUCATION', 'MARRIAGE']
        for col in cate_columns:
            self.cateColumn([data], col, '0')
        data.to_csv('credit_card.csv', index=False)
        '''
        data = pd.read_csv(self.dir + 'credit_card/credit_card.csv')
        train, test = self.splitByPortion(data, 0.9)
        train_clf, train_mdp = self.splitByPortion(train, self.portion)
        return (train, train_clf, train_mdp, test)


    '''
    (329, 35)
    (37, 35)
    '''
    def dematology(self):
        data = pd.read_csv(self.dir + 'dermatology/dermatology.data.csv', header=None)
        columns = ['erythema', 'scaling', 'definite borders', 'itching', 
                   'koebner phenomenon', 'polygonal papules', 'follicular papules', 
                   'oral mucosal involvement', 'knee and elbow involvement', 
                   'scalp involvement', 'family history', 'melanin incontinence', 
                   'eosinophils in the infiltrate', 'PNL infiltrate', 
                   'fibrosis of the papillary dermis', 'exocytosis', 'acanthosis', 
                   'hyperkeratosis', 'parakeratosis', 'clubbing of the rete ridges', 
                   'elongation of the rete ridges', 'thinning of the suprapapillary epidermis', 
                   'spongiform pustule', 'munro microabcess', 'focal hypergranulosis', 
                   'disappearance of the granular layer', 'vacuolisation and damage of basal layer', 
                   'spongiosis', 'saw-tooth appearance of retes', 'follicular horn plug', 
                   'perifollicular parakeratosis', 'inflammatory monoluclear inflitrate', 
                   'band-like infiltrate', 'Age', 'Class']
        data.columns = columns
        data = data.infer_objects()
        self.fillEmpty([data], 'Age', '?', True)

        train, test = self.splitByPortion(data, 0.9)
        train_clf, train_mdp = self.splitByPortion(train, self.portion)
        return (train, train_clf, train_mdp, test)


    '''
    (302, 8)
    (34, 8)
    '''
    def ecoli(self):
        data = pd.read_csv(self.dir + 'ecoli/ecoli.data.csv', header=None)
        columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'Class']
        data.columns = columns
        data.drop('Sequence Name', axis = 1, inplace = True)
        data = data.infer_objects()
        train, test = self.splitByPortion(data, 0.9)
        train_clf, train_mdp = self.splitByPortion(train, self.portion)
        return (train, train_clf, train_mdp, test)

    '''
    (193, 10)
    (21, 10)
    '''
    def glass(self):
        data = pd.read_csv(self.dir + 'glass/glass.data.csv', header=None)
        columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']
        data.columns = columns
        data.drop('Id', axis = 1, inplace = True)
        data = data.infer_objects()
        train, test = self.splitByPortion(data, 0.9)
        train_clf, train_mdp = self.splitByPortion(train, self.portion)
        return (train, train_clf, train_mdp, test)


    '''
    (140, 20)
    (15, 20)
    '''
    def hepatitis(self):
        data = pd.read_csv(self.dir + 'hepatitis/hepatitis.data.csv', header=None)
        columns = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 
                   'MALAISE', 'ANOREXIA', 'LIVER BIG', 'LIVER FIRM', 
                   'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 
                   'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']
        bool_columns = ['STEROID', 'FATIGUE', 'MALAISE', 'ANOREXIA', 
                        'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 
                        'SPIDERS', 'ASCITES', 'VARICES']
        miss_columns = ['BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME']
        data.columns = columns
        for col in bool_columns:
            self.boolColumn([data], col, '1', '2')
        data = data.infer_objects()
        for col in miss_columns:
            self.fillEmpty([data], col, '?', True)
        # move class to last column
        col_size = data.shape[1]
        data.insert(col_size - 1, 'Class', data.pop('Class'))

        train, test = self.splitByPortion(data, 0.9)
        train_clf, train_mdp = self.splitByPortion(train, self.portion)
        return (train, train_clf, train_mdp, test)


    '''
    (7352, 562)
    (2947, 562)
    '''
    def human_activity(self):
        train = pd.read_csv('data/human-activity/train.csv')
        test = pd.read_csv('data/human-activity/test.csv')
        train.drop('subject', axis = 1, inplace = True)
        test.drop('subject', axis = 1, inplace = True)

        train_clf, train_mdp = self.splitByPortion(train, self.portion)
        return (train, train_clf, train_mdp, test)


    '''
    (270, 5)
    (30, 5)
    '''
    def iris(self):
        iris = pd.read_csv(self.dir + 'iris/iris.csv', header=None)
        bezd = pd.read_csv(self.dir + 'iris/bezdekIris.csv', header=None)
        iris = pd.concat([iris, bezd], ignore_index = True)

        train, test = self.splitByPortion(iris, 0.9)
        train_clf, train_mdp = self.splitByPortion(train, self.portion)
        return (train, train_clf, train_mdp, test)


    '''
    (133, 39)
    (15, 39)
    '''
    def lymphography(self):
        data = pd.read_csv(self.dir + 'lymphography/lymphography.data.csv', header=None)
        columns = ['class', 'lymphatics', 'block of affere', 'bl. of lymph. c', 
                   'bl. of lymph. s', 'by pass', 'extravasates', 'regeneration of', 
                   'early uptake in', 'lym.nodes dimin', 'lym.nodes enlar', 
                   'changes in lym.', 'defect in node', 'changes in node', 
                   'changes in stru', 'special forms', 'dislocation of', 
                   'exclusion of no', 'no. of nodes in']
        cate_columns = ['lymphatics', 'changes in lym.', 'defect in node', 
                        'changes in node', 'changes in stru', 'special forms']
        data.columns = columns
        for col in cate_columns:
            self.cateColumn([data], col, '?')
        data = data.infer_objects()
        # move class to last column
        col_size = data.shape[1]
        data.insert(col_size - 1, 'class', data.pop('class'))

        train, test = self.splitByPortion(data, 0.9)
        train_clf, train_mdp = self.splitByPortion(train, self.portion)
        return (train, train_clf, train_mdp, test)


    def read(self, dataset):
        data_map = {
            'audiology': self.audiology,
            'breast_cancer': self.breast_cancer,
            'breast_w': self.breast_w,
            'cmc': self.cmc,
            'credit_card': self.credit_card,
            'dematology': self.dematology,
            'ecoli': self.ecoli,
            'glass': self.glass,
            'hepatitis': self.hepatitis,
            'human_activity': self.human_activity,
            'iris': self.iris,
            'lymphography': self.lymphography
        }
        return data_map[dataset]()


if __name__ == '__main__':
    rdr = Reader(666)

