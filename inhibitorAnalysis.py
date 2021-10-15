# Time series analysis of corrosion rate (dataInhibitor)

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.inspection import permutation_importance


matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['axes.linewidth'] = 1.5
target = {'regression': 'corrosion_mm_yr'}

# ----------------------------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------------------------
param = dict(test_size=0.25, cv=5, scoring='mse', iterations=20,
             summary=False, grid_search=False, compare=False, importance=False, parity=False,
             production=False, sensitivity=False)


# ----------------------------------------------------------------------------------------------------------------------
# Implemented Functions
# ----------------------------------------------------------------------------------------------------------------------
def stack_data(df, _set):
    conc = df['concentration_ppm'].unique()
    df2 = pd.DataFrame()
    i = 0
    for c in conc:
        if c == conc[0]:
            df2 = df.loc[df['concentration_ppm'] == c].reset_index(drop=True)
            df2['time_hrs_original'] = df2['time_hrs']
            _min, _max = df2['time_hrs'].min(), df2['time_hrs'].max()
            df2['time_hrs'] = df2['time_hrs'] - _min
            df2['pre_concentration_zero'] = 'Yes'
            df2['pre_concentration_ppm'] = 0
            if _set == 'training':
                df2['initial_corrosion_mm_yr'] = df.loc[0, 'corrosion_mm_yr']
        else:
            df3 = df.loc[df['concentration_ppm'] == c].reset_index(drop=True)
            df3['time_hrs_original'] = df3['time_hrs']
            _min, _max = df3['time_hrs'].min(), df3['time_hrs'].max()
            df3['time_hrs'] = df3['time_hrs'] - _min
            if i == 1:
                df3['pre_concentration_zero'] = 'Yes'
            else:
                df3['pre_concentration_zero'] = 'No'
            df3['pre_concentration_ppm'] = conc[i - 1]
            if _set == 'training':
                df3['initial_corrosion_mm_yr'] = df.loc[0, 'corrosion_mm_yr']
            df2 = pd.concat([df2, df3], ignore_index=True)
        i += 1
    return df2


def read_exp(df, _set):
    df.columns = df.columns.str.replace(', ', '_')
    df.columns = df.columns.str.replace(' ', '_')
    replicas = df['Description'].unique()
    df2 = pd.DataFrame()
    for replica in replicas:
        if replica == replicas[0]:
            df2 = df.loc[df['Description'] == replica].reset_index(drop=True)
            df2 = stack_data(df2, _set)
        else:
            df3 = df.loc[df['Description'] == replica].reset_index(drop=True)
            df3 = stack_data(df3, _set)
            df2 = pd.concat([df2, df3], ignore_index=True)
    return df2


def clean_data(df):
    df = df[df['corrosion_mm_yr'] >= 0.0]
    aux, aux2 = np.log10(df['corrosion_mm_yr']), np.log10(df['initial_corrosion_mm_yr'])
    df = df.drop(['corrosion_mm_yr', 'initial_corrosion_mm_yr'], axis=1)
    df['corrosion_mm_yr'], df['initial_corrosion_mm_yr'] = aux, aux2
    df = df.dropna(axis=0, how='any').reset_index(drop=True)
    df['Lab'] = df['Lab'].str.rstrip()
    df['Type_of_test'] = df['Type_of_test'].str.rstrip()
    df = df.replace({'Type_of_test': {'Sequential Dose': 'sequential_dose',
                                      'Single Dose YP': 'single_dose_YP',
                                      'Single Dose NP': 'single_dose_NP'},
                     'pH': {6: 'Controlled=6'}})
    return df


def read_data(file_name, new):
    if new:
        sheet_names = pd.ExcelFile('{}.xlsx'.format(file_name)).sheet_names
        df = pd.DataFrame()
        n = 0
        for sheet_name in sheet_names:
            if sheet_name == sheet_names[0]:
                df = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name)
                df = read_exp(df, 'training')
                df['Experiment'] = n + 1
            else:
                df2 = pd.read_excel('{}.xlsx'.format(file_name), sheet_name=sheet_name)
                df2 = read_exp(df2, 'training')
                df2['Experiment'] = n + 1
                df = pd.concat([df, df2], ignore_index=True)
            n += 1
            print(n)
        df = clean_data(df)
        excel_output(df, _root='', file_name='{}Cleaned'.format(file_name), csv=True)
    else:
        df = pd.read_csv('{}Cleaned.csv'.format(file_name))
        df = df.drop(['Unnamed: 0'], axis=1)
        n = len(df['Experiment'].unique())
    return df, n


def filter_lab(df, lab):
    if lab != 'All':
        df = df[df['Lab'] == lab].reset_index(drop=True)
    return df


def update_data(df, lab):
    df2 = filter_lab(df, lab)
    return df2


def columns_stats(df, _set, _root):
    statistics = pd.DataFrame()
    for column in df.columns:
        if (column != 'time_hrs') and (column != 'time_hrs_original') and \
                (column != 'corrosion_mm_yr') and (column != 'initial_corrosion_mm_yr'):
            if column == df.columns[0]:
                statistics = pd.DataFrame(df[column].value_counts()).reset_index(drop=False)
                statistics.rename(columns={'index': column, column: 'Num_samples'}, inplace=True)
            else:
                temp = pd.DataFrame(df[column].value_counts()).reset_index(drop=False)
                temp.rename(columns={'index': column, column: 'Num_samples'}, inplace=True)
                statistics = pd.concat([statistics, temp], axis=1)
    excel_output(statistics, _root=_root, file_name='columnsStats_{}'.format(_set), csv=False)


def experiments_stats(df, _set, _root):
    statistics = pd.DataFrame(columns=['Experiment', 'num_replica', 'CI concentration (ppm, hrs)', 'Length_hrs',
                                       'Pressure_bar_CO2', 'Temperature_C', 'CI', 'Shear_Pa',
                                       'Brine_Ionic_Strength', 'pH', 'Brine_Type', 'Type_of_test', 'Lab'])
    _experiments = df['Experiment'].unique()
    for _exp in _experiments:
        df2 = df.loc[df['Experiment'] == _exp].reset_index(drop=True)
        df3 = df2.groupby('concentration_ppm')['time_hrs'].max()
        n_replica = len(df2['Description'].unique())
        conc = df2['concentration_ppm'].unique()
        _conc_ppm = ''
        for c in conc:
            tt = df3.loc[c]
            if c == conc[0]:
                _conc_ppm = _conc_ppm + '({:.0f}, {:.0f})'.format(c, tt)
            else:
                _conc_ppm = _conc_ppm + ' - ({:.0f}, {:.0f})'.format(c, tt)
        statistics = statistics.append({'Experiment': _exp,
                                        'num_replica': n_replica,
                                        'CI concentration (ppm, hrs)': _conc_ppm,
                                        'Length_hrs': '~ {:.0f}'.format(df3.sum()),
                                        'Pressure_bar_CO2': df2.loc[0, 'Pressure_bar_CO2'],
                                        'Temperature_C': df2.loc[0, 'Temperature_C'],
                                        'CI': df2.loc[0, 'CI'],
                                        'Shear_Pa': df2.loc[0, 'Shear_Pa'],
                                        'Brine_Ionic_Strength': df2.loc[0, 'Brine_Ionic_Strength'],
                                        'pH': df2.loc[0, 'pH'],
                                        'Brine_Type': df2.loc[0, 'Brine_Type'],
                                        'Type_of_test': df2.loc[0, 'Type_of_test'],
                                        'Lab': df2.loc[0, 'Lab']}, ignore_index=True)
    excel_output(statistics, _root=_root, file_name='experimentsStats_{}'.format(_set), csv=False)


def view_data_exp(df, y_axis_scale, _set, _root):
    _root = '{}/{}{}'.format(_root, _set, y_axis_scale)
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    _experiments = df['Experiment'].unique()
    for _exp in _experiments:
        df2 = df.loc[df['Experiment'] == _exp]
        replicas = df2['Description'].unique()
        fig, ax = plt.subplots(1, figsize=(9, 9))
        _X_plot = pd.Series(dtype='float64')
        n = 1
        for rep in replicas:
            df3 = df2.loc[df['Description'] == rep]
            _X = df3['time_hrs_original']
            _y = 10 ** (df3['corrosion_mm_yr'])
            plt.scatter(_X, _y, label='Replica {}'.format(n))
            if n == 1:
                _X_plot = _X
            n += 1
        if y_axis_scale == 'Log':
            plt.yscale('log')
            # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if _exp == 14:
                ax.set_ylim(0.001, 100)
                # ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            else:
                ax.set_ylim(0.01, 100)
        # ---------------------------------
        plt.text(0.02, 1.03, 'Experiment {}'.format(_exp),
                 ha='left', va='center', transform=ax.transAxes, fontdict={'color': 'k', 'weight': 'bold', 'size': 21})
        # ---------------------------------
        plt.grid(linewidth=0.5)
        x_axis_max = 10 * (1 + int(np.max(_X_plot) / 10))
        if _exp == 6:
            x_axis_max = 40
        elif _exp == 11 or _exp == 13 or _exp == 17 or _exp == 18 or _exp == 19:
            x_axis_max = 25
        elif _exp == 14:
            x_axis_max = 30
        elif _exp == 16:
            x_axis_max = 15
        x_axis_index = np.linspace(0, x_axis_max, num=6)
        ax.set_xticks(x_axis_index)
        ax.set_xlim(0, x_axis_max)
        ax.set_xticklabels(x_axis_index, fontsize=20)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xlabel('Time (hr)', fontsize=27)
        plt.yticks(fontsize=20)
        ax.set_ylabel('Corrosion Rate (mm/year)', fontsize=27)
        n_col, leg_fontsize = 1, 20
        if _exp == 10 or _exp == 14:
            n_col, leg_fontsize = 2, 18
        plt.legend(loc='upper right', fontsize=leg_fontsize, ncol=n_col, fancybox=True, shadow=True)
        plt.tight_layout()
        plt.savefig('{}/exp{}.png'.format(_root, _exp))
        plt.close()


def experiments_types(df, y_axis_scale, _experiments, _root):
    _root = '{}/experimentsTypes{}'.format(_root, y_axis_scale)
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    for _e in _experiments:
        df2 = df.loc[df['Experiment'] == _e[0]]
        df3 = df2.loc[df['Description'] == _e[1]]
        fig, ax = plt.subplots(1, figsize=(9, 9))
        _X = df3['time_hrs_original'].to_numpy()
        _y = 10 ** (df3['corrosion_mm_yr'].to_numpy())
        marker_size = [50 + i * 0 for i in _y]
        plt.scatter(_X, _y, s=marker_size, c='black')
        if y_axis_scale == 'Log':
            plt.yscale('log')
            ax.set_ylim(0.01, 100)
            plt.yticks(fontsize=20)
        else:
            if _e[0] == 3:
                y_axis_mas = 40
            elif _e[0] == 20:
                y_axis_mas = 10
            else:
                y_axis_mas = 6
            y_axis_index = np.linspace(0, y_axis_mas, num=6)
            ax.set_yticks(y_axis_index)
            ax.set_ylim(0, y_axis_mas)
            ax.set_yticklabels(y_axis_index, fontsize=20)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        # ---------------------------------
        plt.text(0.02, 1.03, '{}'.format(_e[2]),
                 ha='left', va='center', transform=ax.transAxes, fontdict={'color': 'k', 'weight': 'bold', 'size': 21})
        # ---------------------------------
        plt.grid(linewidth=0.5)
        x_axis_index = np.linspace(0, 10 * (1 + int(np.max(_X) / 10)), num=6)
        ax.set_xticks(x_axis_index)
        ax.set_xlim(0, 10 * (1 + int(np.max(_X) / 10)))
        ax.set_xticklabels(x_axis_index, fontsize=20)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xlabel('Time (hr)', fontsize=27)
        ax.set_ylabel('Corrosion Rate (mm/year)', fontsize=27)
        plt.tight_layout()
        plt.savefig('{}/exp{}.png'.format(_root, _e[0]))
        plt.close()


def summary_data(df):
    _root = 'regression/dataSummary'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    columns_stats(df, 'allReplicas', _root)
    experiments_stats(df, 'allReplicas', _root)
    view_data_exp(df, 'Log', 'allReplicas', _root)
    view_data_exp(df, 'Normal', 'allReplicas', _root)


def excel_output(_object, _root, file_name, csv):
    if csv:
        if _root != '':
            _object.to_csv('{}/{}.csv'.format(_root, file_name))
        else:
            _object.to_csv('{}.csv'.format(file_name))
    else:
        if _root != '':
            _object.to_excel('{}/{}.xls'.format(_root, file_name))
        else:
            _object.to_excel('{}.xls'.format(file_name))


# ----------------------------------------------------------------------------------------------------------------------
def select_features(df):
    df = df[['concentration_ppm', 'pre_concentration_ppm', 'time_hrs',
             'Pressure_bar_CO2', 'Temperature_C', 'CI', 'Shear_Pa', 'Brine_Ionic_Strength',
             'pH', 'Brine_Type', 'Type_of_test', 'initial_corrosion_mm_yr', 'Description', 'Experiment',
             'corrosion_mm_yr']]
    return df


def encode_data(df):
    cat_index = ['CI', 'pH', 'Brine_Type', 'Type_of_test']
    num_index = ['Pressure_bar_CO2', 'Temperature_C', 'Shear_Pa', 'Brine_Ionic_Strength']
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    sc = StandardScaler()
    ct = make_column_transformer((ohe, cat_index), (sc, num_index), remainder='passthrough')
    ct.fit_transform(df)
    df2 = ct.transform(df)
    # ---------------------------------
    names = []
    for cat in cat_index:
        unique = df[cat].value_counts().sort_index()
        for name in unique.index:
            names.append('{}_{}'.format(cat, name))
    for num in num_index:
        names.append(num)
    names.append('concentration_ppm')
    names.append('pre_concentration_ppm')
    names.append('time_hrs')
    names.append('initial_corrosion_mm_yr')
    names.append('Description')
    names.append('Experiment')
    names.append('corrosion_mm_yr')
    # ---------------------------------
    df2 = pd.DataFrame(df2)
    df2.columns = names
    return df2


def split_data_random(df):
    df = df.copy(deep=True)
    df = shuffle(df)
    head = int((1 - param['test_size']) * len(df))
    tail = len(df) - head
    df_train = df.head(head).reset_index(drop=True)
    df_test = df.tail(tail).reset_index(drop=True)
    return df_train, df_test


def split_xy(df, _shuffle):
    if _shuffle:
        df = shuffle(df)
    df = df.drop(['Description', 'Experiment'], axis=1)
    _X = df.iloc[:, 0:-1].reset_index(drop=True)
    _y = df.iloc[:, -1].to_numpy()
    return _X, _y


def grid_search(model):
    models = []
    hp1 = {'MLP': [(2,), (4,), (6,), (8,), (10,),
                   (2, 2), (4, 4), (6, 6), (8, 8), (10, 10),
                   (2, 2, 2), (4, 4, 4), (6, 6, 6), (8, 8, 8), (10, 10, 10),
                   (2, 2, 2, 2), (4, 4, 4, 4), (6, 6, 6, 6), (8, 8, 8, 8), (10, 10, 10, 10),
                   (2, 2, 2, 2, 2), (4, 4, 4, 4, 4), (6, 6, 6, 6, 6), (8, 8, 8, 8, 8), (10, 10, 10, 10, 10)],
           'SVM': [1, 0.1, 0.01, 0.001, 0.0001],
           'RF': [10, 50, 100, 200, 500],
           'KNN': [1, 2, 3, 4, 5, 6, 7]}
    hp2 = {'MLP': ['constant'],
           'SVM': [1, 5, 10, 100, 1000],
           'RF': [0.6, 0.7, 0.8, 0.9, 1.0],
           'KNN': ['uniform', 'distance']}
    for n in hp1[model]:
        for m in hp2[model]:
            if model == 'MLP':
                models.append(('MLP_{}_{}'.format(n, m), MLPRegressor(max_iter=10000, random_state=5,
                                                                      hidden_layer_sizes=n, learning_rate=m)))
            elif model == 'SVM':
                models.append(('SVM_{}_{}'.format(n, m), SVR(gamma=n, C=m)))
            elif model == 'RF':
                models.append(('RF_{}_{}'.format(n, m), RandomForestRegressor(random_state=5,
                                                                              n_estimators=n, max_features=m)))
            elif model == 'KNN':
                models.append(('KNN_{}_{}'.format(n, m), KNeighborsRegressor(n_neighbors=n, weights=m)))
    return models


def compare_models(df, models):
    scoring, cv, iterations = 'neg_mean_squared_error', param['cv'], param['iterations']
    if param['scoring'] == 'r2':
        scoring = 'r2'
    # ---------------------------------
    results = pd.DataFrame()
    for i in range(iterations):
        _X_train, _y_train = split_xy(df, True)
        temp = []
        for name, model in models:
            print(name)
            cv_results = cross_val_score(model, _X_train, _y_train, cv=cv, scoring=scoring)
            cv_results = np.mean(cv_results)
            temp.append(cv_results)
        if i == 0:
            results = pd.DataFrame(temp)
        else:
            results = pd.concat([results, pd.DataFrame(temp)], axis=1, ignore_index=True)
    results['mean'] = results.mean(axis=1)
    results['std'] = results.std(axis=1)
    # ---------------------------------
    _names, _models = [], []
    for name, model in models:
        _names.append(name)
        _models.append(model)
    results['name'] = pd.Series(_names)
    results['model'] = pd.Series(_models)
    # ---------------------------------
    id_best = results['mean'].idxmax()
    _best = results.loc[id_best, 'model']
    return results, _best


def prediction(df, estimator):
    errors = pd.DataFrame()
    for i in range(param['iterations']):
        df_training, df_testing = split_data_random(df)
        _X_train, _y_train = split_xy(df_training, True)
        estimator.fit(_X_train, _y_train)
        _X_test, _y_test = split_xy(df_testing, True)
        _y_pred = estimator.predict(_X_test)
        errors.loc[i, 'r2'] = r2_score(_y_test, _y_pred)
        errors.loc[i, 'mse'] = mean_squared_error(_y_test, _y_pred)
        errors.loc[i, 'mae'] = mean_absolute_error(_y_test, _y_pred)
        errors.loc[i, 'rmse'] = np.sqrt(mean_squared_error(_y_test, _y_pred))
    _scores = [('R2', np.mean(errors['r2']), np.std(errors['r2'])),
               ('MSE', np.mean(errors['mse']), np.std(errors['mse'])),
               ('MAE', np.mean(errors['mae']), np.std(errors['mae'])),
               ('RMSE', np.mean(errors['rmse']), np.std(errors['rmse']))]
    return _scores


def split_data_exp(df, _seat_out):
    df_train = df.copy(deep=True)
    df_test = pd.DataFrame()
    for _exp in _seat_out:
        df_train = df_train.loc[df_train['Experiment'] != _exp]
        df_test = pd.concat([df_test, df.loc[df['Experiment'] == _exp]], ignore_index=True)
    return df_train, df_test


def production(df_x, df_xy):
    df_x_prod = df_x.copy(deep=True)
    df_xy_prod = df_xy.copy(deep=True)
    replicas = [i for i in df_x_prod['initial_corrosion_mm_yr'].unique()]
    df_prod_final = df_x_prod.loc[df_x_prod['initial_corrosion_mm_yr'] == replicas[0]]
    for replica in replicas:
        df_temp = df_x_prod.loc[df_x_prod['initial_corrosion_mm_yr'] == replica]
        df_prod_final = df_temp if len(df_temp) < len(df_prod_final) else df_prod_final
    _y_prod = []
    for replica in replicas:
        _y_temp = df_xy_prod.loc[df_xy_prod['initial_corrosion_mm_yr'] == replica].iloc[:, -1].to_numpy()
        _y_temp = _y_temp[:len(df_prod_final)]
        _y_prod = _y_temp.tolist() if replica == replicas[0] else [xx + yy for xx, yy in zip(_y_prod, _y_temp)]
    _y_prod = [i / len(replicas) for i in _y_prod]
    _y_prod = np.array(_y_prod)
    return df_prod_final, _y_prod


def sensitivity(df_original, df, _experiment):
    df_time = df_original.copy(deep=True)
    df_time = df_time.loc[df_time['Experiment'] == _experiment].reset_index(drop=True)
    replicas_time = df_time['initial_corrosion_mm_yr'].unique()
    df_time = df_time.loc[df_time['initial_corrosion_mm_yr'] == replicas_time[0]]
    time_hrs_sens = df_time['time_hrs_original']
    # ---------------------------------
    replicas = df['initial_corrosion_mm_yr'].unique()
    df = df.loc[df['initial_corrosion_mm_yr'] == replicas[0]]
    return df, time_hrs_sens


# ----------------------------------------------------------------------------------------------------------------------
def compare_models_box_plot(df):
    _root = 'regression/gridSearchModels'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    cv, iterations = param['cv'], param['iterations']
    # ---------------------------------
    x_axis_labels = [name for name in df['name']]
    df = df.drop(['name', 'mean', 'std', 'model'], axis=1)
    df = df.transform(lambda x: -x)
    _y_matrix = df.values.tolist()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    plt.boxplot(_y_matrix, labels=x_axis_labels, sym='',
                medianprops=dict(color='lightgrey', linewidth=1.0),
                meanprops=dict(linestyle='-', color='black', linewidth=1.5), meanline=True, showmeans=True)
    # ---------------------------------
    info = '{}-fold cross validation analysis \n{} replications per algorithm'.format(cv, iterations)
    plt.text(0.03, 0.96, info,
             ha='left', va='top', transform=ax.transAxes,
             fontdict={'color': 'k', 'size': 18},
             bbox={'boxstyle': 'round', 'fc': 'snow', 'ec': 'gray', 'pad': 0.5})
    # ---------------------------------
    ax.grid(axis='y', linewidth=0.35, zorder=0)
    x_axis_index = [i + 1 for i in np.arange(len(x_axis_labels))]
    ax.set_xticks(x_axis_index)
    ax.set_xticklabels(x_axis_labels, fontsize=30)
    y_axis_index = np.arange(0, 0.06, 0.01)
    ax.set_yticks(y_axis_index)
    ax.set_yticklabels(['{:.2f}'.format(i) for i in y_axis_index], fontsize=20)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=28)
    plt.tight_layout()
    plt.savefig('{}/comparison.png'.format(_root))
    plt.close()


def importance_plot(df, estimator, _x, _y):
    _root = 'regression/bestModelPerformance'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    names = df.columns
    imp = estimator.feature_importances_
    indices = np.argsort(imp)
    fig, ax = plt.subplots(1, figsize=(12, 9))
    plt.barh(range(len(indices)), imp[indices], color='black', align='center')
    x_axis_index = np.arange(0, 0.6, 0.1)
    ax.set_xticks(x_axis_index)
    ax.set_xticklabels(x_axis_index, fontsize=20)
    ax.set_xticklabels(['{:.2f}'.format(i) for i in x_axis_index], fontsize=20)
    ax.set_xlabel('Relative Importance', fontsize=30)
    plt.yticks(range(len(indices)), [names[i] for i in indices], fontsize=14)
    plt.tight_layout()
    plt.savefig('{}/featuresImp.png'.format(_root))
    plt.close()
    excel_output(pd.DataFrame(imp), _root, file_name='rf_feature_imp', csv=False)
    # ---------------------------------
    permute_imp_results = permutation_importance(estimator, _x, _y, scoring='neg_mean_squared_error')
    permute_imp = permute_imp_results.importances_mean
    excel_output(pd.DataFrame(permute_imp), _root, file_name='permutation_imp', csv=False)
    return imp, permute_imp


def parity_plot(_y_test, _y_pred, _scores):
    _root = 'regression/bestModelPerformance'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    info = '{} = {:.3f} +/- {:.3f}\n{} = {:.3f} +/- {:.3f}\n{} = {:.3f} +/- {:.3f}\n{} = {:.3f} +/- {:.3f}'. \
        format(_scores[0][0], _scores[0][1], _scores[0][2],
               _scores[1][0], _scores[1][1], _scores[1][2],
               _scores[2][0], _scores[2][1], _scores[2][2],
               _scores[3][0], _scores[3][1], _scores[3][2])
    # ---------------------------------
    fig, ax = plt.subplots(1, figsize=(9, 9))
    _y_test = 10 ** _y_test
    _y_pred = 10 ** _y_pred
    plt.scatter(_y_pred, _y_test, c='black', label='Testing set')
    a, b = min(_y_test.min(), _y_pred.min()), max(_y_test.max(), _y_pred.max())
    plt.plot([a, b], [a, b], '-', c='goldenrod', linewidth=7.0, label='y = x')
    # ---------------------------------
    plt.text(0.03, 0.96, info,
             ha='left', va='top', transform=ax.transAxes,
             fontdict={'color': 'k', 'size': 18},
             bbox={'boxstyle': 'round', 'fc': 'snow', 'ec': 'gray', 'pad': 0.5})
    # ---------------------------------
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0.01, 100)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Corrosion rate (mm/year) - Predicted', fontsize=25)
    plt.ylabel('Corrosion rate (mm/year) - True', fontsize=25)
    plt.legend(loc='upper right', fontsize=18, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig('{}/parityPlot.png'.format(_root))
    plt.close()
    # ---------------------------------
    df = pd.DataFrame(columns=['True_value', 'Predicted_value'])
    df['True_value'] = _y_test
    df['Predicted_value'] = _y_pred
    excel_output(df, _root, file_name='parityPlotData', csv=False)


def production_plot(df_all, df_selected, _y_pred, _mse_prod, folder_name, y_axis_scale, _exp, _seat_out):
    _root = 'regression/postProcessing/{}{}'.format(folder_name, y_axis_scale)
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    fig, ax = plt.subplots(1, figsize=(10, 9))
    # ---------------------------------
    df2 = df_all.copy(deep=True)
    df2 = df2.loc[df2['Experiment'] == _exp]
    replicas = df2['Description'].unique()
    n = 1
    for rep in replicas:
        df3 = df2.loc[df_all['Description'] == rep]
        _X = df3['time_hrs_original']
        _y = 10 ** (df3['corrosion_mm_yr'])
        _color, _zorder = 'gray', 0
        plt.scatter(_X, _y, c=_color, label='Replica {}'.format(n), zorder=_zorder)
        n += 1
    # ---------------------------------
    df2 = df_selected.copy(deep=True)
    df2 = df2.loc[df2['Experiment'] == _exp]
    _X_pred = df2['time_hrs_original']
    plt.scatter(_X_pred, 10 ** _y_pred, c='darkred', marker='^', s=[75], label='Prediction', zorder=7)
    # ---------------------------------
    if y_axis_scale == 'Log':
        plt.yscale('log')
        if _exp == 14:
            ax.set_ylim(0.01, 100)
        else:
            ax.set_ylim(0.01, 100)
    # ---------------------------------
    _info = [i for i in _seat_out]
    plt.grid(linewidth=0.5)
    x_axis_max = 10 * (1 + int(np.max(_X_pred) / 10))
    if _exp == 6:
        x_axis_max = 40
    elif _exp == 11 or _exp == 13 or _exp == 17 or _exp == 18 or _exp == 19:
        x_axis_max = 25
    elif _exp == 14:
        x_axis_max = 30
    elif _exp == 16:
        x_axis_max = 15
    x_axis_index = np.linspace(0, x_axis_max, num=6)
    ax.set_xticks(x_axis_index)
    ax.set_xlim(0, x_axis_max)
    ax.set_xticklabels(x_axis_index, fontsize=30)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_xlabel('Time (hr)', fontsize=40, labelpad=20)
    plt.yticks(fontsize=30)
    ax.set_ylabel('Corrosion rate (mm/yr)', fontsize=40, labelpad=25)
    n_col, legend_font_size = 1, 25
    if _exp == 10 or _exp == 14 or _exp == 29:
        n_col = 2
    if _exp == 14:
        legend_font_size = 18
    leg = plt.legend(loc='upper right', fontsize=legend_font_size, ncol=n_col, fancybox=True, shadow=True)
    for handle, text in zip(leg.legendHandles, leg.get_texts()):
        text.set_color(handle.get_facecolor()[0])
    plt.tight_layout()
    plt.savefig(f'{_root}/{_info} exp{_exp} mse({_mse_prod}).png')
    plt.close()


def sensitivity_plot(df, _exp, y_axis_scale, _feature):
    _root = 'regression/sensitivityAnalysis/variation{}{}'.format(_exp, y_axis_scale)
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    fig, ax = plt.subplots(1, figsize=(10, 9))
    # ---------------------------------
    _color = ['black', 'blue', 'green', 'olive', 'brown']
    _marker = ['o', 'x', '^', 's', 'D']
    _X = df['time_hrs']
    i = 0
    for column in df.columns:
        if column == 'time_hrs':
            continue
        _y = 10 ** (df[column])
        plt.scatter(_X, _y, c=_color[i], marker=_marker[i], s=[75], label='{}'.format(column))
        i += 1
    if y_axis_scale == 'Log':
        plt.yscale('log')
        ax.set_ylim(0.01, 100)
    # ---------------------------------
    plt.grid(linewidth=0.5)
    x_axis_max = 10 * (1 + int(np.max(_X) / 10))
    if _exp == 6:
        x_axis_max = 40
    elif _exp == 11 or _exp == 13 or _exp == 17 or _exp == 18 or _exp == 19:
        x_axis_max = 25
    elif _exp == 14:
        x_axis_max = 30
    elif _exp == 16:
        x_axis_max = 15
    x_axis_index = np.linspace(0, x_axis_max, num=6)
    ax.set_xticks(x_axis_index)
    ax.set_xlim(0, x_axis_max)
    ax.set_xticklabels(x_axis_index, fontsize=30)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_xlabel('Time (hr)', fontsize=40, labelpad=20)
    plt.yticks(fontsize=30)
    ax.set_ylabel('Corrosion rate (mm/yr)', fontsize=40, labelpad=25)
    legend_font_size = 23
    plt.legend(loc='upper right', fontsize=legend_font_size, ncol=1, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(_root, _feature))
    plt.close()
    excel_output(df, _root, file_name='{}'.format(_feature), csv=False)


# --------------------------------------------------------------------------------------------------------------------
# BEGIN
# --------------------------------------------------------------------------------------------------------------------

# reading data
dataAll, n_exp = read_data('dataInhibitor', new=False)

# data summary (one-time output)
if param['summary']:
    summary_data(df=dataAll)

# --------------------------------------------------------------------------------------------------------------------
# REGRESSION PROBLEM
# --------------------------------------------------------------------------------------------------------------------

# # pre-processing data
dataSelected = select_features(dataAll)
inhibitor = encode_data(dataSelected)

# grid-search to find the best hyper-parameters of each algorithm (one-time output)
if param['grid_search']:
    root = 'regression/gridSearchModels'
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    best_models = {}
    df_scores = pd.DataFrame()
    for algorithm in ['MLP', 'SVM', 'RF', 'KNN']:
        print(algorithm)
        algorithms = grid_search(algorithm)
        scores, best = compare_models(inhibitor, algorithms)
        best_models[algorithm] = best
        df_scores['{}_mean'.format(algorithm)], df_scores['{}_std'.format(algorithm)] = scores['mean'], scores['std']
        printOut = pd.DataFrame(algorithms)
        printOut['mean'], printOut['std'] = [-x for x in scores['mean']], scores['std']
        excel_output(printOut, root, file_name='{}'.format(algorithm), csv=False)
    models_reg = [('MLP', best_models['MLP']),
                  ('SVM', best_models['SVM']),
                  ('RF', best_models['RF']),
                  ('KNN', best_models['KNN'])]
else:
    # The following models with the set hyper-parameters are found after the one time grid searching
    models_reg = [('MLP', MLPRegressor(hidden_layer_sizes=(8, 8, 8, 8), max_iter=10000)),
                  ('SVM', SVR(C=1000, gamma=1)),
                  ('RF', RandomForestRegressor(max_features=0.7, n_estimators=500)),
                  ('KNN', KNeighborsRegressor(n_neighbors=3, weights='distance'))]

# comparing four ML models
_best_reg = models_reg[2][1]
if param['compare']:
    scores_reg, _best_reg = compare_models(inhibitor, models_reg)
    compare_models_box_plot(scores_reg)
    excel_output(scores_reg, 'regression/gridSearchModels', file_name='comparison', csv=False)
best_reg = _best_reg

# features importance
if param['importance']:
    X, y = split_xy(inhibitor, True)
    best_reg.fit(X, y)
    feature_importance, permute_importance = importance_plot(inhibitor, best_reg, X, y)

# parity plot for 25% of the data
if param['parity']:
    training_reg, testing_reg = split_data_random(inhibitor)
    X_train, y_train = split_xy(training_reg, True)
    best_reg.fit(X_train, y_train)
    X_test, y_test = split_xy(testing_reg, True)
    y_pred = best_reg.predict(X_test)
    scores_pred = prediction(inhibitor, best_reg)
    parity_plot(y_test, y_pred, scores_pred)
    excel_output(X_train, 'regression/bestModelPerformance', file_name='trainFeatureMatrixNorm', csv=False)

# testing the model when 4 experiment randomly selected to seat out
if param['production']:
    experiments = inhibitor['Experiment'].unique()
    seatOuts = [[int(i) for i in np.random.choice(a=experiments, size=4, replace=False)]]
    for seatOut in seatOuts:
        print(seatOut)
        training_test, testing_test = split_data_exp(inhibitor, seatOut)
        X_train, y_train = split_xy(training_test, True)
        best_reg.fit(X_train, y_train)
        for exp in seatOut:
            testing_temp = testing_test.loc[testing_test['Experiment'] == exp].reset_index(drop=True)
            X_test, y_test = split_xy(testing_temp, False)
            X_prod, y_prod = production(X_test, testing_temp)
            mse_prod = 0
            for iteration in range(param['iterations']):
                y_pred = best_reg.predict(X_prod)
                mse_prod += mean_squared_error(y_prod, y_pred)
            y_pred = best_reg.predict(X_prod)
            mse_prod = mse_prod / param['iterations']
            production_plot(dataAll, dataSelected, y_pred, mse_prod, f'testingTheModel/{seatOut}', 'Log', exp, seatOut)
            production_plot(dataAll, dataSelected, y_pred, mse_prod, f'testingTheModel/{seatOut}', 'Norm', exp, seatOut)

# sensitivity analysis
if param['sensitivity']:
    experiments = [int(i) for i in inhibitor['Experiment'].unique()]
    # experiments = [11]
    experiment = [i for i in np.random.choice(a=experiments, size=1, replace=False)]
    features_reg = {'CI': [['EC1612A', 'CORR12148SP'], [0.0, 0.0], 'Corrosion inhibitor', 'Inhibitor type', ''],
                    'pH': [['Controlled=6', 'Uncontrolled'], [0.0, 0.0], 'pH', 'pH', ''],
                    'Brine_Type': [['TH', 'Galapagos'], [0.0, 0.0], 'Brine type', 'Brine type', ''],
                    'Pressure_bar_CO2': [[0.5, 5, 12, 50], [4.51, 3.15],
                                         'CO2 partial pressure', '$p_{CO2}$', 'bar'],
                    'Temperature_C': [[30, 90, 110, 132], [106.69, 19.34],
                                      'Temperature', 'T', '$^oC$'],
                    'Shear_Pa': [[20, 50, 100, 200, 300], [32.85, 56.01],
                                 'Wall shear stress', '$\u03C4$', 'Pa'],
                    'Brine_Ionic_Strength': [[0.5, 1.5, 2.5, 5, 10], [0.87, 0.62],
                                             'Brine ionic strength', '$C_{Brine}$', 'M'],
                    'concentration_ppm': [[10, 100, 200, 300, 1000], [190.21, 131.99],
                                          'CI concentration', '$C_{CI}$', 'ppm']}
    for experiment in experiments:
        print(experiment)
        training_sens, testing_sens = split_data_exp(inhibitor, [experiment])
        X_train, y_train = split_xy(inhibitor, True)
        best_reg.fit(X_train, y_train)
        testing_sens, time_sens = sensitivity(dataSelected, testing_sens, experiment)
        for key in features_reg:
            print(key)
            first = True
            sensitivity_df = pd.DataFrame(index=range(len(testing_sens)))
            sensitivity_df['time_hrs'] = time_sens
            for value in features_reg[key][0]:
                testing_temp = testing_sens.copy(deep=True)
                if first and key in ['CI', 'pH', 'Brine_Type']:
                    testing_temp['{}_{}'.format(key, features_reg[key][0][0])] = [1.0] * len(testing_sens)
                    testing_temp['{}_{}'.format(key, features_reg[key][0][1])] = [0.0] * len(testing_sens)
                    first = False
                elif key in ['CI', 'pH', 'Brine_Type']:
                    testing_temp['{}_{}'.format(key, features_reg[key][0][0])] = [0.0] * len(testing_sens)
                    testing_temp['{}_{}'.format(key, features_reg[key][0][1])] = [1.0] * len(testing_sens)
                else:
                    key_mean, key_std = np.mean(dataSelected[key]), np.std(dataSelected[key])
                    zero_norm = (0 - key_mean) / float(key_std)
                    value_norm = (value - key_mean) / float(key_std)
                    if key != 'concentration_ppm':
                        testing_temp[key] = [value_norm] * len(testing_sens)
                    else:
                        for v in range(len(testing_temp)):
                            if testing_temp.loc[v, 'concentration_ppm'] != 0:
                                testing_temp.loc[v, 'concentration_ppm'] = value
                X_sens = testing_temp.drop(['Description', 'Experiment', 'corrosion_mm_yr'], axis=1)
                y_sens = best_reg.predict(X_sens)
                value = 130 if value == 132 else value
                if value == 'EC1612A':
                    sensitivity_df['{} = {} {}'.format(features_reg[key][3], 'CI-1', features_reg[key][4])] = y_sens
                elif value == 'CORR12148SP':
                    sensitivity_df['{} = {} {}'.format(features_reg[key][3], 'CI-2', features_reg[key][4])] = y_sens
                elif value == 'Controlled=6':
                    sensitivity_df['{} = {} {}'.format(features_reg[key][3], 'controlled 6',
                                                       features_reg[key][4])] = y_sens
                elif value == 'Uncontrolled':
                    sensitivity_df['{} = {} {}'.format(features_reg[key][3], 'uncontrolled',
                                                       features_reg[key][4])] = y_sens
                elif value == 'TH':
                    sensitivity_df['{} = {} {}'.format(features_reg[key][3], 'A', features_reg[key][4])] = y_sens
                elif value == 'Galapagos':
                    sensitivity_df['{} = {} {}'.format(features_reg[key][3], 'B', features_reg[key][4])] = y_sens
                else:
                    sensitivity_df['{} = {} {}'.format(features_reg[key][3], value, features_reg[key][4])] = y_sens
            sensitivity_plot(sensitivity_df, experiment, 'Log', features_reg[key][2])
            sensitivity_plot(sensitivity_df, experiment, 'Norm', features_reg[key][2])


# ----------------------------------------------------------------------------------------------------------------------
# The End
# ----------------------------------------------------------------------------------------------------------------------
print('DONE!')
