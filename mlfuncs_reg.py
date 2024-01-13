#################################################################################
################################################################################
import numpy as np
import pandas as pd
import functools, operator

from progressbar import ProgressBar

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import TruncatedSVD


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from seaborn import heatmap, distplot
from seaborn import displot
sns.set_style('whitegrid')



#################################################################################
#################################################################################



    
def fn_tr_ts_split_reg(df_Xy_, ts_size = 0.2, rand_state = 63):    

    df_Xy = df_Xy_
    df_X = df_Xy.iloc[:, :-1]
    
    sss = ShuffleSplit(n_splits=1, test_size=ts_size, random_state=rand_state).split(df_X)
    train_idxs, test_idxs = list(sss)[0]    
   
    return train_idxs, test_idxs
    

def fn_tr_val_ts_split_reg(df_Xy_, val_size = 0.2, ts_size = 0.2):

    idxs_tr, idxs_ts_ = fn_tr_ts_split_reg(df_Xy_, ts_size = ts_size + val_size)

    df_tr = df_Xy_.iloc[idxs_tr]
    df_ts_ = df_Xy_.iloc[idxs_ts_]

    idxs_val, idxs_ts = fn_tr_ts_split_reg(df_ts_, ts_size = ts_size/(ts_size + val_size))

    df_val = df_ts_.iloc[idxs_val]
    df_ts = df_ts_.iloc[idxs_ts]

    return df_tr, df_val, df_ts


def fn_standardize_df(df_tr_, to_transform = []):   

    def fn_X(df): return df.iloc[:, :-1].values     
    def fn_y(df): return df.iloc[:, -1].values
    def fn_df(X, y): 
        cols = df_tr_.columns[:-1]
        return pd.DataFrame(X, columns = cols).assign(labels = y)


    Xs = [fn_X(df) for df in [df_tr_, *to_transform]]
    ys = [fn_y(df) for df in [df_tr_, *to_transform]]

    scaler = StandardScaler().fit(Xs[0])
    transformed_Xs = [scaler.transform(X) for X in Xs]  

    dfs = [fn_df(X, y) for X, y in zip(transformed_Xs, ys)]
    return [*dfs, scaler]



#################################################################################
#################################################################################
def fn_hist_plot(rv, rv_name, fig_aspect = 0.4):
    fig = plt.figure(figsize=plt.figaspect(fig_aspect))

    ax1 = plt.subplot(121)
    plt.hist(rv, alpha = 0.5) 
    plt.xlabel(rv_name)
    plt.title('PDF')

    ax2 = plt.subplot(122)
    sns.kdeplot(rv, fill=True, cumulative=True,ax=ax2)
    plt.xlabel(rv_name)
    plt.title('CDF')

    plt.show()
    
    

def fn_distr_labels_reg(y_):
    s = pd.DataFrame(pd.Series(y_), columns = ['values'])
    displot(data = s, x = 'values', alpha = 0.4, kde = True)
    plt.title('LABEL_DISTRIBUTION') 
    plt.show()




def fn_feat_select_reg(df_tr_raw_, thresh_feat_label = 0, thresh_feat_feat = 1, plot = True, figsize = (15, 7)):

    
    def fn_features(label_corr):
    
        listO_other_feats = list(label_corr.index)
        top_feat = listO_other_feats.pop(0)

        return top_feat, listO_other_feats


    
    def fn_drop_corr_feats(df_corr, top_feat, listO_other_feats, thresh = 0.7):    
    
        for f in listO_other_feats:

            if df_corr.loc[top_feat, f] >= thresh:

                df_corr = df_corr.drop(f, axis = 0)
                df_corr = df_corr.drop(f, axis = 1)

        df_corr = df_corr.drop(top_feat, axis = 0)
        df_corr = df_corr.drop(top_feat, axis = 1)
        
        return df_corr


        
    
    def fn_feat_select(df_corr, thresh_label = 0.3, thresh_feat = 0.4):
    
        df_corr1 = df_corr.copy().abs()
        
        cond = df_corr1.labels >= thresh_label
        df_corr1 = df_corr1[cond]
        df_corr1 = df_corr1.loc[df_corr1.index.values, df_corr1.index.values]    

        filtered_feats = []
        label_corr = df_corr1.labels.sort_values(ascending = False).drop('labels')
        iter_feats = list(label_corr.index)

        while len(iter_feats) >= 2:

            top_feat, listO_other_feats = fn_features(label_corr)
            filtered_feats.append(top_feat)
            df_corr1 = fn_drop_corr_feats(df_corr1, top_feat, listO_other_feats, thresh = thresh_feat)

            label_corr = df_corr1.labels.sort_values(ascending = False).drop('labels')
            iter_feats = list(label_corr.index)

        return filtered_feats



    def fn_plot_corr(df_corr_mat, best_feats, figsize = (15, 7)):
    

        label_corr =  df_corr_mat.loc[best_feats].labels        
        df_corr = df_corr_mat.loc[best_feats, best_feats]
        
        if len(label_corr)==0 or len(df_corr)==0:
            print("NO GOOD FEATURES FOR THIS SETTING")
            return

        plt.figure(figsize=figsize)

        plt.subplot(1,2,1)
        label_corr.sort_values().plot(kind = 'barh', alpha = 0.6)
        plt.title('FEATURE_LABEL_CORRELATIONS (SPEARMANS)')
        plt.xlabel('Degree_of_Correlation', fontsize = 14)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.subplot(1,2,2)        
        heatmap(df_corr, annot=False, cmap="YlGnBu")
        plt.title('FEATURE_FEATURE_CORRELATIONS (SPEARMANS)')
        plt.xticks(fontsize=15, rotation = 90)
        plt.yticks(fontsize=0, rotation = 0)
        plt.tight_layout()
        plt.show()

    df_corr = df_tr_raw_.corr(method = 'spearman').abs()
    best_feats = fn_feat_select(df_corr, thresh_label = thresh_feat_label, 
                                          thresh_feat = thresh_feat_feat) 

    if plot == True:
        fn_plot_corr(df_corr, best_feats, figsize = (figsize))

    return best_feats

    


###########################################################################################################
#############################################################################################
    
def fn_scatter_feats_labels(df_tr_, n_top_feats = 4, figsize = (12, 8)):
            
    # DETERMINING APPRORIATE SUBPLOT_GRID SHAPE:
    n_cols = n_top_feats
    z = n_cols - n_cols//2
    n_rows = z if (z!=0) else z + 1
    df_tr = df_tr_

    # SCATTER_PLOTS:
    subplot_grid = (n_rows, 2)
    fig, ax = plt.subplots(*subplot_grid,  figsize = figsize,  sharey = True)      
    
    for idx in range(n_top_feats):

        x, y =  df_tr.iloc[:, idx].values, df_tr.iloc[:, -1].values
        ax.ravel()[idx].scatter(x, y, s = 40, alpha = 0.4)  
        ax.ravel()[idx].set_title(f'{df_tr.columns[idx]}', fontsize=18, weight ='bold')
       
    fig.suptitle('INDIVIDUAL_FEATURES Vs LABELS', fontsize=15, weight ='bold', y = 1.02)          
    plt.tight_layout() 
    
    
    
def fn_plot_3d_reg(df_tr_, figsize = (13, 10), listO_views = [(10, 10), (-40, 0), (70, 0), (10, 80)]):

    n_views = len(listO_views)
    z1, z2 = n_views/2, n_views//2
    z3 = z1 - z2
    n_rows = z2 if (z3==0) else z2 + 1
    subplot_grid = (n_rows, 2)
    
    df_tr = df_tr_
    cols = df_tr.columns
    fig = plt.figure(figsize = figsize)
   
    for idx, view in enumerate(listO_views):   

        ax = fig.add_subplot(*subplot_grid, idx+1, projection='3d')
        x1, x2, x3 = df_tr.iloc[:, 0], df_tr.iloc[:, 1], df_tr.iloc[:, -1],
        ax.scatter(x1, x2, x3, marker='o', s=30)         

        ax.set_xlabel(cols[0], fontsize = 20, labelpad = 10) 
        ax.set_ylabel(cols[1], fontsize = 20, labelpad = 10)
        ax.set_zlabel(cols[-1], fontsize = 20, labelpad = 10)
        ax.view_init(*view) 

    plt.tight_layout()



def fn_param_grid(param_grid_):
    return ParameterGrid(param_grid_)


def fn_train_models(X_std, y, model_class, param_grid):

    X = X_std
    trained_models = []
    pbar = ProgressBar()
    for hyp_params in pbar(param_grid):
        trained_model = model_class(**hyp_params).fit(X, y)
        trained_models.append(trained_model)
    trained_models = pd.Series(trained_models)
    return trained_models


######################################################################################
######################################################################################


def fn_reg_performance(y, y_pred):

    y, y_pred = y.ravel(), y_pred.ravel()    
    rmse = (((y-y_pred)**2).mean())**(1/2)
    mae = abs(y-y_pred).mean()
    mape = (abs(y-y_pred)/abs(y + 1e-15)).mean()
    r2 = 1 - (y - y_pred).std()**2/(y + 1e-15).std()**2 

    return [rmse, mae, mape, r2]



def fn_reg_metrics_tr_val(y_tr, y_tr_pred, y_val, y_val_pred, tr_alpha = 0.2, val_alpha = 0.45, fig_aspect = 0.4):

    rmse, mae, mape, r2 = np.array(fn_reg_performance(y_tr, y_tr_pred)).round(3)
    absolute_error, percent_error = abs(y_tr - y_tr_pred), abs(y_tr - y_tr_pred)/abs(y_tr+1e-15)

    rmse_2, mae_2, mape_2, r2_2 = np.array(fn_reg_performance(y_val, y_val_pred)).round(3)
    absolute_error_2, percent_error_2 = abs(y_val - y_val_pred), abs(y_val - y_val_pred)/abs(y_val+1e-15)

    fig = plt.figure(figsize=plt.figaspect(fig_aspect))

    ax1 = plt.subplot(121)
    sns.kdeplot(absolute_error, fill=True, cumulative=True, alpha = tr_alpha, label = 'TRAIN', ax=ax1)
    sns.kdeplot(absolute_error_2, fill=True, cumulative=True, alpha = val_alpha, label = 'VAL', ax=ax1)
    plt.vlines(mae, 0, 1, label = 'MAE TR')
    plt.vlines(mae_2, 0, 1, label = 'MAE VAL', linestyle = '--')
    plt.title('CDF ABSOLUTE ERROR')
    plt.legend(loc='lower right', prop={'weight':'bold'})

    ax2 = plt.subplot(122)
    sns.kdeplot(percent_error, fill=True, cumulative=True, alpha = tr_alpha, label = 'TRAIN', ax=ax2)
    sns.kdeplot(percent_error_2, fill=True, cumulative=True, alpha = val_alpha, label = 'VAL', ax=ax2)
    plt.vlines(mape, 0, 1, label = 'MAPE TR')
    plt.vlines(mape_2, 0, 1, label = 'MAPE VAL', linestyle = '--')
    plt.title('CDF PERCENTAGE ERROR')
    plt.legend(loc='lower right', prop={'weight':'bold'})

    plt.tight_layout()
    plt.show()





def fn_test_model_reg(df_ts_, model_):

    X_ts = df_ts_.iloc[:, :-1].values
    y_ts = df_ts_.iloc[:, -1].values
    y_ts_pred = model_.predict(X_ts)   

    df = pd.DataFrame(fn_reg_performance(y_ts, y_ts_pred)[1:])
    df.index = 'MAE MAPE R2'.split()
    df = df.T
    df.index = ['Performance: ']

    return df['MAPE R2 MAE'.split()]
    