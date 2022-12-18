from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def get_mse(pdf, rdf):
    nzactual = pdf.values[np.nonzero(pdf.values)]
    nzpred = rdf.values[np.nonzero(pdf.values)]
    mse = mean_squared_error(nzactual, nzpred)
    print("\nactual vs pred mse:", mse)

def new_pred_mat(pdf, sdf):
    rpred2 = np.zeros(pdf.shape)    
    print("\npivotDf shape = pred shape:", pdf.shape==rpred2.shape)
    for col in range(pdf.shape[1]):
        top20sim_idx = np.argsort(sdf.values[:,col])[:-21:-1]
        for row in range(pdf.shape[0]):
            rpred2[row, col] = sdf.values[col,:][top20sim_idx].dot(pdf.values[row,:][top20sim_idx])
            rpred2[row, col] /= np.abs(sdf.values[col,:][top20sim_idx]).sum()
    return rpred2

def get_nonseen_movie(df, id):
    idSeries = df.loc[id, :]
    already_seen = idSeries[idSeries > 0].index.tolist()
    allMovies = df.columns.tolist()
    unseen = [movie for movie in allMovies if movie not in already_seen]
    return unseen

def recom_movies(df, unseenList, id, num):
    recomSeries = df.loc[id, unseenList].sort_values(ascending=False)[:num]
    return recomSeries

def rp2df(pdf, simMatDf):
    rpred2 = new_pred_mat(pdf, simMatDf)
    rpred2df = pd.DataFrame(rpred2, index=pdf.index, columns=pdf.columns)
    return rpred2df

def result_df(pdf, rpred2df, id, num):
    unseen_list = get_nonseen_movie(pdf, num)
    recMvs = recom_movies(rpred2df, unseen_list, id, num)
    recMvsDf = pd.DataFrame(recMvs.values, index=recMvs.index, columns=["pred"])
    return recMvsDf