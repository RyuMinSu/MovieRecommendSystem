import numpy as np
import pandas as pd

import warnings; warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from items_mod import *

movies = pd.read_csv(r"data\grouplens_movies.csv", sep=",")
ratings = pd.read_csv(r"data\grouples_ratings.csv", sep=",")
print("movies shape:", movies.shape)
print("ratings shape:", ratings.shape)

movies = movies[["movieId", "title"]] #데이터 가공 (사용자-아이템: 평점)데이터프레임 만들기
ratings = ratings[["userId", "movieId", "rating"]]
df = pd.merge(ratings, movies, on="movieId", how="inner")
print("merged df shape:", df.shape)

pivotDf = pd.pivot_table(data=df, index="userId", columns="title", values="rating")
pivotDf.fillna(0, inplace=True)
print("pivot table shape:", pivotDf.shape)

pivotDft = pivotDf.T
print("transposed pivotDf shape", pivotDft.shape)

simMat = cosine_similarity(pivotDft, pivotDft) #영화별 유사도(평점기반) 구하기
print("cosine similarity type", type(simMat))
print("cosine similarity shape", simMat.shape)
simMatDf = pd.DataFrame(simMat, index=pivotDf.columns, columns=pivotDf.columns)

###개인화된 예측평점 구하기
##아직 안본 영화에도 평점 채우기: 평점이 0인곳 채우기: 예측평점: 모든 영화 사용
#예측평점 = (유사도)*dot(실제평점) / sum(abs(유사도))

rpred = pivotDf.values.dot(simMatDf.values) / np.array([np.abs(simMatDf.values).sum(axis=1)])
rpredDf = pd.DataFrame(rpred, index=pivotDf.index, columns=pivotDf.columns)

get_mse(pivotDf, rpredDf) ##실제평점과 예측평점의 차이 구하기: mse

rpred2df = rp2df(pivotDf, simMatDf)##특정 영화와 유사도가 높은 영화들로만 예측평점 구하기
print("rpred2df")

get_mse(pivotDf, rpred2df)

#9번 사용자가 어떤 영화 좋아하는지 확인
user9 = pivotDf.loc[9,:]

recMvsDf = result_df(pivotDf, rpred2df, 9, 10)
recMvsDf