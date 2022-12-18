import numpy as np
import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.dataset import DatasetAutoFolds

from surprise_mod import *


file_path = r"data\ratings_noh.csv"
min = 0.5
max = 5
trainset = make_train(file_path, min, max)

algo = SVD(n_epochs=20, n_factors=50, random_state=0) #알고리즘정의
algo.fit(trainset)

movies = pd.read_csv(r"data\grouplens_movies.csv")
ratings = pd.read_csv(r"data\grouples_ratings.csv")
print(f"movies shape: {movies.shape}")
print(f"ratings shape: {ratings.shape}")

#9번 사용자가 보지 않은 영화 추출
recDf = result_rec_df(ratings, algo, 9, 10)
recDf