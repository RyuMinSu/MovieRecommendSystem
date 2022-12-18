#%%
import numpy as np
import pandas as pd

from dataprep.eda import plot, create_report

from ast import literal_eval

from sklearn.metrics.pairwise import cosine_similarity

import warnings; warnings.filterwarnings('ignore')



def weight_average(df, pct):
    pct = pct
    m = df["vote_count"].quantile(pct)
    c = df["vote_average"].mean()
    v = df["vote_count"]
    r = df["vote_average"]
    weight_average = (v / (v+m))*r + (m/(v+m))*c
    return weight_average

def weight_vote_average(df, sorted_sim, title_name, num=10):
    title_mv2 = df[df["title"] == title_name] #해당영화 보기
    title_mv2_idx = title_mv2.index.values #해당영화 본놈
    sim_idx2 = sorted_sim[title_mv2_idx, :(num*2)] #해당영화와 유사도 높은 10개놈의 인덱스
    sim_idx2 = sim_idx2.reshape(-1) #1차원으로 변경
    sim_idx2 = sim_idx2[sim_idx2 != title_mv2_idx]
    sim_mv2 = df.iloc[sim_idx2]
    return sim_mv2.sort_values(by="weight_average", ascending=False)[:10]


movies = pd.read_csv(r"C:\msbigdata\portfolio\data\tmdb_5000_movies.csv")
print(f"movies shape: {movies.shape}")

###필요한 컬럼 추출
movies_df = movies[["id", "title", "genres", "vote_average", "vote_count", "popularity", "keywords", "overview"]]
print(f"movies_df shape: {movies_df.shape}")


### 장르컬럼 원핫 인코딩
#문자열을 [{}]형태로 변환
movies_df["genres"] = movies_df["genres"].apply(literal_eval)

#name만 추출
movies_df["genres"] = movies_df["genres"].apply(lambda x: [y["name"] for y in x])

#장르 유니크값 만들기
genre_list = []
for gen in movies_df["genres"]:
    genre_list.extend(gen)
print(f"genre_list lenth: {len(genre_list)}")
genre_list = np.unique(genre_list)
print(f"unique genre_list lenth: {len(genre_list)}")

#원핫 인코딩 매트릭스 만들기
zeroMat = np.zeros(shape=(movies_df.shape[0], len(genre_list)))
print(f"zeroMat shape:{zeroMat.shape}")
zero_df = pd.DataFrame(zeroMat, columns=genre_list)

for idx, genre in enumerate(movies_df["genres"]):
    indices = zero_df.columns.get_indexer(genre)
    zero_df.iloc[idx, indices] = 1


###유사도구하기
gen_df = zero_df.copy() #사본
gen_sim = cosine_similarity(gen_df, gen_df)
print(f"gen_sim shape: {gen_sim.shape}")

#정렬
sorted_gen_sim = gen_sim.argsort()[:,::-1]

#1차 추천: 9번사람
# title_mv = movies_df[movies_df["title"] == "The Godfather"] #해당영화 보기
# title_mv_idx = title_mv.index.values #해당영화 본놈

# sim_idx = sorted_gen_sim[title_mv_idx, :10] #해당영화와 유사도 높은 10개놈의 인덱스
# sim_idx = sim_idx.reshape(-1) #1차원으로 변경

# sim_mv = movies_df.iloc[sim_idx][["title", "vote_average"]]
# print(sim_mv)


##가중평점으로 추천
#가중평점 = (v / (v+m))*r + (m/(v+m))*c
# v: 영화별 평점 투표 횟수
# m: 평점 부여를 위한 최소 투표 횟수
# r: 영화별 평균 평점
# c: 전체 영화의 평균 평점
weight_average = weight_average(movies_df, 0.6)
movies_df["weight_average"] = weight_average
movies_df[["title", "vote_average", "weight_average", "vote_count"]].sort_values(by="weight_average", ascending=False)

###추천
sim_mv2 = weight_vote_average(movies_df, sorted_gen_sim, "The Godfather")
sim_mv2 = sim_mv2[["title", "weight_average", "vote_average"]]
sim_mv2