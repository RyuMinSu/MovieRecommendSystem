#%%
from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split


#%%
#데이터정의
data = Dataset.load_builtin("ml-100k")
trainset, testset = train_test_split(data, test_size=.25, random_state=0)


# %%
#잠재요인 협업필터링 수행
#순서: 알고리즘 정의 > 학습 > 예측(test: 전체데이터세트, predict: 개별사용자)

#알고리즘 정의 및 학습
algo = SVD()
algo.fit(trainset) #데이터 학습



#%%
#전체예측: test
predictions = algo.test(testset) #테스트 전체에 대한 추천영화 평점 데이터 반환
print("prediction type:", type(predictions), "size:", len(predictions))
print("prediction 결과의 최초 5개 추출")
print(predictions[:5])

#반환된 객체 값 가져오기
[(pred.uid, pred.iid, pred.est) for pred in predictions[:3]]



#%%
#개별예측: predict(사용자아이디(문자형), 아이템아이디(문자형), 실제평점(선택사항))
uid = str(196) #196번 사용자
iid = str(302) #302번 아이템

pred = algo.predict(uid, iid)
print(pred)




# %%
#정확도평가: 실제평점과 예측평점간의 차이를 검사(test로 반환된 예측평점)
accuracy.rmse(predictions)




# %%

#주요API
# 1. Dataset.load_builtin: 데이터 내려받기
# 2.Dataset.load_from_file(file_path, reader): os파일에서 데이터를 로딩
#   - file_path: os파일명
#   - reader: 파일의 포맷 
# 3. Dataset.load_from_df(df, reader): 판다스 데이터프레임에서 데이터 로딩
#   - df: 데이터프레임(userId, itemId, rating 순으로 컬럼이 정해져 있어야 함)
#   - reader: 파일의 포맷



#%%
#os파일 데이터 surprise데이터 세트로 로딩
import pandas as pd

#데이터불러오기
ratings = pd.read_csv(r"data\grouples_ratings.csv")
#컬럼명 제거 후 데이터 저장
ratings.to_csv(r"C:\msbigdata\portfolio\recommedation\data\ratings_noh.csv", index=False, header=False)




# %%
#os파일 불러오기
#os파일을 불러오기 위해서는 불러오려는 데이터프레임의 원래컬럼, 평점 관련 정보를 입력해야함
from surprise import Reader

reader = Reader(line_format="user item rating timestamp", sep=",", rating_scale=(0.5, 5))
data = Dataset.load_from_file(r"data\ratings_noh.csv", reader=reader)

trainset, testset = train_test_split(data, test_size=.25, random_state=0)

algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)




# %%
#판다스 데이터프레임에서 Surprise세트로 로딩
import pandas as pd
from surprise import Reader
from surprise.model_selection import train_test_split

ratings = pd.read_csv(r"data\grouples_ratings.csv")

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader=reader)

trainset, testset = train_test_split(data, test_size=0.25, random_state=0)

algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)





# %%
# svd parameter
# n_factors = 잠재요인개수 (개수 달라져도 성능 향상 거의 없음 걍 안만지면됨)
# n_epochs = SGD수행시 반복 횟수
# biased = 베이스라인 사용자 편향 적용 여부, 디폴트는 True



#%%
#Baseline평점: 사용자의 성향을 반영하여 평점계산
#   - 수식: 전체 평균 평점 + 사용자 편향 점수 + 아이템 편향 점수
#   - 전체 평균평점: 모든 사용자의 아이템에 대한 평점을 평균한 값
#   - 사용자 편향 점수(): 사용자별 아이템 평점 평균 값 - 전체 평균 평점
#   - 아이템 편향 점수(): 아이템별 평점 평균 - 전체 평균 평점
#   - 예시
#     - 전체 평균평점(DataFrame에 rating평균): 3.5
#     - 어벤저스(아이템)의 평균평점: 4.2
#     - 윤대호의 평균평점: 3.0
#     - 윤대호가 어벤저스를 몇점줄까?: 3.5 + (3.0 - 3.5) + (4.2 - 3.5) = 3.7


#%%
#Surprise 교차검증 및 하이퍼파라미터 튜닝 cross_validate

import pandas

from surprise import SVD
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

ratings = pd.read_csv(r"data\grouples_ratings.csv", sep=",")

reader = Reader(rating_scale=(0.5, 5))

data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader=reader)

algo = SVD()
cvDict = cross_validate(algo=algo, data=data, measures=["RMSE", "MAE"], cv=5, verbose=False)
cvDict_df = pd.DataFrame(cvDict)
cvDict_df = cvDict_df.T
cvDict_df.columns = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
cvDict_df





# %%
import pandas as pd

from surprise import Dataset
from surprise import SVD
from surprise import Reader
from surprise.model_selection import GridSearchCV

ratings = pd.read_csv(r"data\grouples_ratings.csv")

reader = Reader(rating_scale=(0.5, 5))

data = Dataset.load_from_df(df=ratings[["userId", "movieId", "rating"]], reader=reader)

# algo = SVD()

param_grid = {
    "n_epochs": [20, 40, 60],
    "n_factors": [50, 100, 200]
}
gcv = GridSearchCV(algo_class=SVD, param_grid=param_grid, measures=["RMSE", "MAE"], cv=3)

gcv.fit(data)

print(gcv.best_score)
print(gcv.best_params)

gcvDf = pd.DataFrame(gcv.cv_results)
gcvDf[["params", "mean_test_rmse", "rank_test_rmse", "mean_test_mae", "rank_test_mae"]][:5]


# %%
