from surprise import *
from surprise.dataset import DatasetAutoFolds
import pandas as pd


def make_train(file_path, minrating, maxrating):
    reader = Reader(line_format="user item rating", sep=",", rating_scale=(minrating, maxrating))
    daf = DatasetAutoFolds(ratings_file=file_path, reader=reader)
    trainset = daf.build_full_trainset()
    return trainset

def unseen_movies(rating_df, userId):
    movieIds = rating_df[rating_df["userId"] == userId]["movieId"].tolist() #본영화
    allmovieIds = movies["movieId"].tolist() #전체영화
    usmovieIds = [id for id in allmovieIds if id not in movieIds] #안본영화
    print(f"평점 매긴 영화 수: {len(movieIds)}, 전체 영화 수: {len(allmovieIds)}, 추천해야할 영화 수: {len(usmovieIds)}")
    return usmovieIds

def sort_est(pred):
        return pred.est

def result_rec_df(rating_df, algorithm, userId, num):

    # def sort_est(pred):
    #     return pred.est
    testset = unseen_movies(rating_df, userId)

    predictions = [algorithm.predict(str(userId), str(itemId)) for itemId in testset]
    print(f"predictions length: {len(predictions)}")    

    predictions.sort(key=sort_est, reverse=True) #정렬

    top_predictions = predictions[:num] #10개만
    recMvid = [int(pred.iid) for pred in top_predictions] #movieId
    recMvrt = [pred.est for pred in top_predictions] #rating
    recMvtit = movies[movies["movieId"].isin(recMvid)]["title"]
    print(len(recMvid), len(recMvrt), len(recMvtit))

    topMvrt = [(id, title, rating) for id, title, rating in zip(recMvid, recMvtit, recMvrt)]

    llist = []
    for tu in topMvrt:
        slist = []
        slist.extend(tu)
        llist.append(slist)
        # print(tu)

    recDf = pd.DataFrame(llist, columns=["id", "title", "rating"])
    recDf = recDf.iloc[:,1:]
    return recDf