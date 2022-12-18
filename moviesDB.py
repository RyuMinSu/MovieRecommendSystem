import pandas as pd
import pymysql
import sqlalchemy
from sqlalchemy import create_engine
import warnings; warnings.filterwarnings("ignore")

lensMv = pd.read_csv(r"data\grouplens_movies.csv")
lensrt = pd.read_csv(r"data\grouples_ratings.csv")
orders = pd.read_csv(r"data\orders.csv")
tmMv = pd.read_csv(r"data\tmdb_5000_movies.csv")
customers = pd.read_csv(r"data\train_customers.csv")
locations = pd.read_csv(r"data\train_locations.csv")
vendors = pd.read_csv(r"data\vendors.csv")

def createTb(df, id, pw, host, dbName, tbName, exists):
    dbConnPath = f"mysql+pymysql://{id}:{pw}@{host}/{dbName}"
    dbConn = create_engine(dbConnPath)
    Conn = dbConn.connect()
    df.to_sql(name=tbName, con=dbConn, if_exists=exists, index=False)
    print(f"succesful create {tbName} in {dbName} DataBase")

def readTb(host, id, pw, dbName, rsql):
    conn = pymysql.connect(host=host, user=id, passwd=str(pw), db=dbName, charset="utf8")
    cur = conn.cursor()
    df = pd.read_sql(rsql, con=conn)
    print(f"df shape: {df.shape}")
    return df

#csv > db
id = ""
pw = ""
host = ""
dbName = ""
exists = ""

createTb(lensMv, id, pw, host, dbName, "lensmovie", exists)
createTb(lensrt, id, pw, host, dbName, "lensratings", exists)
createTb(orders, id, pw, host, dbName, "movieorders", exists)
createTb(tmMv, id, pw, host, dbName, "tmdb5000", exists)
createTb(customers, id, pw, host, dbName, "moviecustomer", exists)
createTb(locations, id, pw, host, dbName, "movielocation", exists)
createTb(vendors, id, pw, host, dbName, "vendors", exists)


rsql1 = "select * from lensmovie"
rsql2 = "select * from lensratings"
rsql3 = "select * from movieorders"
rsql4 = "select * from tmdb5000"
rsql5 = "select * from moviecustomer"
rsql6 = "select * from movielocation"
rsql7 = "select * from vendors"


df1 = readTb(host, id, pw, dbName, rsql1)
df2 = readTb(host, id, pw, dbName, rsql2)
df3 = readTb(host, id, pw, dbName, rsql3)
df4 = readTb(host, id, pw, dbName, rsql4)
df5 = readTb(host, id, pw, dbName, rsql5)
df6 = readTb(host, id, pw, dbName, rsql6)
df7 = readTb(host, id, pw, dbName, rsql7)