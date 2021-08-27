import codecs
import pymysql
from sqlalchemy import create_engine


# 向数据库中导入数据：
# 连接数据库：
duph

db_ip = "192.168.1.162" #hst
db_user = "root" #usr
db_password = "root" #pw
db_name = "orig_article" #db

conn = pymysql.connect(db_ip,
                       db_user,
                       db_password,
                       db_name,
                       charset = "utf8",
                       autocommit = 1)
cursor = conn.cursor()
engine = create_engine("mysql+pymysql://root:root@192.168.1.162/orig_article")
df = pd.read_sql_query("SELECT * FROM article_all_brad WHERE state = 2", engine)
df_copy = df.copy()


# create/import df called merged_cols, and:
# 导入数据库：
for i in range(len(merged_cols)):
    val0 = merged_cols.iloc[i, :5][0]
    val1 = merged_cols.iloc[i, :5][1]
    val2 = merged_cols.iloc[i, :5][2]
    val3 = merged_cols.iloc[i, :5][3]
    val4 = int(merged_cols.iloc[i, :5][4])
    try:
        cursor.execute("""INSERT IGNORE INTO article_all_brad_20190910 (title, article, url, typ, state) VALUES(%s, %s, %s, %s, %s)""", (val0,val1,val2,val3,val4))
    except Exception as e:
        print(e)
    print("Saved 第{}行".format(i))
    print("===")


# 导入总数据库article:

for i in range(len(df0909_copy_authorid)):
    val0 = df0909_copy_authorid.iloc[i, :6][0]
    val1 = df0909_copy_authorid.iloc[i, :6][1]
    val2 = df0909_copy_authorid.iloc[i, :6][2]
    val3 = df0909_copy_authorid.iloc[i, :6][3]
    val4 = int(df0909_copy_authorid.iloc[i, :6][4])
    val5 = int(df0909_copy_authorid.iloc[i, :6][5])
    try:
        cursor.execute("""INSERT IGNORE INTO article (title, article, url, typ, state, authorid) VALUES(%s, %s, %s, %s, %s, %s)""", (val0,val1,val2,val3,val4,val5))
    except Exception as e:
        print(e)
    print("Saved 第{}行".format(i))
    print("===")

# ====================================================================================================

# 从数据库中导出数据：


db_ip = '127.0.0.1'
db_user = 'root'#用户
db_password = '1q2w3e4rasdf'#密码
db_name = 'market_data'#数据库
db_port = "3306"
engine = create_engine("mysql+pymysql://root:1q2w3e4rasdf@localhost:3306/market_data")

sql = "SELECT * from market_data.if000_5mins"
df = pd.read_sql_query(sql, engine)


# ====================================================================================================

# 查询去掉重复之后的数量

"""SELECT COUNT(DISTINCT title) FROM article WHERE authorid=2"""

"""SELECT COUNT(DISTINCT title) FROM article_all_brad_20190910"""

"""SELECT title,url,article,typ,state FROM send_article_cj WHERE (title LIKE '%KDJ%') AND state = 0"""

"""UPDATE send_article_cj SET state=7 WHERE (title like '%KDJ%') AND state = 0"""

"""UPDATE 20190923_brad_from_article_no_pics SET state=0 WHERE state = 6"""

"""SELECT * FROM send_article_brad where (typ like "%沪深指数%") AND state = 0"""

"""DELETE FROM send_article_brad where (typ like "%expmazhishi%") AND STATE = 0"""

"""INSERT IGNORE INTO article_all_brad_20190910 (title, article, url, typ, state) VALUES(%s, %s, %s, %s, %s)""", (val0,val1,val2,val3,val4)

# ====================================================================================================


for title in titles:
    try:
        cursor.execute("""UPDATE article SET state=7 WHERE (title like '%{}%') AND authorid = 2""".format(title))
        print(title, ": Processed")
    except Exception as e:
        print(title, ": Failed")
        print(e)



# ====================================================================================================



调用tushare之前可以直接设置：

ts.set_token('2f31c3932ead9fcc3830879132cc3ec8df3566550f711889d4a30f67')




#===========================================================================================
# 张伟版本封装函数：

# coding:utf-8
import csv
import codecs
import pymysql
import xlrd

db_ip = '192.168.1.162'
db_user = 'root'
db_password = 'root'
db_name = 'orig_article'


def get_conn():
    """数据库连接"""
    conn = pymysql.connect(db_ip,
                           db_user,
                           db_password,
                           db_name,
                           charset="utf8",
                           autocommit=1)
    return conn


def read_from_csv(path):
    """
    从csv中读取文章
    """
    articles = []
    with codecs.open(path, 'r', encoding='utf-8') as f:
        data = csv.reader(f)
        headers = next(data)  # 行标题, list
        for row in data:
            articles.append([row[0], row[1], row[2], row[3], row[4], row[5]])
    return articles


def article_to_db(articles: list):
    """
    文章入库
    """
    conn = get_conn()
    cursor = conn.cursor()
    for index, article in enumerate(articles):
        print('正在存第 {} 篇文章'.format(index + 1))
        sql = '''INSERT INTO send_gary_csai_cn(title, article, url, typ, sen_words, state)
             VALUES (%s, %s, %s, %s, %s, %s)'''
        try:
            cursor.execute(sql, article)
        except Exception as e:
            print('Exception in article_to_db: ', e)


if __name__ == '__main__':
    path = 'Content_tag_ad_update_typ_with_sen_words.csv'   # 传入要插入数据库的文章表格
    articles = read_from_csv(path)
    article_to_db(articles)


