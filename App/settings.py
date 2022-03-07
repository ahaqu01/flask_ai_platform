import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_db_uri(dbinfo):

    engine = dbinfo.get("ENGINE") or "sqlite"
    driver = dbinfo.get("DRIVER") or "sqlite"
    user = dbinfo.get("USER") or ""
    password = dbinfo.get("PASSWORD") or ""
    host = dbinfo.get("HOST") or ""
    port = dbinfo.get("PORT") or ""
    name = dbinfo.get("NAME") or ""

    return "{}+{}://{}:{}@{}:{}/{}".format(engine, driver, user, password, host, port, name)


class Config:

    DEBUG = False

    TESTING = False

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    SECRET_KEY = "BEFKJJIOAEJIOTEWTJOWIENETWJIORTwejiontwji0o"


'''
mysql: 先创建数据库:
    1.登入：mysql -u root -p # 输入密码登入
    2、>create database dbname charset=utf8;
'''

class DevelopConfig(Config):

    DEBUG = True

    dbinfo = {
        "ENGINE": "mysql",
        "DRIVER": "pymysql",
        "USER": "root",
        "PASSWORD": "AQ#123",
        "HOST": "localhost",
        "PORT": "3306",
        "NAME": "flask_deeplearn01"
    }

    MAIL_SERVER = "smtp.163.com"

    MAIL_PORT = 25

    MAIL_USERNAME = "zhouyuhua_ict@163.com" #邮箱

    MAIL_PASSWORD = "authorization code"   #授权码，不是邮箱密码

    MAIL_DEFAULT_SENDER = MAIL_USERNAME

    # ACCESS_KEY_ID/ACCESS_KEY_SECRET 根据实际申请的账号信息进行替换
    # ACCESS_KEY_ID = "AK"
    # ACCESS_KEY_SECRET = "SK"

    SQLALCHEMY_DATABASE_URI = get_db_uri(dbinfo)


class TestConfig(Config):
    TESTING = True

    dbinfo = {
        "ENGINE": "mysql",
        "DRIVER": "pymysql",
        "USER": "user",
        "PASSWORD": "password",
        "HOST": "localhost",
        "PORT": "3306",
        "NAME": "flask_deeplearn01"
    }

    SQLALCHEMY_DATABASE_URI = get_db_uri(dbinfo)


class StagingConfig(Config):

    dbinfo = {
        "ENGINE": "mysql",
        "DRIVER": "pymysql",
        "USER": "root",
        "PASSWORD": "AQ#123",
        "HOST": "localhost",
        "PORT": "3306",
        "NAME": "flask_deeplearn01"
    }

    SQLALCHEMY_DATABASE_URI = get_db_uri(dbinfo)


class ProductConfig(Config):

    dbinfo = {
        "ENGINE": "mysql",
        "DRIVER": "pymysql",
        "USER": "root",
        "PASSWORD": "AQ#123",
        "HOST": "localhost",
        "PORT": "3306",
        "NAME": "flask_deeplearn01"
    }

    SQLALCHEMY_DATABASE_URI = get_db_uri(dbinfo)


envs = {
    "develop": DevelopConfig,
    "testing": TestConfig,
    "staging": StagingConfig,
    "product": ProductConfig,
    "default": DevelopConfig
}

ADMINS = ('admin', 'zhou')

FILE_PATH_PREFIX = "/static/uploads/"

UPLOADS_DIR = os.path.join(BASE_DIR, 'App/static/uploads/')