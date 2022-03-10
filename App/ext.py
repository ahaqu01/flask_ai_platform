from flask_mail import Mail
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
import logging

db = SQLAlchemy()
migrate = Migrate()
mail = Mail()
cache = Cache(
    config={
        "CACHE_TYPE": "redis"
    }
)

def init_ext(app):
    logging.info("in init_ext" )
    db.init_app(app)
    migrate.init_app(app, db)
    mail.init_app(app)
    cache.init_app(app)