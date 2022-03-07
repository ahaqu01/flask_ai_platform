from werkzeug.security import generate_password_hash, check_password_hash

from App.ext import db
from App.models import BaseModel

PERMISSION_NONE = 0
PERMISSION_COMMON = 1


class AdminUser(BaseModel):

    username = db.Column(db.String(32), unique=True)
    _password = db.Column(db.String(256))
    is_delete = db.Column(db.Boolean, default=False)              # 该账号是否已经被删除
    is_super = db.Column(db.Boolean, default=False)               # 是否为超级管理员，超级管理员有且只有一个，管理员可以有多个
    permission = db.Column(db.Integer, default=PERMISSION_COMMON)

    @property
    def password(self):
        raise Exception("can't access")

    @password.setter
    def password(self, password_value):
        self._password = generate_password_hash(password_value)

    def check_password(self, password_value):
        return check_password_hash(self._password, password_value)

    def check_permission(self, permission):

        return self.is_super or permission & self.permission == permission
