from werkzeug.security import generate_password_hash, check_password_hash

from App.ext import db
from App.models import BaseModel
from App.models.customer_user.customer_constant import PERMISSION_NONE

COMMON_USER = 0  # 普通用户
BLACK_USER = 1   # 黑名单用户
VIP_USER = 2     # VIP用户


class CustomerUser(BaseModel):

    username = db.Column(db.String(32), unique=True)
    _password = db.Column(db.String(256))
    phone = db.Column(db.String(32), unique=True)
    is_delete = db.Column(db.Boolean, default=False)
    permission = db.Column(db.Integer, default=PERMISSION_NONE)

    @property
    def password(self):
        raise Exception("can't access")

    @password.setter
    def password(self, password_value):
        self._password = generate_password_hash(password_value)

    def check_password(self, password_value):
        return check_password_hash(self._password, password_value)

    def check_permission(self, permission):
        if (BLACK_USER & self.permission) == BLACK_USER: #黑名单
            return False
        else:
            return permission & self.permission == permission
