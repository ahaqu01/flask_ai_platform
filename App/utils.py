import uuid

ADMIN_USER = "admin_user" # 管理员用户
THIRD_DEVELOPER_USER = "third_developer_user" # 第三方开发者用户
CUSTOM_USER = "custom_user" # 客户用户


def generate_token(prefix=None):
    token = prefix + uuid.uuid4().hex
    return token

def generate_admin_user_token():
    return generate_token(prefix=ADMIN_USER)

def generate_third_developer_user_token():
    return generate_token(prefix=ADMIN_USER)

def generate_customer_user_token():
    return generate_token(prefix=CUSTOM_USER)

