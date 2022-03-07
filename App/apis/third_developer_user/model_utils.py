from App.models.third_developer_user.third_developer_user_model import ThirdDeveloperUser


def get_third_developer_user(user_ident):

    if not user_ident:
        return None

    # 根据id
    user = ThirdDeveloperUser.query.get(user_ident)

    if user:
        return user

    user = ThirdDeveloperUser.query.filter(ThirdDeveloperUser.phone == user_ident).first()

    if user:
        return user

    user = ThirdDeveloperUser.query.filter(ThirdDeveloperUser.username == user_ident).first()

    if user:
        return user

    return None