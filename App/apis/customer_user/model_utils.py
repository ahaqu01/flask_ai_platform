from App.models.customer_user.customer_user_model import CustomerUser


def get_customer_user(user_ident):

    if not user_ident:
        return None

    # 根据id
    user = CustomerUser.query.get(user_ident)

    if user:
        return user

    user = CustomerUser.query.filter(CustomerUser.phone == user_ident).first()

    if user:
        return user

    user = CustomerUser.query.filter(CustomerUser.username == user_ident).first()

    if user:
        return user

    return None