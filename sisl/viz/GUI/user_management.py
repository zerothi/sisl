from functools import wraps
from flask_login import LoginManager, UserMixin, current_user, login_user, \
    logout_user, user_logged_in
from flask_session import Session
from flask_socketio import emit


class User(UserMixin):
    def __init__(self, id=None):
        self.id = id

        self.permissions = {
            "see": True,
            "edit": True,
            "get_new_plots": False,
            "manage_users": False,
            "load_session": False,
            "save_session": False
        }

    def change_permissions(self, new_permissions):
        if current_user.has_permissions("manage_users"):
            self.permissions = {**self.permissions, **new_permissions}
        else:
            raise Exception("You don't have the rights to change user permissions.")
    
    def has_permissions(self, *perms):
        for perm in perms:
            if not self.permissions[perm]:
                return False
        else:
            return True

def if_user_can(*perms):

    def with_permissions_check(f):

        @wraps(f)
        def wrapped(*args, **kwargs):
            if not current_user.is_authenticated:
                emit("auth_required")
            elif current_user.has_permissions(*perms):
                return f(*args, **kwargs)
            else:
                raise Exception(f"You don't have the required permissions. \n These are your permissions: {current_user.permissions}")
        return wrapped
    
    return with_permissions_check

def with_user_management(app):

    # User management
    app.config['SECRET_KEY'] = 'top-secret!'

    login_manager = LoginManager(app)
    login_manager.init_app(app)

    Session(app)

    @login_manager.user_loader
    def load_user(id):
        return User(id)

def listen_to_users(socketio_on, emit_session):

    @socketio_on('login')
    def login(username):

        login_user(User(username))

        emit_session()
