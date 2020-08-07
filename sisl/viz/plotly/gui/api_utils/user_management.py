from functools import wraps
from flask_login import LoginManager, UserMixin, current_user, login_user, \
    logout_user, user_logged_in
from flask_session import Session
from flask_socketio import emit

__all__ = ["User", "if_user_can", "with_user_management", "listen_to_users"]

__WITH_USERS__ = False


class User(UserMixin):
    """
    Class used for users that are accessing the session through the GUI.
    """

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
        """
        Changes the permissions of this user.

        This method can be executed only by:
        - A user that has permission to manage_users (through the GUI).
        - The launcher of the app, by executing the python method directly
        (in the console or jupyter notebook).

        Parameters
        -----------
        new_permissions: dict
            A dictionary containing new permissions that will overwrite the old ones
        """
        # If the current_user environment variable is not present, this means the
        # method is being executed from python directly
        if 'current_user' not in locals() or current_user.has_permissions("manage_users"):
            self.permissions = {**self.permissions, **new_permissions}
        else:
            raise Exception("You don't have the rights to change user permissions.")

    def has_permissions(self, *perms):
        """
        Checks if the user has the provided permissions.

        Parameters
        -----------
        *perms: str
            The permissions that you want to check. As many as you wish.

        Returns
        ---------
        boolean
            only True if it fulfills ALL the requested permissions.
        """
        for perm in perms:
            if not self.permissions[perm]:
                return False
        else:
            return True


def if_user_can(*perms):
    """
    Wrapper that restricts actions to only users with the required permissions.

    Note that the permissions of each user are stored under user.permissions.

    Parameters
    -----------
    *perms: str
        The permissions that must be fulfilled for the user to be able to perform this action.

        As many as you wish.

    Usage
    -------
    This is to be used when responding to requests using flask-socketio.

    ```
    @socketio.on("destroy the world") # Indicates which events are we listening to
    @if_user_can("use nuclear weapons") # Restricts access to only users with needed permissions
    def destroy(*args, **kwargs):
        # All the code here will only be executed if the user has permission to use nuclear weapons
    ```
    """
    def with_permissions_check(f):

        @wraps(f)
        def wrapped(*args, **kwargs):
            if __WITH_USERS__:
                if not current_user.is_authenticated:
                    emit("auth_required")
                elif current_user.has_permissions(*perms):
                    return f(*args, **kwargs)
                else:
                    raise Exception(f"You don't have the required permissions. \n These are your permissions: {current_user.permissions}")
            else:
                return f(*args, **kwargs)
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


def listen_to_users(socketio_on):
    """
    Sets the necessary socketio event listeners to manage users.

    Parameters
    ----------
    socketio_on: socketio.on
        The function to be used to listen to socketio events
    """
    @socketio_on('login')
    def login(username):

        login_user(User(username))

        emit('logged_in')
