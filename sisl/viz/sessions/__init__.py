from .blank import BlankSession

from .._user_customs import get_user_sessions as _get_user_sessions

_user_sessions = _get_user_sessions()

for SessionClass in _user_sessions:
    locals()[SessionClass.__name__] = SessionClass