import argparse

from sisl.viz.plotly.plotutils import get_session_classes

from sisl.viz.plotly._user_customs import SESSION_FILE
from .launch import launch


def general_arguments(parser):

    parser.add_argument('--only-api', dest='only_api', action="store_true",
        help="Pass this flag if all you want to do is initialize the API, but not the GUI. You can do this if you plan"+
        "to access the graphical interface by some other means (e.g. https://sisl-siesta.xyz)"
    )

    parser.add_argument('--no-api', dest='no_api', action="store_true",
        help="Pass this flag if all you want to is to open the GUI. This will probably make sense if you already have an api running"+
        " or someone else is running the API for you."
    )


def sgui():
    """
    Command line interface for launching GUI related stuff
    """
    from sisl.viz import Session

    avail_session_classes = get_session_classes()

    parser = argparse.ArgumentParser(prog='sgui',
                                     description="Command line utility to launch sisl's graphical interface.")

    general_arguments(parser)

    parser.add_argument('--load', '-l', type=str, nargs="?", default=None,
                        help='The path to the session that you want to open in the GUI. If not provided, a fresh new session will be used.')

    for param in Session._get_class_params()[0]:
        if param.dtype is not None and not isinstance(param.dtype, str):
            parser.add_argument(f'--{param.key}', type=param.parse, required=False, help=getattr(param, "help", ""))

    subparsers = parser.add_subparsers(
        help="YOU DON'T NEED TO PASS A SESSION CLASS. You can provide a session file to load a saved session (see the --load flag)."+
        " However, if you want to start a new session and the default one (BlankSession) is not good for you"+
        " you can pass a session class. By doing so, you will also get access to session-specific settings. Try sgui BlankSession -h, for example." +
        " Note that you can also build your own sessions that will be automatically available here." +
        f" Sisl is looking to import plots defined in {SESSION_FILE}",
        dest="session_class"
    )

    for name, SessionClass in avail_session_classes.items():
        doc = SessionClass.__doc__ or ""
        specific_parser = subparsers.add_parser(name, help=doc.split(".")[0])
        general_arguments(specific_parser)
        for param in SessionClass._get_class_params()[0]:
            if param.dtype is not None and not isinstance(param.dtype, str):
                specific_parser.add_argument(f'--{param.key}', type=param.parse, required=False, help=getattr(param, "help", ""))

    args = parser.parse_args()

    # Note that it doesn't matter if we include invalid settings. Configurable will just ignore them
    settings = {key: val for key, val in vars(args).items() if val is not None}

    launch(interactive=True, load_session=args.load, session_settings=settings, session_cls=args.session_class, only_api=args.only_api, no_api=args.no_api)
