"""
Easy conversion of data from different formats to other formats.
"""

import sys
import argparse
import sisl
from sisl.utils import cmd

from .plot import Plot
from .plotutils import get_avail_presets
from ._user_customs import PRESETS_FILE, PRESETS_VARIABLE

__all__ = ['splot']

def splot(argv=None, sile=None):

    parser = argparse.ArgumentParser(prog='splot', 
                                     description="Command utility to plot files fast. This command allows great customability." +
                                     "\n\nOnly you know how you like your plots. Therefore, a nice way to use this command is by " +
                                     "using presets that you stored previously. Note that you can either use sisl's provided presets" +
                                     f" or define your own presets. Sisl is looking for presets under the '{PRESETS_VARIABLE}' variable" +
                                     f" defined in {PRESETS_FILE}. It should be a dict containing all your presets."
    )
  
    parser.add_argument('files', type=str, nargs="*", default=None,
                        help='The files that you want to plot. As many as you want')

    parser.add_argument('--presets', '-p', type=str, nargs="*", required=False,
                        help=f'The names of the stored presets that you want to use for the settings. Current available presets: {get_avail_presets()}')
    
    parser.add_argument('--save', '-s', type=str, required=False,
                        help='The path where you want to save the plot. Note that you can add the extension .html to save to html.')
    
    parser.add_argument('--no-show', dest='show', action='store_false',
                        help="Pass this flag if you don't want the plot to be displayed.")
            
    for param in Plot._parameters:
        if param.dtype is not None and not isinstance(param.dtype, str):
            parser.add_argument(f'--{param.key}', type=param._parse, required=False, help=getattr(param, "help", ""))
  
    args = parser.parse_args()

    settings = { param.key: getattr(args, param.key) for param in Plot._parameters if getattr(args, param.key, None) is not None}

    # Add default sisl version stuff
    cmd.add_sisl_version_cite_arg(parser)

    print("Building plot...")
    plot = Plot(*args.files, presets=args.presets, **settings)

    if args.save:
        print(f"Saving it to {args.save}...")
        plot.save(args.save)

    if getattr(args, "show", True):
        plot.show()
