"""
Easy conversion of data from different formats to other formats.
"""

import sys
import argparse
import sisl
from sisl.utils import cmd

from .plot import Plot
from .plotutils import get_avail_presets, get_plot_classes
from ._user_customs import PRESETS_FILE, PRESETS_VARIABLE, PLOTS_FILE, PLOTS_VARIABLE

__all__ = ['splot']

def general_arguments(parser):

    parser.add_argument('--presets', '-p', type=str, nargs="*", required=False,
                    help=f'The names of the stored presets that you want to use for the settings. Current available presets: {get_avail_presets()}')
    
    parser.add_argument('--save', '-s', type=str, required=False,
                        help='The path where you want to save the plot. Note that you can add the extension .html to save to html.')
    
    parser.add_argument('--no-show', dest='show', action='store_false',
                        help="Pass this flag if you don't want the plot to be displayed.")

def splot():

    parser = argparse.ArgumentParser(prog='splot', 
                                     description="Command utility to plot files fast. This command allows great customability." +
                                     "\n\nOnly you know how you like your plots. Therefore, a nice way to use this command is by " +
                                     "using presets that you stored previously. Note that you can either use sisl's provided presets" +
                                     f" or define your own presets. Sisl is looking for presets under the '{PRESETS_VARIABLE}' variable" +
                                     f" defined in {PRESETS_FILE}. It should be a dict containing all your presets.",
    )

    # Add default sisl version stuff
    cmd.add_sisl_version_cite_arg(parser)

    parser.add_argument('--files', "-f", type=str, nargs="*", default=[],
                        help='The files that you want to plot. As many as you want.')

    general_arguments(parser)
            
    for param in Plot._parameters:
        if param.dtype is not None and not isinstance(param.dtype, str):
            parser.add_argument(f'--{param.key}', type=param._parse, required=False, help=getattr(param, "help", ""))

    subparsers = parser.add_subparsers(
        help="YOU DON'T NEED TO PASS A PLOT CLASS. You can provide a file (see the -f flag) and sisl will decide for you."+
        " However, if you want to avoid sisl automatic choice, you can use these subcommands to select a"+
        " plot class. By doing so, you will also get access to plot-specific settings. Try sgui BandsPlot -h, for example."+
        " Note that you can also build your own plots that will be automatically available here." +
        f" Sisl is looking for your plots under the '{PLOTS_VARIABLE}' variable" +
        f" defined in {PLOTS_FILE}. It should be a list containing all your plots.",
        dest="plot_class"
    )

    avail_plots = get_plot_classes()

    for PlotClass in avail_plots:
        doc = PlotClass.__doc__ or ""
        specific_parser = subparsers.add_parser(PlotClass.plotName(), help=doc.split(".")[0], conflict_handler="resolve")
        general_arguments(specific_parser)
        for param in PlotClass._get_class_params()[0]:
            if param.dtype is not None and not isinstance(param.dtype, str):
                specific_parser.add_argument(f'--{param.key}', type=param._parse, required=False, help=getattr(param, "help", ""))
  
    args = parser.parse_args()

    # Select the plotclass that the user requested
    plot_class = Plot
    if args.plot_class:
        for PlotClass in avail_plots:
            if PlotClass.plotName() == args.plot_class:
                plot_class = PlotClass
                break

    settings = { param.key: getattr(args, param.key) for param in plot_class._get_class_params()[0] if getattr(args, param.key, None) is not None}

    print("Building plot...")
    plot = plot_class(*args.files, presets=args.presets, **settings)

    if args.save:
        print(f"Saving it to {args.save}...")
        plot.save(args.save)

    if getattr(args, "show", True):
        plot.show()
