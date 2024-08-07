# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import sisl


def on_nodify():
    from nodify.server.session import register_file_plot_handler

    def _get_plot_options(file_name: str) -> list[str]:
        try:
            plot_handler = sisl.get_sile(file_name).plot
        except:
            return []

        options = list(plot_handler._dispatchs.keys())

        return options

    def _plot_file(file_name: str, method: str):
        plot_handler = sisl.get_sile(file_name).plot
        print(plot_handler)
        if method is None:
            plot = plot_handler()
        else:
            plot = getattr(plot_handler, method)()

        try:
            plot.get()
        except:
            pass

        return plot

    register_file_plot_handler("sisl", _get_plot_options, _plot_file)
