Intro to the framework
=======================

Before starting to show you how to build things, we might as well show you **what is it that will support your plots**.

The plotting backend
---------------------

`Plotly <https://plotly.com/python/>`_ is the backend used to do all the plotting. You can check all the cool things that 
can be done with it to get some inspiration. Its main strength is the interactivity it provides seamlessly, which gives
a dinamic feel to visualizations.

.. note::
    In the future, plot classes might be able to support multiple plotting backends so that users can chose their preferred one.

Sisl's wrapper
---------------

If everything was done by plotting backend, we wouldn't call this the sisl visualization module. Sisl **wraps the
plotting process** to separate the data processing steps from the rendering steps, provide scientifically meaningful plot settings
that know which steps to run when updated, as well as scientifically meaningful methods to modify the plot and support to be displayed in
sisl's graphical interface.

The `Plot` class
###############

Each representation is a python class that inherits from the :code:`Plot` class. We all have things in common, and so do plots. For this reason, we have put all the repetitive stuff in this class so that **you can focus on what makes your plot special**.
    
But wait, there's more to this class. It will **control the flow of your plots** for you so that you don't need to think about it:

*As an example, let's say you have developed a plot that reads data from a 20GB file and takes some bits of it to plot them. Now, 10 days later, another user, which is excited about the plot they got with almost no effort, wants to add a new line to the plot using the information already read. It would be a pity if the plot had to be reset and it took 5 more minutes to read the file again, right? This won't happen thanks to the :code:`Plot` class, because it automatically knows which methods to run in order to waste as little time as possible.*

This control of the flow will also **make the behaviour of all the plots consistent**, so that you can confidently use a plot developed by another user because it will be familiar to you.

This class is meant to **make your job as simple as possible**, so we encourage you to get familiar with it and understand all its possibilities.

.. note :: 
    :code:`MultiplePlot`, :code:`SubPlots` and :code:`Animation` are classes that mostly work like :code:`Plot` but are adapted to particular use cases (and support multiprocessing to keep things fast).

The `Configurable` class
#######################

Although you will probably not need to ever write this class' name in your code, it is good to know that every plot class you build automatically inherits from it. This will **make your plots automatically tunable** and it will provide them with some useful methods to **safely tweak parameters, keep a settings history**, etc...

That's all you need to know for now, you will see more about the details in other notebooks.

The `Session` class
##################

Just as :code:`Plot` is the parent of all plots, :code:`Session` is the parent of all sessions. **Sessions store plots and allow you to organize them into tabs.** They are specially useful for the `graphical user interface <https://github.com/pfebrer96/sislGUIpublic)>`_, where the users can easily see all their plots at the same time and easily modify them as they wish.

However, clicking things to create your plots may be slow and specially annoying if you have to repeat the same process time and time again. That's why you have the possibility to **create custom sessions that will do all the repetitive work with very little input**, so that all the user needs to do is enjoy the beauty of their automatically created plots in the GUI.

For an example on how to use sessions to your benefit, see `this notebook <../basic-tutorials/GUI%20with%20Python%20Demo.html>`_.
    
.. note ::
    You can find all these classes under :code:`sisl.viz.plotly`.