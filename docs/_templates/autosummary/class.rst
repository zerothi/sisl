{{ fullname | escape | underline}}

{#

Global variables specified for use further down.

This uses the externally defined variables in the
autosummary_context variable.

   - sisl_dispatch_attributes
   - sisl_skip_methods
#}


{% set found_ns = namespace(found=[]) %}

{% macro parse_sections() %}
{#

Check for variables that should be in specific sections.
This will store in `found_ns.found` the items that are found.
It is not optimal in the sense that the searched variables
are defined in `conf.py`. But the actual parsing requires
that we have them explicitly in the docs.

#}

{% for things in varargs %}
{% for item in things %}

   {% for attr in sisl_dispatch_attributes %}
      {% if not attr in found_ns.found %}
         {% if item.startswith(attr + '.') %}
            {% set found_ns.found = found_ns.found + [attr] %}
         {%- endif %}
      {%- endif %}
   {%- endfor %}

{%- endfor %}
{%- endfor %}
{%- endmacro %}


{% macro extract_startswith(start, short) %}
{#

Write out a list of items that starts with `start`

Parameters
 start :
      what to check if the item starstwith. If yes, then keep it.
 short :
      whether it should be formatted like ~{{name}}.{{item}} (true)
      or without the ~.

#}

{% for things in varargs %}
{% for item in things %}

   {% if item.startswith(start) %}
      {% if short %}
      ~{{ name }}.{{ item }}
      {% else %}
      {{ name }}.{{ item }}
      {%- endif %}
   {%- endif %}

{%- endfor %}
{%- endfor %}
{%- endmacro %}


{% macro accepted_methods() %}

{# Loop on the arguments #}
{% for things in varargs %}
{% for item in things %}

   {% set tmp = namespace(keep=true) %}

   {#

   Use the namespace variable to pass down
   contexts. This is needed because variables
   in Jinja are hard scoped.

   #}

   {% if item.startswith('_') %}
      {% set tmp.keep = false %}
   {%- endif %}
   {% if item in sisl_skip_methods + sisl_dispatch_attributes %}
      {#
      I am actually not really sure this is needed.
      Because the `sisl_skip` method in `conf.py` already
      lists these as skipped. So likely not required...
      Oh well...
      #}
      {% set tmp.keep = false %}
   {%- endif %}

   {% for starts in sisl_dispatch_attributes %}
      {% if item.startswith(starts + '.') %}
         {% set tmp.keep = false %}
      {%- endif %}
   {%- endfor %}

   {% if tmp.keep %}
      {# Call the parent caller with the item that has
         passed all tests for eligibility
      #}
   {{ caller(item) }}
   {%- endif %}

{%- endfor %}
{%- endfor %}

{%- endmacro %}


.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
{#
   Just call the macro, this won't write anything,
   only set some variables in the check_ns namespace variable
#}
{{ parse_sections(attributes, methods) }}



{#
Now the actual writing of things happen!
#}

{% block conversion %}
{% if 'to' in found_ns.found or 'new' in found_ns.found %}
   .. rubric:: {{ _('Conversion') }}

   .. autosummary::
      :removeprefix: {{ name }}.

      {% if 'new' in found_ns.found %}
      {{ name }}.new
      {% endif %}
      {{ extract_startswith('new.', false, attributes, methods) }}
      {% if 'to' in found_ns.found %}
      {{ name }}.to
      {% endif %}
      {{ extract_startswith('to.', false, attributes, methods) }}

{%- endif %}
{%- endblock %}


{% block plotting %}
{% if 'plot' in found_ns.found %}
   .. rubric:: {{ _('Plotting') }}

   .. autosummary::
      :removeprefix: {{ name }}.

      {{ name }}.plot
      {{ extract_startswith('plot.', false, attributes, methods) }}

{%- endif %}
{%- endblock %}


{% block dispatching %}
{% if 'apply' in found_ns.found %}
   .. rubric:: :math:`k`-{{ _('point') }} {{ _('calculations') }}

   .. autosummary::
      :removeprefix: {{ name }}.

      {{ name }}.apply
      {{ extract_startswith('apply.', false, attributes, methods) }}

{%- endif %}
{%- endblock %}


{% block methods %}
{% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      {% call(item) accepted_methods(methods) %}
      ~{{ name }}.{{ item }}
      {%- endcall %}
{%- endif %}
{%- endblock %}


{% block attributes %}
{% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      {% call(item) accepted_methods(attributes) %}
      ~{{ name }}.{{ item }}
      {%- endcall %}
{%- endif %}
{%- endblock %}
