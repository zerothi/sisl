{{ fullname | escape | underline}}

{# Global variables specified for use further down #}

{% set check_ns = namespace(
      plot=false,
      to=false,
      new=false,
      apply=false,
      )
%}
{% set specials = [
      'plot',
      'apply',
      'to',
      'new',
   ]
%}

{% macro check_sections() %}
{# Check for variables that should be in specific sections #}

{% for things in varargs %}
{% for item in things %}

   {# Determine whether we should do any Plotting section #}
   {% if item.startswith('plot.') %}
      {% set check_ns.plot = true %}
   {%- endif %}

   {% if item.startswith('to.') %}
      {% set check_ns.to = true %}
   {%- endif %}

   {% if item.startswith('new.') %}
      {% set check_ns.new = true %}
   {%- endif %}

   {# Determine whether we should do any Dispatch section #}
   {% if item.startswith('apply.') %}
      {% set check_ns.apply = true %}
   {%- endif %}

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
{% set skip_methods =
                   ['ArgumentParser', 'ArgumentParser_out',
                    'is_keys', 'key2case', 'keys2case',
                    'line_has_key', 'line_has_keys', 'readline',
                    'step_either', 'step_to',
                    'isDataset', 'isDimension', 'isGroup',
                    'isRoot', 'isVariable'] + specials
%}

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
   {% if item in skip_methods %}
      {% set tmp.keep = false %}
   {%- endif %}

   {% for starts in specials %}
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
{{ check_sections(attributes, methods) }}


{#
Now the actual writing of things happen!
#}

{% if check_ns.to or check_ns.new %}
   .. rubric:: {{ _('Conversions') }}

   .. autosummary::
      {% if check_ns.new %}
      {{ name }}.new
      {% endif %}
      {{ extract_startswith('new.', false, attributes, methods) }}
      {% if check_ns.to %}
      {{ name }}.to
      {% endif %}
      {{ extract_startswith('to.', false, attributes, methods) }}

{%- endif %}

{% if check_ns.plot %}
   .. rubric:: {{ _('Plotting') }}

   .. autosummary::
      {{ name }}.plot
      {{ extract_startswith('plot.', false, attributes, methods) }}

{%- endif %}

{% if check_ns.apply %}
   .. rubric:: {{ _('Dispatching') }}

   .. autosummary::
      {{ name }}.apply
      {{ extract_startswith('apply.', false, attributes, methods) }}

{%- endif %}

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
