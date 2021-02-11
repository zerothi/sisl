{% extends "!autosummary/class.rst" %}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
    {% for item in attributes %}
        {% if not item.startswith('_') %}
          ~{{ name }}.{{ item }}
	{% endif %}
    {%- endfor %}
    {% endif %}
    {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      {% if not (item.startswith('_') or item in
                         ['ArgumentParser', 'ArgumentParser_out',
			  'is_keys', 'key2case', 'keys2case',
                          'line_has_key', 'line_has_keys', 'readline',
                          'step_either', 'step_to',
			  'isDataset', 'isDimension', 'isGroup',
			  'isRoot', 'isVariable']) %}
           ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
