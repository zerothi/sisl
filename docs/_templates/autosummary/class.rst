{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:
    :inherited-members:

    {% block attributes %}
    {% if attributes %}
    .. rubric:: Attributes

    .. autosummary::
    {% for item in attributes %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block methods %}
    {% if methods %}
    .. rubric:: Methods

    .. autosummary::
    {% for item in methods %}
       {% if item not in ['ArgumentParser', 'ArgumentParser_out',
			  'is_keys', 'key2case', 'keys2case',
                          'line_has_key', 'line_has_keys', 'readline',
                          'step_either', 'step_to',
			  'isDataset', 'isDimension', 'isGroup',
			  'isRoot', 'isVariable'] %}
            ~{{ name }}.{{ item }}
       {% endif %}
    {%- endfor %}
    {% endif %}
    {% endblock %}
