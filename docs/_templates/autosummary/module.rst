.. automodule:: {{ fullname }}
    :no-members:
    :no-undoc-members:
    :no-inherited-members:

    {% if (functions + classes + exceptions)|length > 1 %}

    {% block classes %}
    {% if classes %}

    .. autosummary::
    {% for item in classes %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block functions %}
    {% if functions %}

    .. autosummary::
    {% for item in functions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block exceptions %}
    {% if exceptions %}

    .. autosummary::
    {% for item in exceptions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% endif %}
