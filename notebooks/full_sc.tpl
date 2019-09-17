{%- extends 'basic_sc.tpl' -%}
{% from 'mathjax.tpl' import mathjax %}


{%- block header -%}
<!DOCTYPE html>
<html>
<head>
{%- block html_head -%}
<meta charset="utf-8" />
{% set nb_title = nb.metadata.get('title', '') or resources['metadata']['name'] %}
<title>'{{nb_title}}'</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>

{% for css in resources.inlining.css -%}
    <style type="text/css">
    {{ css }}
    </style>
{% endfor %}

<style type="text/css">
  /* Overrides of notebook CSS for static HTML export */
  body {
    overflow: visible;
    padding: 8px;
  }

  div#notebook {
    overflow: visible;
    border-top: none;
  }

  {%- if resources.global_content_filter.no_prompt-%}
  div#notebook-container{
    padding: 6ex 12ex 8ex 12ex;
  }
  {%- endif -%}

  @media print {
    div.cell {
      display: block;
      page-break-inside: avoid;
    } 
    div.output_wrapper { 
      display: block;
      page-break-inside: avoid; 
    }
    div.output { 
      display: block;
      page-break-inside: avoid;
    }
  }
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

<!-- Loading mathjax macro -->
{{ mathjax() }}
{%- endblock html_head -%}
</head>
{%- endblock header -%}

{% block body %}
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">    
    <div class="container" id="notebook-container">
      <div style="text-align: right">
      <img src="https://smartcitizen.me/assets/images/smartcitizen_logo.svg" alt="Smart Citizen Logo" width=40px height=40px>
      </div>
      {{ super() }}
    </div>
  </div>
</body>
{%- endblock body %}

{% block footer %}
{{ super() }}
</html>
{% endblock footer %}
