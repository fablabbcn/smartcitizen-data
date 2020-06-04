{%- extends 'templates/basic_sc.tpl' -%}
{% from 'mathjax.tpl' import mathjax %}


{%- block header -%}
<!DOCTYPE html>
<html>
<head>
{%- block html_head -%}
<meta charset="utf-8" />
{% set nb_title = nb.metadata.get('title', '') or resources['metadata']['name'] %}
<title>Smart Citizen Delivery</title>
<link rel="shortcut icon" type="image/png" href="https://smartcitizen.me/assets/images/smartcitizen_logo.svg"/>
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
    margin: 15px;
  }

  div.prompt{
    display:none;
  }
  div.input_prompt{
    display:none;
  }

  div#notebook {
    overflow: visible;
    border-top: none;
    padding-top: 0px;
  }

  div#notebook-container {
    padding: 30px 60px;
    overflow: visible;
    border-top: none;
  }

  div.container {
    margin-top: 0px;
    width: 100%;
  }

  {%- if resources.global_content_filter.no_prompt-%}
  div#notebook-container{
    padding: 6ex 12ex 8ex 12ex;
  }
  {%- endif -%}

  div.text_cell_render{
    padding: 0px;
  }

  div.cell{
    padding: 0px;
  }

  div.output_area pre {
    display: none !important;
  }

  div.output_subarea {
    max-width: 100% !important;
  }

  .rendered_html p {
    text-align: justify;
  }

  div#report-header {
    display: flex;
    flex-direction: row;
    -ms-flex-align: center;
    -webkit-align-items: center;
    align-items: center;
    -ms-flex-pack: justify;
    -webkit-justify-content: space-between;
    justify-content: space-between;
    color: #FFFFFF;
    height: 120px;
    padding-left: 60px;
    overflow: hidden;
    border-top: none;
    background: #000000;
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    border-radius: 5px 5px 0px 0px;
    padding-right: 60px;
  }
    
  h1#top {
    margin: 0px !important;
    padding: 0px !important;
    font-size: 250% !important;
    font-weight: bold;
  }

  @page  
  { 
    size: auto;   /* auto is the initial value */ 

    /* this affects the margin in the printer settings */ 
    margin-top: 10mm;
    margin-bottom: 10mm;
    margin-left: 0mm !important;
    margin-right: 0mm !important;
    
    :first {
        margin-top: 0mm !important;
        margin-bottom: 0mm !important;
    }
  } 

  @media print {
    body{
      padding: 0mm !important;
      margin: 0mm !important;
    }

    div#report-header{
      background: #000000 !important;
      font-size: 90% !important;
      page-break-inside: avoid;
      border-radius: 0px 0px 0px 0px;
    }

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

<!-- Loading mathjax macro -->
{{ mathjax() }}
{%- endblock html_head -%}
</head>
{%- endblock header -%}

{% block body %}
<body>
    <div id="report-header">
      {% set nb_title = nb.metadata.get('title', '') or resources['metadata']['name'] %}
      <h1 id="top">Delivery report</h1>
      
      <div style="text-align: right">
        <img src="https://smartcitizen.me/assets/images/smartcitizen_logo.svg" alt="Smart Citizen Logo" width=50px height=50px>
      </div>
  </div>

  <div tabindex="-1" id="notebook" class="border-box-sizing">    
    <div class="container" id="notebook-container">
      {{ super() }}
    </div>
  </div>

</body>
{%- endblock body %}

{% block footer %}
{{ super() }}
</html>
{% endblock footer %}
