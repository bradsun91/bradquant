from IPython.core.display import display, HTML, Javascript
import nbformat
import nbparameterise
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor, TagRemovePreprocessor
import gmbp_quant.common.utils.miscs as mu

from gmbp_quant.common.logger import LOG
logger = LOG.get_logger(__name__)


def set_width_global(width=100):
    display(HTML(f'<style>.container {{ width:{width}% !important; }} </style>'))
#


def disable_auto_scroll():
    display(Javascript('''
        IPython.OutputArea.prototype._should_scroll = function(lines) {
            return false;
        }
    '''))
#


def read_notebook(nb_input_file):
    with open(nb_input_file) as fh:
        nb = nbformat.read(fh, as_version=4)
    #
    return nb
#


def _set_params(nb, param_dict=dict(), title=None):
    orig_params = nbparameterise.extract_parameters(nb)
    params = nbparameterise.parameter_values(orig_params, **param_dict)
    nb = nbparameterise.replace_definitions(nb, params, execute=False)

    for p in params:
        if p.name == 'title':
            nb.metadata.title = p.value
            return nb
        #
    #
    if title is None:
        title = 'Notebook'
    #
    nb.metadata.title = title

    return nb
#


def export_notebook_to_html(nb=None, nb_input_file=None, nb_output_file=None, template=None):
    if nb is None:
        if nb_input_file is None:
            raise Exception('Please provide either nb or nb_input_file !')
        #
        nb = read_notebook(nb_input_file=nb_input_file)
    #

    exporter = HTMLExporter(template_file=template) if template else HTMLExporter()
    html, resources = exporter.from_notebook_node(nb)
    if nb_output_file is None:
        if nb_input_file is not None:
            nb_output_file = nb_input_file.replace('.ipynb', '.html')
        #
    #
    if nb_output_file is not None:
        with open(nb_output_file, 'w', encoding='utf-8') as fh:
            fh.write(html)
            logger.info(f'notebook exported to {nb_output_file}')
        #
    #

    return html
#


def execute_notebook(nb_input_file, nb_output_file,
                     param_dict=None, title=None,
                     run_path='', timeout=600,
                     export_html=True,
                     remove_cell_tags='remove_cell',
                     template=None):
    nb = read_notebook(nb_input_file=nb_input_file)
    if param_dict is None or not isinstance(param_dict, dict):
        new_nb = _set_params(nb=nb, title=title)
        return new_nb
    #

    new_nb = _set_params(nb=nb, param_dict=param_dict, title=title)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')

    remove_cell_tags = list(mu.iterable_to_tuple(remove_cell_tags, raw_type='str'))
    tr = TagRemovePreprocessor()
    tr.remove_cell_tags = remove_cell_tags

    ep.preprocess(new_nb, {'metadata': {'path': run_path}})
    new_nb, r = tr.preprocess(new_nb, {})

    if nb_output_file is not None and nb_output_file.endswith('.ipynb'):
        with open(nb_output_file, mode='wt') as fh:
            nbformat.write(new_nb, fh)
        #
    #
    if export_html:
        return export_notebook_to_html(nb=new_nb,
                                       nb_output_file=nb_output_file if nb_output_file is not None and nb_output_file.endswith('.html') else None,
                                       template=template)
    #

    return new_nb
#
