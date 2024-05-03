''' Implementation of csv and html export for devices in test '''

from os.path import join, dirname, exists
from os import makedirs
from scdata.tools.custom_logger import logger
import flask
from re import sub

def to_csv(self, path = None, forced_overwrite = False):
    """
    Exports devices in test to desired path
    Parameters
    ----------
        path: string
        	None
            The path (directory) to export the csv(s) into. If None, exports to test_path/processed/
        forced_overwrite: boolean
        	False
            To overwrite existing files
    Returns
    -------
        True if export successul
    """
    export_ok = True

    if path is None: epath = join(self.path, 'processed')
    else: epath = path

    # Export to csv
    for device in self.devices:
        export_ok &= device.export(epath, forced_overwrite = forced_overwrite)

    if export_ok:
        logger.info(f'Test {self.name} exported successfully')
    else:
        logger.error(f'Error while exporting test: {self.name}')

    return export_ok

def to_html(self, title = 'Your title here', template = 'sc_template.html', path = None,
            details = True, devices_summary = True, full = True, header = True):
    '''
    Generates an html description for the test
    Inspired by the code of rbardaji in: https://github.com/rbardaji/mooda
    Parameters
    ----------
        title: String
            Your title here
            Document title
        template: String
            sc_template.html
            Template to fill out (in templates/)
        path: String
            None
            Directory to export it to. If None, writes it to default test folder
        details: bool
            True
            Show test details (author, date, comments, etc.)
        devices_summary: bool
            True
            Show devices summary
        full: bool
            True
            Whether to return a full html or not
        header: bool
            True
            Whether to include a header or not
    Returns
    ----------
        rendered:
            flask rendered template
    '''
    # TODO - Update or remove
    raise NotImplementedError

    # Find the path to the html templates directory
    template_folder = join(dirname(__file__), 'templates')

    if path is None: path = join(self.path, 'export')

    if not exists(path):
        logger.info('Creating folder for test export')
        makedirs(path)

    filename = join(path, f'{self.name}.html')

    docname = sub('.','_', self.name)
    app = flask.Flask(docname, template_folder = template_folder)

    with app.app_context():
        rendered = flask.render_template(
            template,
            title = title,
            descriptor = self.descriptor,
            content = self.content,
            details = details,
            devices_summary = devices_summary,
            full = full,
            header = header
        )

    with open(filename, 'w') as handle:
        handle.write(rendered)

    logger.info (f'File saved to: {filename}')

    return filename, rendered
