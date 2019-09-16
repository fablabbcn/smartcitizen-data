from copy import deepcopy
from IPython.utils.traitlets import Unicode
from nbconvert.preprocessors import Preprocessor

class SCPreprocessor(Preprocessor):

    expression = Unicode('True', config=True, help="Cell tag expression")

    def preprocess(self, nb, resources):

        # Loop through each cell, remove cells that don't match the query.
        remove_indicies = []
        for index, cell in enumerate(nb.cells):

            if not self.validate_cell_tags(cell):
                print ('[SC Preprocessor]: Removing cell with tag {}'.format(cell['metadata']))
                remove_indicies.append(index)

        for index in remove_indicies[::-1]:
            del nb.cells[index]

        resources['notebook_copy'] = deepcopy(nb)

        return nb, resources


    def validate_cell_tags(self, cell):
        if 'tags' in cell['metadata']:
            return self.eval_tag_expression(cell['metadata']['tags'], self.expression)
        return False

    def eval_tag_expression(self, tags, expression):
        
        # Create the tags as True booleans.  This allows us to use python 
        # expressions.
        for tag in tags:
            exec (tag + " = True")

        # Attempt to evaluate expression.  If a variable is undefined, define
        # the variable as false.
        while True:
            try:
                return eval(expression)
            except NameError as Error:
                exec (str(Error).split("'")[1] + " = False")
