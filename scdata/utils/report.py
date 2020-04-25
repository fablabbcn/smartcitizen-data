from reportlab.pdfgen.canvas import Canvas
from pdfrw import PdfReader
from pdfrw.toreportlab import makerl
from pdfrw.buildxobj import pagexobj

def include_footer(input_file_path, output_file_path, link = None):

    # Get pages
    reader = PdfReader(input_file_path)
    pages = [pagexobj(p) for p in reader.pages]


    # Compose new pdf
    canvas = Canvas(output_file_path)

    for page_num, page in enumerate(pages, start=1):

        # Add page
        canvas.setPageSize((page.BBox[2], page.BBox[3]))
        canvas.doForm(makerl(canvas, page))

        # Draw footer
        footer_text = f"This report is reproducible with FAIR data. Find the source datasets here: {link}"
        x = 80
        canvas.saveState()
        canvas.setStrokeColorRGB(0, 0, 0)
        canvas.setLineWidth(0.5)
        canvas.line(66, 40, page.BBox[2] - 66, 40)
        canvas.setFont('Helvetica', 8)
        canvas.drawString(66, 25, footer_text)
        canvas.linkURL(link, (66, 25, page.BBox[2] - 66, 40))
        canvas.restoreState()

        canvas.showPage()

    canvas.save()