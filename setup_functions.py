import matplotlib.font_manager as fm
from IPython.core.display import HTML


def display_matplotlib_fonts(columns=3):
    """Displays a list of fonts currently configured in matplotlib"""

    make_html = lambda x: f"<p>{x}: <span style='font-family:{x}; font-size: 20px;'>{x}</p>"
    font_list = sorted(set([f.name for f in fm.fontManager.ttflist]))

    code = "\n".join([make_html(font) for font in font_list])

    return HTML(f"<div style='column-count: {columns};'>{code}</div>")


def extend_matplotlib_fonts(font_directory):
    """Facilitates the use of external fonts for use in matplotlib. Currently dependent on a deprecated function. Suggested
    matplotlib.font_manager.FontManager.addfont method does not appear to give expected results."""

    font_dirs = [font_directory]
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    font_list = fm.createFontList(font_files)
    fm.fontManager.ttflist.extend(font_list)
