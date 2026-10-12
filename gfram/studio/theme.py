"""Visual theme: dark presentation style that reads well on a projector."""

BG = '#0f141b'
SURFACE = '#171e27'
SURFACE2 = '#1f2833'
BORDER = '#2c3644'
TEXT = '#e8edf3'
MUTED = '#93a1b3'
ACCENT = '#3fa7ff'      # geometry / primary
ACCENT2 = '#b07cff'     # appearance network
GOOD = '#35c47c'
BAD = '#ff5d6c'
WARN = '#ffb547'
FUSION = '#22c3c3'

import sys

FONT = {'darwin': "'Helvetica Neue'", 'win32': "'Segoe UI'"}.get(sys.platform, 'Arial')

SERIES = [ACCENT, ACCENT2, FUSION, WARN, GOOD, BAD, '#8fa3bf']

QSS = f"""
* {{ font-family: {FONT}; font-size: 14px; color: {TEXT}; }}
QMainWindow, QWidget#page, QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; }}
QWidget#sidebar {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
QPushButton#nav {{
    text-align: left; padding: 11px 16px; border: none; border-radius: 8px;
    background: transparent; color: {MUTED}; font-size: 15px;
}}
QPushButton#nav:hover {{ background: {SURFACE2}; color: {TEXT}; }}
QPushButton#nav:checked {{ background: #1d3550; color: {TEXT}; font-weight: 600; }}
QFrame#card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; }}
QLabel#h1 {{ font-size: 26px; font-weight: 700; }}
QLabel#h2 {{ font-size: 18px; font-weight: 650; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#statValue {{ font-size: 30px; font-weight: 700; }}
QLabel#statLabel {{ color: {MUTED}; font-size: 13px; }}
QPushButton {{
    background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 14px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #24364a; }}
QPushButton:disabled {{ color: #5d6a7a; border-color: {SURFACE2}; }}
QPushButton#primary {{ background: #1f6fd1; border: 1px solid #2c80e8; font-weight: 600; }}
QPushButton#primary:hover {{ background: #2a7fe6; }}
QPushButton#primary:disabled {{ background: #1c2a3a; border-color: #24364a; color: #6b7d92; }}
QPushButton#danger {{ background: #5a1f27; border: 1px solid #8a2c38; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 10px;
}}
QComboBox QAbstractItemView {{ background: {SURFACE2}; selection-background-color: #1d3550; }}
QTableWidget, QListWidget, QTreeWidget, QPlainTextEdit, QTextBrowser {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; gridline-color: {BORDER};
}}
QHeaderView::section {{ background: {SURFACE2}; border: none; border-bottom: 1px solid {BORDER}; padding: 7px; font-weight: 600; }}
QTableWidget::item {{ padding: 4px; }}
QTableWidget::item:selected, QListWidget::item:selected {{ background: #1d3550; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; padding: 9px 16px; color: {MUTED}; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QProgressBar {{ background: {SURFACE2}; border: none; border-radius: 6px; height: 12px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QSlider::groove:horizontal {{ height: 6px; background: {SURFACE2}; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 16px; margin: -6px 0; border-radius: 8px; background: {ACCENT}; }}
QToolTip {{ background: {SURFACE2}; border: 1px solid {BORDER}; padding: 6px; }}
"""


def apply_matplotlib():
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': BORDER, 'axes.labelcolor': MUTED, 'axes.titlecolor': TEXT,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': TEXT,
        'grid.color': BORDER, 'grid.alpha': 0.7, 'axes.grid': True, 'grid.linestyle': '-',
        'axes.spines.top': False, 'axes.axisbelow': True, 'axes.spines.right': False,
        'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
        'legend.frameon': False, 'axes.prop_cycle': matplotlib.cycler(color=SERIES),
    })
