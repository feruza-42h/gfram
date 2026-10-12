"""Main window: sidebar navigation over the demonstration pages."""

import os
import sys

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QHBoxLayout, QLabel, QMainWindow, QSplashScreen,
                               QStackedWidget, QVBoxLayout, QWidget, QComboBox)

from . import __version__, theme
from .i18n import L, language, set_language

# Research pages (geometric analysis, live LFW benchmark, results charts) are hidden for the seminar demo
SHOW_RESEARCH_PAGES = False


class MainWindow(QMainWindow):
    def __init__(self, engine, restart):
        super().__init__()
        from .pages.about import AboutPage
        from .pages.analysis import AnalysisPage
        from .pages.benchmark import BenchmarkPage
        from .pages.database import DatabasePage
        from .pages.home import HomePage
        from .pages.identify import IdentifyPage
        from .pages.journal import JournalPage
        from .pages.updates import UpdatesPage
        from .pages.people import PeoplePage
        from .pages.live import LivePage
        from .pages.results import ResultsPage
        from .pages.verify import VerifyPage

        self.engine = engine
        self.setWindowTitle('GFRAM Studio')
        self.resize(1500, 940)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        side = QWidget()
        side.setObjectName('sidebar')
        side.setFixedWidth(275)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(14, 20, 14, 16)
        sl.setSpacing(4)
        logo = QLabel('GFRAM <span style="color:%s">Studio</span>' % theme.ACCENT)
        logo.setStyleSheet('font-size:24px;font-weight:800;padding:0 6px 2px 6px')
        sl.addWidget(logo)
        sub = QLabel(L('Gibrid yuz tanish tizimi', 'Гибридное распознавание лиц'))
        sub.setObjectName('muted')
        sub.setStyleSheet('padding:0 6px 16px 6px')
        sl.addWidget(sub)

        self.stack = QStackedWidget()
        self.pages = {}
        self.buttons = {}
        group = QButtonGroup(self)
        group.setExclusive(True)

        def add(key, icon, title, page):
            self.pages[key] = page
            self.stack.addWidget(page)
            from PySide6.QtWidgets import QPushButton
            b = QPushButton(f'{icon}   {title}')
            b.setObjectName('nav')
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda: self.go(key))
            group.addButton(b)
            sl.addWidget(b)
            self.buttons[key] = b

        self.database = DatabasePage(engine, self.db_changed)
        add('home', '◎', L('Bosh sahifa', 'Главная'), HomePage(engine, self.go))
        self.people = PeoplePage(engine, self.db_changed)
        self.journal = JournalPage(engine)
        add('people', '👤', L('Ro‘yxatga olish', 'Регистрация'), self.people)
        add('live', '●', L('Jonli tanish', 'Живое распознавание'), LivePage(engine, self.db_changed))
        add('database', '▤', L('Baza fayllari va himoya', 'Файлы базы и защита'), self.database)
        add('identify', '⌕', L('Identifikatsiya 1:N', 'Идентификация 1:N'), IdentifyPage(engine))
        add('verify', '⇄', L('Solishtirish 1:1', 'Сравнение 1:1'), VerifyPage(engine))
        add('journal', '☰', L('Tanish jurnali', 'Журнал'), self.journal)
        if SHOW_RESEARCH_PAGES:
            add('analysis', '△', L('Geometrik tahlil', 'Геометрический анализ'), AnalysisPage(engine))
            add('benchmark', '▶', L('Jonli sinov (LFW)', 'Живой тест (LFW)'), BenchmarkPage(engine))
            add('results', '▦', L('Natijalar', 'Результаты'), ResultsPage(engine))
        add('about', '★', L('Ilmiy yangilik', 'Научная новизна'), AboutPage(engine))
        add('updates', '⟳', L('Yangilanishlar', 'Обновления'), UpdatesPage(engine, self.update_found))

        sl.addStretch(1)
        lang = QComboBox()
        lang.addItem('O‘zbekcha', 'uz')
        lang.addItem('Русский', 'ru')
        lang.setCurrentIndex(0 if language() == 'uz' else 1)
        lang.currentIndexChanged.connect(lambda _: restart(lang.currentData()))
        sl.addWidget(lang)
        hint = QLabel(L('F — to‘liq ekran', 'F — полный экран'))
        hint.setObjectName('muted')
        hint.setStyleSheet('padding:6px 4px 0 4px;font-size:12px')
        sl.addWidget(hint)
        from PySide6.QtWidgets import QPushButton
        self.update_badge = QPushButton('')
        self.update_badge.setVisible(False)
        self.update_badge.setCursor(Qt.PointingHandCursor)
        self.update_badge.setStyleSheet(f'background:#4a3510;border:1px solid {theme.WARN};color:{theme.WARN};'
                                        'font-weight:700;padding:7px;border-radius:8px')
        self.update_badge.clicked.connect(lambda: self.go('updates'))
        sl.addWidget(self.update_badge)
        ver = QLabel(f'GFRAM v{__version__}')
        ver.setObjectName('muted')
        ver.setStyleSheet('padding:2px 4px;font-size:12px')
        sl.addWidget(ver)

        root.addWidget(side)
        root.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.current = None
        self.go('home')

        QShortcut(QKeySequence('F'), self, activated=self.toggle_fullscreen)
        QShortcut(QKeySequence('F11'), self, activated=self.toggle_fullscreen)
        QShortcut(QKeySequence('Esc'), self, activated=lambda: self.isFullScreen() and self.showNormal())

    def update_found(self, version):
        self.update_badge.setText(L(f'⬆ Yangi versiya: {version}', f'⬆ Новая версия: {version}'))
        self.update_badge.setVisible(True)

    def toggle_fullscreen(self):
        from PySide6.QtWidgets import QLineEdit
        if isinstance(QApplication.focusWidget(), QLineEdit):
            return
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def go(self, key):
        if self.current and self.current != key:
            leave = getattr(self.pages[self.current], 'leave', None)
            if leave:
                leave()
        self.current = key
        self.stack.setCurrentWidget(self.pages[key])
        self.buttons[key].setChecked(True)
        refresh = getattr(self.pages[key], 'refresh', None)
        if key in ('database', 'people', 'journal') and refresh:
            refresh()

    def db_changed(self):
        if self.current in ('database', 'people', 'journal'):
            self.pages[self.current].refresh()

    def closeEvent(self, e):
        for p in self.pages.values():
            leave = getattr(p, 'leave', None)
            if leave:
                leave()
        super().closeEvent(e)


def splash_pixmap():
    pm = QPixmap(560, 300)
    pm.fill(QColor(theme.BG))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    f = QFont()
    f.setPointSize(40)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor(theme.ACCENT))
    p.drawText(40, 120, 'GFRAM')
    p.setPen(QColor(theme.TEXT))
    f.setPointSize(26)
    p.setFont(f)
    p.drawText(232, 120, 'Studio')
    f.setPointSize(13)
    f.setBold(False)
    p.setFont(f)
    p.setPen(QColor(theme.MUTED))
    p.drawText(42, 160, L('Geometrik usul + sun’iy intellekt', 'Геометрический метод + искусственный интеллект'))
    p.end()
    return pm


def app_icon() -> QIcon:
    pm = QPixmap(256, 256)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(theme.SURFACE))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(8, 8, 240, 240, 54, 54)
    p.setBrush(QColor(theme.ACCENT))
    for x, y in ((88, 104), (168, 104), (128, 142), (100, 182), (156, 182), (128, 70), (70, 150), (186, 150)):
        p.drawEllipse(x - 11, y - 11, 22, 22)
    from PySide6.QtGui import QPen
    p.setPen(QPen(QColor(theme.ACCENT), 5))
    pts = [(128, 70), (88, 104), (70, 150), (100, 182), (156, 182), (186, 150), (168, 104), (128, 70)]
    for a, b in zip(pts, pts[1:]):
        p.drawLine(*a, *b)
    p.end()
    return QIcon(pm)


def ensure_camera_permission(app, parent):
    """Ask the operating system for camera access on first start; explain how to allow it if denied."""
    try:
        from PySide6.QtCore import QCameraPermission
    except ImportError:
        return
    perm = QCameraPermission()

    def denied():
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(parent)
        box.setWindowTitle('GFRAM Studio')
        box.setIcon(QMessageBox.Warning)
        box.setText(L('Kameraga ruxsat berilmagan.', 'Нет доступа к камере.'))
        box.setInformativeText(L('Sozlamalarda GFRAM Studio (yoki Terminal) uchun kamerani yoqing va dasturni qayta oching. '
                                 'Kamerasiz bo‘limlar (rasmlar bilan) ishlayveradi.',
                                 'Разрешите камеру для GFRAM Studio (или Terminal) в настройках и перезапустите программу. '
                                 'Разделы без камеры (с фото) работают и так.'))
        open_btn = box.addButton(L('Sozlamalarni ochish', 'Открыть настройки'), QMessageBox.AcceptRole)
        box.addButton(L('Keyinroq', 'Позже'), QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is open_btn:
            url = ('ms-settings:privacy-webcam' if sys.platform == 'win32'
                   else 'x-apple.systempreferences:com.apple.preference.security?Privacy_Camera')
            QDesktopServices.openUrl(QUrl(url))

    try:
        status = app.checkPermission(perm)
        if status == Qt.PermissionStatus.Undetermined:
            app.requestPermission(perm, parent, lambda p: denied() if app.checkPermission(perm) == Qt.PermissionStatus.Denied else None)
        elif status == Qt.PermissionStatus.Denied:
            denied()
    except Exception:
        pass


def check_updates_in_background(win):
    from . import updates
    from .widgets import run_task

    def work(task):
        latest = updates.latest_pypi(timeout=5)['version']
        return latest if updates.is_newer(latest, __version__) else None

    run_task(work, lambda v: v and win.update_found(v), lambda e: None)


def _quiet_logs():
    """Hide start-up chatter of MediaPipe / TensorFlow Lite / Qt / matplotlib (errors are still shown)."""
    os.environ.setdefault('GLOG_minloglevel', '2')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('QT_LOGGING_RULES', 'qt.qpa.fonts=false;qt.multimedia.ffmpeg=false;qt.multimedia=false')
    import logging
    logging.basicConfig(level=logging.ERROR)
    for name in ('matplotlib', 'matplotlib.font_manager', 'gfram'):
        logging.getLogger(name).setLevel(logging.ERROR)


def main():
    _quiet_logs()
    from . import STUDIO_HOME
    # The studio keeps its own template key, so key-rotation demos never touch ~/.gfram/keys
    os.environ.setdefault('GFRAM_KEY_DIR', str(STUDIO_HOME / 'keys'))
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    app = QApplication(sys.argv)
    app.setApplicationName('GFRAM Studio')
    app.setOrganizationName('gfram')
    app.setWindowIcon(app_icon())
    settings = QSettings()
    set_language(settings.value('language', 'uz'))
    theme.apply_matplotlib()
    app.setStyleSheet(theme.QSS)

    splash = QSplashScreen(splash_pixmap())
    splash.show()

    def msg(t):
        splash.showMessage('   ' + t, Qt.AlignBottom | Qt.AlignLeft, QColor(theme.MUTED))
        app.processEvents()

    msg(L('Modellar yuklanmoqda…', 'Загрузка моделей…'))
    from .engine import Engine
    engine = Engine()
    engine.warmup(msg)

    holder = {}

    def restart(lang):
        settings.setValue('language', lang)
        set_language(lang)
        old = holder.get('win')
        full = old.isFullScreen() if old else False
        win = MainWindow(engine, restart)
        holder['win'] = win
        win.showFullScreen() if full else win.show()
        if old:
            old.close()
            old.deleteLater()

    restart(language())
    splash.finish(holder['win'])
    QTimer.singleShot(800, lambda: ensure_camera_permission(app, holder['win']))
    QTimer.singleShot(3000, lambda: check_updates_in_background(holder['win']))
    sys.exit(app.exec())
