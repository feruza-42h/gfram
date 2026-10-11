"""Overview: what the system is, how the hybrid pipeline is built, the headline numbers."""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from .. import theme
from ..i18n import L
from ..widgets import StatTile, button, card, label


class PipelineDiagram(QWidget):
    """Architecture of the hybrid system, drawn with QPainter so it scales on any projector."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(390)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        col_w = W / 5.0
        bw, bh = col_w * 0.84, 82

        def box(cx, cy, title, sub, color, h=bh):
            r = QRectF(cx - bw / 2, cy - h / 2, bw, h)
            c = QColor(color)
            fill = QColor(color)
            fill.setAlpha(38)
            p.setPen(QPen(c, 2))
            p.setBrush(fill)
            p.drawRoundedRect(r, 10, 10)
            f = QFont()
            f.setPointSize(12)
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(theme.TEXT))
            p.drawText(r.adjusted(6, 8, -6, -h * 0.55), Qt.AlignHCenter | Qt.AlignVCenter | Qt.TextWordWrap, title)
            f.setPointSize(10)
            f.setBold(False)
            p.setFont(f)
            p.setPen(QColor(theme.MUTED))
            p.drawText(r.adjusted(6, h * 0.45, -6, -6), Qt.AlignHCenter | Qt.AlignVCenter | Qt.TextWordWrap, sub)
            return r

        def arrow(a: QPointF, b: QPointF, color=theme.MUTED):
            p.setPen(QPen(QColor(color), 2))
            path = QPainterPath(a)
            mx = (a.x() + b.x()) / 2
            path.cubicTo(QPointF(mx, a.y()), QPointF(mx, b.y()), b)
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            p.setBrush(QColor(color))
            p.drawPolygon([b, QPointF(b.x() - 9, b.y() - 5), QPointF(b.x() - 9, b.y() + 5)])

        cx = [col_w * (i + 0.5) for i in range(5)]
        mid = H / 2
        top, bot = H * 0.2, H * 0.8

        # module background
        geo_rect = QRectF(cx[1] - bw / 2 - 12, 4, bw + 24, H - 8)
        c = QColor(theme.ACCENT)
        c.setAlpha(18)
        p.setPen(QPen(QColor(theme.ACCENT), 1, Qt.DashLine))
        p.setBrush(c)
        p.drawRoundedRect(geo_rect, 14, 14)
        f = QFont()
        f.setPointSize(10)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(theme.ACCENT))
        p.drawText(geo_rect.adjusted(-80, 6, 80, 0), Qt.AlignHCenter | Qt.AlignTop, L('GEOMETRIK MODUL (muallif)', 'ГЕОМЕТРИЧЕСКИЙ МОДУЛЬ (автор)'))

        b0 = box(cx[0], mid, L('Rasm / kamera', 'Фото / камера'), L('istalgan sharoit', 'любые условия'), theme.MUTED)
        b1 = box(cx[1], H * 0.27 + 4, L('478 ta 3D nuqta', '478 3D-точек'), 'MediaPipe FaceMesh', theme.ACCENT)
        b2 = box(cx[1], H * 0.53, 'GeometricTransformer',
                 L('Prokrust · 128-vektor', 'Прокруст · 128-вектор'), theme.ACCENT)
        b3 = box(cx[1], H * 0.79, L('Poza va tekislash', 'Поза и выравнивание'), 'yaw/pitch/roll · 112×112', theme.ACCENT)
        b4 = box(cx[2], H * 0.79, 'MobileFaceNet', L('512 o‘lchamli ko‘rinish vektori', '512-мерный вектор внешности'), theme.ACCENT2)
        b5 = box(cx[3], mid, L('Adaptiv birlashtirish', 'Адаптивное слияние'),
                 L('10 belgi: o‘xshashlik, sifat, poza', '10 признаков: сходство, качество, поза'), theme.FUSION, 96)
        b6 = box(cx[4], H * 0.3, L('Ehtimollik', 'Вероятность'), L('bitta odam? 0…1', 'один человек? 0…1'), theme.GOOD)
        b7 = box(cx[4], H * 0.72, L('Himoyalangan baza', 'Защищённая база'), L('kalit · AES-256 · rasmsiz', 'ключ · AES-256 · без фото'), theme.WARN)

        arrow(QPointF(b0.right(), b0.center().y()), QPointF(b1.left(), b1.center().y()))
        p.setPen(QPen(QColor(theme.ACCENT), 2))
        p.drawLine(QPointF(b1.center().x(), b1.bottom()), QPointF(b2.center().x(), b2.top()))
        p.drawLine(QPointF(b2.center().x(), b2.bottom()), QPointF(b3.center().x(), b3.top()))
        arrow(QPointF(b3.right(), b3.center().y()), QPointF(b4.left(), b4.center().y()), theme.ACCENT)
        arrow(QPointF(b2.right(), b2.center().y()), QPointF(b5.left(), b5.center().y() - 14), theme.ACCENT)
        arrow(QPointF(b4.right(), b4.center().y()), QPointF(b5.left(), b5.center().y() + 14), theme.ACCENT2)
        arrow(QPointF(b5.right(), b5.center().y() - 10), QPointF(b6.left(), b6.center().y()), theme.FUSION)
        arrow(QPointF(b5.right(), b5.center().y() + 10), QPointF(b7.left(), b7.center().y()), theme.FUSION)


class HomePage(QWidget):
    def __init__(self, engine, go):
        super().__init__()
        self.setObjectName('page')
        self.engine = engine
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        body = QWidget()
        body.setObjectName('page')
        scroll.setWidget(body)
        root = QVBoxLayout(body)
        root.setContentsMargins(28, 24, 28, 28)
        root.setSpacing(18)

        title = label('GFRAM', size=40, bold=True, color=theme.ACCENT)
        root.addWidget(title)
        root.addWidget(label(L('Yuzni geometrik usul va sun’iy intellekt asosida tanishning gibrid tizimi',
                               'Гибридная система распознавания лиц на основе геометрического метода и искусственного интеллекта'),
                             size=22, bold=True))
        root.addWidget(label(L('Muallif: Ortiqova F.S. · gfram kutubxonasi (PyPI) · gfram.uz',
                               'Автор: Ортикова Ф.С. · библиотека gfram (PyPI) · gfram.uz'), muted=True))

        tiles = QGridLayout()
        tiles.setSpacing(14)
        items = [
            ('99,67%', L('LFW aniqligi — gibrid tizim (10 fold)', 'Точность на LFW — гибридная система (10 фолдов)'), theme.GOOD),
            ('53,7% → 78,0%', L('Geometrik usul: klassikdan SI gacha', 'Геометрический метод: от классики к ИИ'), theme.ACCENT),
            ('96,28%', L('CPLFW (old/profil) — birlashtirish bilan', 'CPLFW (анфас/профиль) — со слиянием'), theme.ACCENT2),
            ('×2', L('kamroq xato: 0,65% → 0,33%', 'меньше ошибок: 0,65% → 0,33%'), theme.FUSION),
            ('0 / 50 156', L('himoya qarorlarni o‘zgartirmadi', 'решений изменила защита шаблонов'), theme.WARN),
            ('~24 ms', L('bitta yuzga, protsessorda (15 MB)', 'на лицо, на процессоре (15 МБ)'), theme.TEXT),
        ]
        for i, (v, t, col) in enumerate(items):
            tiles.addWidget(StatTile(v, t, col), i // 3, i % 3)
        root.addLayout(tiles)

        c, lay = card(L('Tizim arxitekturasi', 'Архитектура системы'),
                      L('Geometrik modul yuzni topadi, 478 nuqta bo‘yicha tekislaydi, bosh holatini baholaydi va mustaqil shaxs '
                        'vektorini beradi. Neyron tarmoq tekislangan kesim bilan ishlaydi. Adaptiv birlashtirish sifat va pozani '
                        'hisobga olib yakuniy ehtimollikni chiqaradi.',
                        'Геометрический модуль находит лицо, выравнивает его по 478 точкам, оценивает позу и даёт независимый '
                        'вектор личности. Нейросеть работает с выровненным кадром. Адаптивное слияние с учётом качества и позы '
                        'выдаёт итоговую вероятность.'))
        lay.addWidget(PipelineDiagram())
        root.addWidget(c)

        c, lay = card(L('Namoyish rejasi', 'План демонстрации'))
        steps = [
            (L('1. Ro‘yxatga olish', '1. Регистрация'), L('kamera bilan yo‘naltirilgan ro‘yxatga olish, kartochka, boshqaruv', 'регистрация камерой с подсказками, карточка, управление'), 'people'),
            (L('2. Jonli tanish', '2. Живое распознавание'), L('kamerada 478 nuqta, bosh holati va real vaqtda tanish', '478 точек, поза и распознавание в реальном времени'), 'live'),
            (L('3. Baza fayllari', '3. Файлы базы'), L('diskda rasm yo‘q — isbot; kalitni almashtirish', 'на диске нет фото — доказательство; смена ключа'), 'database'),
            (L('4. Identifikatsiya 1:N', '4. Идентификация 1:N'), L('ko‘p odamlar ichidan qidirish', 'поиск среди множества людей'), 'identify'),
            (L('5. Solishtirish 1:1', '5. Сравнение 1:1'), L('qaror qanday qabul qilinadi — 10 belgi', 'как принимается решение — 10 признаков'), 'verify'),
            (L('6. Tanish jurnali', '6. Журнал'), L('kim, qachon tanildi — rasmsiz', 'кто и когда распознан — без фото'), 'journal'),
        ]
        grid = QGridLayout()
        for i, (t, s, key) in enumerate(steps):
            b = button(t, slot=lambda _=False, k=key: go(k))
            b.setMinimumWidth(240)
            grid.addWidget(b, i, 0)
            grid.addWidget(label(s, muted=True), i, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)
        root.addWidget(c)
