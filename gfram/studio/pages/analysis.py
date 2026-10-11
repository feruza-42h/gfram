"""Step-by-step geometric analysis of one photo: what each stage of the method computes."""

import random
import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QHBoxLayout, QScrollArea, QTabWidget, QVBoxLayout,
                               QWidget)

from .. import data, theme
from ..i18n import L, num
from ..widgets import Chart, FaceView, ImageSlot, button, card, heading, label, run_task

def groups():
    return [
    ('euclidean', 30, L('Masofalar', 'Расстояния')),
    ('differential', 40, L('Egrilik', 'Кривизна')),
    ('topological', 20, L('Topologik', 'Топологические')),
    ('statistical', 30, L('Statistik', 'Статистические')),
    ('symmetry', 15, L('Simmetriya', 'Симметрия')),
    ('graph', 15, L('Graf (Delone)', 'Граф (Делоне)')),
    ('iris', 3, L('Ko‘z qorachig‘i', 'Радужка')),
    ]


class AnalysisPage(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.setObjectName('page')
        self.engine = engine
        self.face = None
        self._extractor = None

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        body = QWidget()
        body.setObjectName('page')
        scroll.setWidget(body)
        root = QVBoxLayout(body)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        root.addWidget(heading(L('Geometrik tahlil', 'Геометрический анализ'),
                               L('Bitta rasm usulning barcha bosqichlaridan o‘tadi: 478 ta 3D nuqta → Prokrust tekislash → '
                                 'GeometricTransformer → bosh holati → neyron tarmoq uchun tekislangan kesim.',
                                 'Одно фото проходит все этапы метода: 478 трёхмерных точек → выравнивание Прокруста → '
                                 'GeometricTransformer → поза головы → выровненный кадр для нейросети.')))

        top = QHBoxLayout()
        top.setSpacing(16)
        self.slot = ImageSlot(L('1. Rasm va 478 nuqta', '1. Фото и 478 точек'), 460)
        self.slot.add_button(button(L('LFW dan tasodifiy', 'Случайное из LFW'), slot=self.random_lfw))
        self.slot.changed.connect(self.analyze)
        self.slot.view.show_mesh = True
        top.addWidget(self.slot, 3)

        side = QVBoxLayout()
        side.setSpacing(16)
        c, lay = card(L('2. Tekislangan kesim (112×112)', '2. Выровненный кадр (112×112)'),
                      L('Geometrik modul ko‘z, burun va lab nuqtalari bo‘yicha yuzni tekislaydi — alohida detektor kerak emas.',
                        'Геометрический модуль выравнивает лицо по точкам глаз, носа и губ — отдельный детектор не нужен.'))
        self.crop_view = FaceView(L('—', '—'), 220)
        lay.addWidget(self.crop_view)
        side.addWidget(c)
        c, lay = card(L('Ko‘rsatkichlar', 'Показатели'))
        grid = QGridLayout()
        self.vals = {}
        rows = [('n', L('Nuqtalar soni', 'Число точек')), ('yaw', 'Yaw'), ('pitch', 'Pitch'), ('roll', 'Roll'),
                ('sharp', L('Tiniqlik', 'Резкость')), ('norm', L('Sifat (norma)', 'Качество (норма)')),
                ('t', L('Vaqt', 'Время'))]
        for i, (k, t) in enumerate(rows):
            grid.addWidget(label(t, muted=True, wrap=False), i, 0)
            v = label('—', bold=True, wrap=False)
            grid.addWidget(v, i, 1, alignment=Qt.AlignRight)
            self.vals[k] = v
        lay.addLayout(grid)
        side.addWidget(c)
        sw = QWidget()
        sw.setLayout(side)
        sw.setFixedWidth(380)
        top.addWidget(sw)
        root.addLayout(top)

        toggles = QHBoxLayout()
        for text, attr, default in ((L('Nuqtalar', 'Точки'), 'show_points', True),
                                    (L('Uchburchak to‘r', 'Сетка'), 'show_mesh', True),
                                    (L('Konturlar', 'Контуры'), 'show_contours', True),
                                    (L('Bosh holati', 'Поза'), 'show_pose', True)):
            cb = QCheckBox(text)
            cb.setChecked(default)
            cb.toggled.connect(lambda v, a=attr: (setattr(self.slot.view, a, v), self.slot.view.update()))
            toggles.addWidget(cb)
        toggles.addStretch(1)
        root.addLayout(toggles)

        tabs = QTabWidget()
        tabs.setMinimumHeight(520)
        # 3D
        w = QWidget()
        lay = QHBoxLayout(w)
        self.chart3d = Chart(6, 5, projection='3d')
        lay.addWidget(self.chart3d, 3)
        lay.addWidget(label(L(
            '<b>Prokrust tekislash.</b><br><br>478 ta nuqta markazlashtiriladi, birlik o‘lchamga keltiriladi va '
            'o‘rtacha yuz shakliga 3D da buriladi (Kabsch algoritmi). Shundan so‘ng bosh holati, masofa va masshtab '
            'yo‘qoladi — faqat yuz <b>shakli</b> qoladi.<br><br>Ko‘k — shu yuz, kulrang — o‘rtacha shakl. '
            'Sichqoncha bilan aylantirish mumkin.<br><br>Faqat shu tekislash (o‘qitishsiz) aniqlikni 53,7% dan 67,5% ga oshiradi.',
            '<b>Выравнивание Прокруста.</b><br><br>478 точек центрируются, приводятся к единичному размеру и '
            'поворачиваются в 3D к средней форме лица (алгоритм Кабша). После этого поза, расстояние и масштаб '
            'исчезают — остаётся только <b>форма</b> лица.<br><br>Синее — это лицо, серое — средняя форма. '
            'Можно вращать мышью.<br><br>Одно это выравнивание (без обучения) поднимает точность с 53,7% до 67,5%.')), 2)
        tabs.addTab(w, L('3D shakl (Prokrust)', '3D-форма (Прокруст)'))
        # classic features
        w = QWidget()
        lay = QVBoxLayout(w)
        self.feat_chart = Chart(10, 4)
        lay.addWidget(label(L('1-bosqich: 153 ta klassik geometrik belgi (gfram 3.0). Ular bosh holatini ko‘proq tavsiflaydi, '
                              'shaxsni emas — LFW da 53,7%. Shu sabab ishda yangi yondashuv taklif etilgan.',
                              'Этап 1: 153 классических геометрических признака (gfram 3.0). Они больше описывают позу, '
                              'чем личность — 53,7% на LFW. Поэтому в работе предложен новый подход.'), muted=True))
        lay.addWidget(self.feat_chart, 1)
        tabs.addTab(w, L('Klassik belgilar (153)', 'Классические признаки (153)'))
        # embeddings
        w = QWidget()
        lay = QVBoxLayout(w)
        self.emb_chart = Chart(10, 4.4)
        lay.addWidget(label(L('GeometricTransformer 478 nuqtadan 128 o‘lchamli shaxs vektorini (423 ming parametr), '
                              'MobileFaceNet esa tekislangan kesimdan 512 o‘lchamli tashqi ko‘rinish vektorini hosil qiladi. '
                              'Bazada ular kalit bilan himoyalangan holda saqlanadi.',
                              'GeometricTransformer строит из 478 точек 128-мерный вектор личности (423 тыс. параметров), '
                              'MobileFaceNet — 512-мерный вектор внешности по выровненному кадру. '
                              'В базе они хранятся защищёнными ключом.'), muted=True))
        lay.addWidget(self.emb_chart, 1)
        tabs.addTab(w, L('Vektorlar (128 + 512)', 'Векторы (128 + 512)'))
        root.addWidget(tabs)
        self.status = label('', muted=True)
        root.addWidget(self.status)
        self._draw_empty()

    def _draw_empty(self):
        for ch in (self.feat_chart, self.emb_chart):
            ax = ch.axes()
            ax.text(0.5, 0.5, L('Rasm tanlang', 'Выберите фото'), ha='center', va='center', color=theme.MUTED,
                    transform=ax.transAxes)
            ax.set_axis_off()
            ch.draw()

    def random_lfw(self):
        if not data.lfw_available():
            return
        people = data.lfw_people()
        name = random.choice(list(people))
        path = random.choice(people[name])
        self.slot.load(path)
        self.slot.source.setText(f'LFW: {data.pretty_name(name)}')

    def analyze(self, img, _source):
        self.status.setText(L('Tahlil…', 'Анализ…'))

        def work(task):
            t = time.perf_counter()
            face = self.engine.analyze_largest(img)
            dt = (time.perf_counter() - t) * 1000
            feats = None
            if face is not None:
                from gfram.detectors import LandmarkNormalizer
                from gfram.geometry.features import GeometricFeatureExtractor
                if self._extractor is None:
                    self._extractor = (LandmarkNormalizer(), GeometricFeatureExtractor())
                n, ex = self._extractor
                feats = ex.extract(n.normalize(face['landmarks']))
            return face, dt, feats

        run_task(work, lambda r: self.show(img, *r), lambda e: self.status.setText(e[-300:]))

    def show(self, img, face, dt, feats):
        self.face = face
        if face is None:
            self.slot.view.set_image(img)
            self.status.setText(L('Yuz topilmadi.', 'Лицо не найдено.'))
            return
        self.slot.view.set_image(img, [{'landmarks': face['landmarks'], 'bbox': face['bbox'], 'pose': face['pose'],
                                        'color': theme.ACCENT}])
        self.crop_view.set_image(face['crop'])
        yaw, pitch, roll = face['pose']
        self.vals['n'].setText('478 (3D)')
        self.vals['yaw'].setText(f'{yaw:+.1f}°')
        self.vals['pitch'].setText(f'{pitch:+.1f}°')
        self.vals['roll'].setText(f'{roll:+.1f}°')
        self.vals['sharp'].setText(num(face['sharp'], 2))
        self.vals['norm'].setText(num(face['app_norm'], 1))
        self.vals['t'].setText(f'{dt:.0f} ms')
        self.status.setText('')
        self._draw_3d(face)
        self._draw_features(feats)
        self._draw_embeddings(face)

    def _draw_3d(self, face):
        from gfram.geometry.alignment import procrustes_align
        ref = self.engine.embedder.reference
        x = procrustes_align(face['landmarks'], ref)
        ax = self.chart3d.axes()
        ax.scatter(ref[:, 0], ref[:, 2], -ref[:, 1], s=2, c='#56606e', alpha=0.35, depthshade=False)
        ax.scatter(x[:, 0], x[:, 2], -x[:, 1], s=4, c=x[:, 2], cmap='cool', depthshade=False)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_box_aspect((1, 0.6, 1.2))
        ax.view_init(elev=8, azim=-70)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((0, 0, 0, 0))
            a.set_ticklabels([])
        ax.set_title(L('Prokrust bo‘yicha tekislangan 478 nuqta', '478 точек после выравнивания Прокруста'))
        self.chart3d.draw()

    def _draw_features(self, feats):
        ax = self.feat_chart.axes()
        if feats is None:
            self.feat_chart.draw()
            return
        f = np.asarray(feats, float)
        z = (f - f.mean()) / (f.std() + 1e-9)
        start = 0
        for i, (key, n, name) in enumerate(groups()):
            idx = np.arange(start, start + n)
            ax.bar(idx, z[start:start + n], color=theme.SERIES[i % len(theme.SERIES)], width=0.85, label=f'{name} ({n})')
            start += n
        ax.set_xlim(-1, len(f))
        ax.set_xlabel(L('Belgi raqami', 'Номер признака'))
        ax.set_ylabel(L('Qiymat (normallangan)', 'Значение (нормир.)'))
        ax.legend(ncol=4, fontsize=9, loc='upper right')
        ax.set_title(L(f'{len(f)} ta klassik geometrik belgi', f'{len(f)} классических геометрических признаков'))
        self.feat_chart.draw()

    def _draw_embeddings(self, face):
        fig = self.emb_chart.figure
        fig.clear()
        a1 = fig.add_subplot(1, 2, 1)
        a2 = fig.add_subplot(1, 2, 2)
        a1.imshow(face['geo'].reshape(8, 16), cmap='coolwarm', aspect='auto')
        a1.set_title(L('Geometrik vektor (128)', 'Геометрический вектор (128)'))
        a2.imshow(face['app'].reshape(16, 32), cmap='PuOr', aspect='auto')
        a2.set_title(L('Tashqi ko‘rinish vektori (512)', 'Вектор внешности (512)'))
        for a in (a1, a2):
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
        self.emb_chart.draw()
