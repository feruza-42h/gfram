"""1:1 verification with the adaptive fusion explained feature by feature."""

import random
import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QProgressBar, QScrollArea, QVBoxLayout, QWidget

from .. import data, theme
from ..i18n import L, num, pct
from ..widgets import Chart, ImageSlot, button, card, heading, label, read_image, run_task


def feature_names():
    return {
        's_app': L('Tashqi ko‘rinish o‘xshashligi', 'Сходство внешности'),
        's_geo': L('Geometrik o‘xshashlik', 'Геометрическое сходство'),
        's_app*s_geo': L('Ikkala o‘xshashlik ko‘paytmasi', 'Произведение сходств'),
        'norm_min': L('Sifat: yomonroq rasm', 'Качество: худшее фото'),
        'norm_max': L('Sifat: yaxshiroq rasm', 'Качество: лучшее фото'),
        'sharp_min': L('Tiniqlik (minimal)', 'Резкость (минимум)'),
        'yaw_diff': L('Burilish farqi', 'Разница поворота'),
        'yaw_max': L('Eng katta burilish', 'Наибольший поворот'),
        's_app*yaw_diff': L('Ko‘rinish × burilish farqi', 'Внешность × разница поворота'),
        's_geo*yaw_diff': L('Geometriya × burilish farqi', 'Геометрия × разница поворота'),
    }


class VerifyPage(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.setObjectName('page')
        self.engine = engine
        self.truth = None

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

        root.addWidget(heading(L('Solishtirish (1:1 verifikatsiya)', 'Сравнение (верификация 1:1)'),
                               L('Ikki rasm bitta odammi? Adaptiv birlashtirish 10 ta belgini — ikkala o‘xshashlik, rasm sifati va '
                                 'bosh holatini — hisobga olib, kalibrlangan ehtimollik beradi.',
                                 'Один ли человек на двух фото? Адаптивное слияние учитывает 10 признаков — оба сходства, качество '
                                 'фото и позу головы — и выдаёт калиброванную вероятность.')))

        row = QHBoxLayout()
        row.addWidget(button(L('LFW: bitta odam', 'LFW: один человек'), slot=lambda: self.sample('lfw', True)))
        row.addWidget(button(L('LFW: turli odamlar', 'LFW: разные люди'), slot=lambda: self.sample('lfw', False)))
        row.addWidget(button(L('CPLFW: old va profil (bitta odam)', 'CPLFW: анфас и профиль (один человек)'),
                             slot=lambda: self.sample('cplfw', True)))
        row.addWidget(button(L('CPLFW: turli odamlar', 'CPLFW: разные люди'), slot=lambda: self.sample('cplfw', False)))
        row.addStretch(1)
        root.addLayout(row)

        pair = QHBoxLayout()
        pair.setSpacing(16)
        self.a = ImageSlot(L('1-rasm', 'Фото 1'), 330)
        self.b = ImageSlot(L('2-rasm', 'Фото 2'), 330)
        for s in (self.a, self.b):
            s.changed.connect(self._manual)
            pair.addWidget(s, 1)

        res, lay = card(L('Natija', 'Результат'))
        res.setFixedWidth(360)
        self.verdict = label('—', size=26, bold=True)
        lay.addWidget(self.verdict)
        self.prob = label('', size=44, bold=True)
        lay.addWidget(self.prob)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        lay.addWidget(self.bar)
        self.thr = label('', muted=True)
        lay.addWidget(self.thr)
        grid = QGridLayout()
        self.vals = {}
        for i, (k, t) in enumerate((('s_app', L('Ko‘rinish o‘xshashligi', 'Сходство внешности')),
                                    ('s_geo', L('Geometrik o‘xshashlik', 'Геометрическое сходство')),
                                    ('yaw', L('Burilish farqi', 'Разница поворота')),
                                    ('time', L('Vaqt (2 rasm)', 'Время (2 фото)')),
                                    ('truth', L('Haqiqatda', 'На самом деле')))):
            grid.addWidget(label(t, muted=True, wrap=False), i, 0)
            v = label('—', bold=True, wrap=False)
            grid.addWidget(v, i, 1, alignment=Qt.AlignRight)
            self.vals[k] = v
        lay.addLayout(grid)
        lay.addStretch(1)
        pair.addWidget(res)
        root.addLayout(pair)

        c, lay = card(L('Qaror qanday qabul qilindi', 'Как принято решение'),
                      L('Har bir belgining logistik modeldagi hissasi: o‘ngga (yashil) — «bitta odam», chapga (qizil) — «turli odamlar». '
                        'Hissalar yig‘indisi sigmoid orqali ehtimollikka aylanadi.',
                        'Вклад каждого признака в логистическую модель: вправо (зелёный) — «один человек», влево (красный) — «разные». '
                        'Сумма вкладов через сигмоиду превращается в вероятность.'))
        self.chart = Chart(10, 4.6)
        lay.addWidget(self.chart)
        root.addWidget(c)
        self.status = label('', muted=True)
        root.addWidget(self.status)

    # ------------------------------------------------------------------

    def sample(self, dataset, same):
        if dataset == 'lfw' and data.lfw_available():
            pairs = data.lfw_pairs()
        elif dataset == 'cplfw' and data.cplfw_available():
            pairs = data.cplfw_pairs()
        else:
            self.status.setText(L('Ma’lumotlar to‘plami topilmadi.', 'Набор данных не найден.'))
            return
        pool = [p for p in pairs if p[3] == same]
        engine = self.engine
        self.status.setText(L('Juftlik tanlanmoqda…', 'Выбор пары…'))

        def work(task):
            # MediaPipe misses ~17% of extreme CPLFW profiles: show a pair where both faces are found
            for _ in range(25):
                _, pa, pb, s = random.choice(pool)
                ia, ib = read_image(pa), read_image(pb)
                if ia is not None and ib is not None and engine.analyze_largest(ia) and engine.analyze_largest(ib):
                    return pa, pb, ia, ib, s
            return None

        def done(r):
            if r is None:
                return
            pa, pb, ia, ib, s = r
            self._loading = True
            self.a.set(ia, f'{dataset.upper()}: {pa.name}')
            self.b.set(ib, f'{dataset.upper()}: {pb.name}')
            self._loading = False
            self.truth = s
            self.run()

        run_task(work, done, lambda e: self.status.setText(e[-300:]))

    def _manual(self, *_):
        if getattr(self, '_loading', False):
            return
        self.truth = None
        self.run()

    def run(self):
        if self.a.image is None or self.b.image is None:
            return
        ia, ib = self.a.image, self.b.image
        self.status.setText(L('Hisoblanmoqda…', 'Вычисление…'))

        def work(task):
            t = time.perf_counter()
            fa, fb = self.engine.analyze_largest(ia), self.engine.analyze_largest(ib)
            dt = (time.perf_counter() - t) * 1000
            br = self.engine.pair_breakdown(fa, fb) if fa is not None and fb is not None else None
            return fa, fb, br, dt

        run_task(work, self.show, lambda e: self.status.setText(e[-300:]))

    def show(self, result):
        fa, fb, br, dt = result
        self.status.setText('')
        for slot, f in ((self.a, fa), (self.b, fb)):
            faces = [] if f is None else [{'landmarks': f['landmarks'], 'bbox': f['bbox'], 'pose': f['pose'],
                                           'color': theme.ACCENT}]
            slot.view.set_image(slot.image, faces)
        if br is None:
            missing = [str(i + 1) for i, f in enumerate((fa, fb)) if f is None]
            self.verdict.setText(L(f'{", ".join(missing)}-rasmda yuz topilmadi', f'Лицо не найдено на фото {", ".join(missing)}'))
            self.verdict.setStyleSheet(f'font-size:22px;font-weight:700;color:{theme.WARN}')
            self.prob.setText('')
            self.bar.setValue(0)
            return
        thr = self.engine.current_threshold()
        p = br['probability']
        same = p >= thr
        col = theme.GOOD if same else theme.BAD
        self.verdict.setText(L('✅ Bitta odam', '✅ Один человек') if same else L('❌ Turli odamlar', '❌ Разные люди'))
        self.verdict.setStyleSheet(f'font-size:26px;font-weight:700;color:{col}')
        self.prob.setText(pct(p, 1))
        self.prob.setStyleSheet(f'font-size:44px;font-weight:700;color:{col}')
        self.bar.setValue(int(p * 1000))
        self.bar.setStyleSheet(f'QProgressBar::chunk{{background:{col}}}')
        self.thr.setText(L(f'Bitta odam ekanligi ehtimoli · chegara {num(thr)}',
                           f'Вероятность, что это один человек · порог {num(thr)}'))
        self.vals['s_app'].setText(num(br['s_app']))
        self.vals['s_geo'].setText(num(br['s_geo']))
        self.vals['yaw'].setText(f'{br["yaw_diff"]:.1f}°')
        self.vals['time'].setText(f'{dt:.0f} ms')
        if self.truth is None:
            self.vals['truth'].setText('—')
            self.vals['truth'].setStyleSheet('')
        else:
            ok = self.truth == same
            self.vals['truth'].setText((L('bitta odam', 'один человек') if self.truth else L('turli odamlar', 'разные люди'))
                                       + (' ✓' if ok else ' ✗'))
            self.vals['truth'].setStyleSheet(f'font-weight:700;color:{theme.GOOD if ok else theme.BAD}')
        self._draw(br)

    def _draw(self, br):
        names = feature_names()
        ax = self.chart.axes()
        contrib = np.asarray(br['contrib'])
        order = np.argsort(np.abs(contrib))
        labels = [names.get(br['features'][i], br['features'][i]) for i in order]
        vals = contrib[order]
        ax.barh(labels, vals, color=[theme.GOOD if v > 0 else theme.BAD for v in vals])
        ax.axvline(0, color=theme.MUTED, lw=1)
        ax.set_xlabel(L(f'Logit hissasi (yig‘indi {br["logit"]:+.2f}, ozod had {br["intercept"]:+.2f})',
                        f'Вклад в логит (сумма {br["logit"]:+.2f}, свободный член {br["intercept"]:+.2f})'))
        ax.grid(axis='y', visible=False)
        self.chart.draw()
