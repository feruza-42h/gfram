"""Live benchmark: the library runs the standard LFW / CPLFW protocol in front of the audience."""

import random
import time

import numpy as np
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QProgressBar, QVBoxLayout, QWidget

from .. import data, theme
from ..i18n import L, num, pct
from ..widgets import Chart, FaceView, StatTile, button, card, heading, label, read_image, run_task


def roc(scores, labels):
    order = np.argsort(-scores)
    y = labels[order].astype(float)
    tpr = np.concatenate([[0], np.cumsum(y) / max(1, y.sum())])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / max(1, (1 - y).sum())])
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, 'trapezoid') else float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


class BenchmarkPage(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.setObjectName('page')
        self.engine = engine
        self.task = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(14)
        root.addWidget(heading(L('Jonli sinov: LFW va CPLFW', 'Живой тест: LFW и CPLFW'),
                               L('Kutubxona xalqaro test juftliklarini hozir, auditoriya oldida qayta ishlaydi: har bir juftlik uchun '
                                 'yuz topiladi, ikkala vektor hisoblanadi va birlashtirish qaror qabul qiladi. Yagona qat’iy chegara.',
                                 'Библиотека прямо сейчас, при зрителях, обрабатывает международные тестовые пары: для каждой пары '
                                 'находится лицо, считаются оба вектора и слияние принимает решение. Один фиксированный порог.')))
        row = QHBoxLayout()
        self.dataset = QComboBox()
        self.dataset.addItem(L('LFW — 6000 juftlik (standart)', 'LFW — 6000 пар (стандарт)'), 'lfw')
        self.dataset.addItem(L('CPLFW — old/profil, 6000 juftlik', 'CPLFW — анфас/профиль, 6000 пар'), 'cplfw')
        row.addWidget(self.dataset, 2)
        self.count = QComboBox()
        for n in (200, 600, 2000, 6000):
            self.count.addItem(L(f'{n} juftlik', f'{n} пар'), n)
        self.count.setCurrentIndex(1)
        row.addWidget(self.count, 1)
        self.start_btn = button(L('▶  Sinovni boshlash', '▶  Начать тест'), 'primary', self.toggle)
        row.addWidget(self.start_btn)
        row.addStretch(1)
        root.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        root.addWidget(self.progress)

        tiles = QHBoxLayout()
        self.t_acc = StatTile('—', L('aniqlik (accuracy)', 'точность (accuracy)'), theme.GOOD)
        self.t_tar = StatTile('—', L('to‘g‘ri qabul (TAR)', 'верный допуск (TAR)'), theme.ACCENT)
        self.t_far = StatTile('—', L('noto‘g‘ri qabul (FAR)', 'ложный допуск (FAR)'), theme.BAD)
        self.t_auc = StatTile('—', 'ROC AUC', theme.ACCENT2)
        self.t_speed = StatTile('—', L('juftlik/soniya', 'пар/сек'), theme.FUSION)
        self.t_done = StatTile('—', L('bajarildi', 'выполнено'))
        for t in (self.t_acc, self.t_tar, self.t_far, self.t_auc, self.t_speed, self.t_done):
            tiles.addWidget(t)
        root.addLayout(tiles)

        main = QHBoxLayout()
        main.setSpacing(16)
        c, lay = card(L('Joriy juftlik', 'Текущая пара'))
        pair = QHBoxLayout()
        self.va, self.vb = FaceView('', 220), FaceView('', 220)
        pair.addWidget(self.va)
        pair.addWidget(self.vb)
        lay.addLayout(pair)
        self.pair_info = label('', size=16, bold=True)
        lay.addWidget(self.pair_info)
        c.setFixedWidth(470)
        main.addWidget(c)
        c, lay = card(L('Ehtimolliklar taqsimoti va ROC egri chizig‘i', 'Распределение вероятностей и ROC-кривая'))
        self.chart = Chart(8, 3.8)
        lay.addWidget(self.chart, 1)
        main.addWidget(c, 1)
        root.addLayout(main, 1)
        self.note = label(L('Eslatma: birlashtirish modeli LFW va CPLFW juftliklarida kalibrlangan; mustaqil baho — '
                            'kross-validatsiyadagi 99,67% (dissertatsiyada).',
                            'Примечание: модель слияния калибрована на парах LFW и CPLFW; независимая оценка — '
                            '99,67% в кросс-валидации (в диссертации).'), muted=True)
        root.addWidget(self.note)

    def toggle(self):
        if self.task:
            self.task.cancelled = True
            return
        ds = self.dataset.currentData()
        if ds == 'lfw' and not data.lfw_available() or ds == 'cplfw' and not data.cplfw_available():
            self.pair_info.setText(L('Ma’lumotlar topilmadi.', 'Данные не найдены.'))
            return
        pairs = list(data.lfw_pairs() if ds == 'lfw' else data.cplfw_pairs())
        n = self.count.currentData()
        if n < len(pairs):
            rng = random.Random(7)
            same = [p for p in pairs if p[3]]
            diff = [p for p in pairs if not p[3]]
            pairs = rng.sample(same, n // 2) + rng.sample(diff, n - n // 2)
            rng.shuffle(pairs)
        thr = self.engine.current_threshold()
        engine = self.engine
        self.progress.setRange(0, len(pairs))
        self.start_btn.setText(L('■  To‘xtatish', '■  Остановить'))
        self.scores, self.labels = [], []
        self.missing = 0
        self.t0 = time.perf_counter()

        def work(task):
            cache = {}

            def feat(path):
                if path not in cache:
                    img = read_image(path)
                    cache[path] = (img, engine.analyze_largest(img) if img is not None else None)
                return cache[path]

            for i, (_, pa, pb, same) in enumerate(pairs):
                if task.cancelled:
                    break
                ia, fa = feat(pa)
                ib, fb = feat(pb)
                p = engine.pair_breakdown(fa, fb)['probability'] if fa is not None and fb is not None else None
                task.report((i, ia, fa, ib, fb, same, p))
            return True

        self.task = run_task(work, self._finished, self._failed, lambda v: self._step(v, thr, len(pairs)))

    def _step(self, v, thr, total):
        i, ia, fa, ib, fb, same, p = v
        if p is None:
            self.missing += 1
            score = 0.0                          # no face -> "different" (as the library does)
        else:
            score = p
        self.scores.append(score)
        self.labels.append(same)
        self.progress.setValue(i + 1)
        s = np.array(self.scores)
        y = np.array(self.labels)
        pred = s >= thr
        acc = float((pred == y).mean())
        tar = float(pred[y].mean()) if y.any() else 0.0
        far = float(pred[~y].mean()) if (~y).any() else 0.0
        dt = time.perf_counter() - self.t0
        self.t_acc.set(pct(acc, 2))
        self.t_tar.set(pct(tar, 2))
        self.t_far.set(pct(far, 2))
        self.t_speed.set(f'{(i + 1) / max(dt, 1e-6):.1f}')
        self.t_done.set(f'{i + 1}/{total}')
        if i % 3 == 0 or i == total - 1:
            col = theme.GOOD if (score >= thr) == same else theme.BAD
            for view, img, f in ((self.va, ia, fa), (self.vb, ib, fb)):
                view.set_image(img, [] if f is None else [{'landmarks': f['landmarks'], 'bbox': None, 'color': col}])
            truth = L('bitta odam', 'один человек') if same else L('turli odamlar', 'разные люди')
            dec = L('bitta odam', 'один человек') if score >= thr else L('turli odamlar', 'разные люди')
            self.pair_info.setText(f'{L("Haqiqat", "Истина")}: {truth} · {L("Tizim", "Система")}: {dec} · {pct(score, 1)}')
            self.pair_info.setStyleSheet(f'font-size:15px;font-weight:700;color:{col}')
        if i % 100 == 99:
            self._draw(thr)

    def _draw(self, thr):
        s = np.array(self.scores)
        y = np.array(self.labels, bool)
        if len(s) < 10 or y.all() or (~y).all():
            return
        fpr, tpr, auc = roc(s, y)
        self.t_auc.set(num(auc, 4))
        fig = self.chart.figure
        fig.clear()
        a1 = fig.add_subplot(1, 2, 1)
        bins = np.linspace(0, 1, 41)
        a1.hist(s[y], bins=bins, alpha=0.75, color=theme.GOOD, label=L('bitta odam', 'один человек'))
        a1.hist(s[~y], bins=bins, alpha=0.75, color=theme.BAD, label=L('turli odamlar', 'разные люди'))
        a1.axvline(thr, color=theme.WARN, lw=2, ls='--', label=L('chegara', 'порог'))
        a1.set_yscale('log')
        a1.set_xlabel(L('ehtimollik', 'вероятность'))
        a1.legend(fontsize=9)
        a2 = fig.add_subplot(1, 2, 2)
        a2.plot(fpr, tpr, color=theme.ACCENT, lw=2.2)
        a2.set_xscale('symlog', linthresh=1e-3)
        a2.set_xlim(0, 1)
        a2.set_ylim(0.5, 1.005)
        a2.set_xlabel('FAR')
        a2.set_ylabel('TAR')
        a2.set_title(f'ROC · AUC {auc:.4f}')
        self.chart.draw()

    def _finished(self, _):
        self._draw(self.engine.current_threshold())
        self.task = None
        self.start_btn.setText(L('▶  Sinovni boshlash', '▶  Начать тест'))
        n = len(self.scores)
        self.note.setText(L(f'Tugadi: {n} juftlik, yuz topilmagan juftliklar: {self.missing} («turli odamlar» deb hisoblandi). '
                            f'Vaqt {time.perf_counter() - self.t0:.0f} s.',
                            f'Готово: {n} пар, пар без лица: {self.missing} (засчитаны как «разные»). '
                            f'Время {time.perf_counter() - self.t0:.0f} с.'))

    def _failed(self, e):
        self.task = None
        self.start_btn.setText(L('▶  Sinovni boshlash', '▶  Начать тест'))
        self.note.setText(e[-400:])

    def leave(self):
        if self.task:
            self.task.cancelled = True
