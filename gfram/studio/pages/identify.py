"""1:N identification: search one face over the whole protected database."""

import random

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QHeaderView, QProgressBar, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from .. import data, theme
from ..i18n import L, num, pct
from ..widgets import ImageSlot, StatTile, button, card, heading, label, read_image, run_task


class IdentifyPage(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.setObjectName('page')
        self.engine = engine
        self.expected = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)
        root.addWidget(heading(L('Identifikatsiya (1:N qidiruv)', 'Идентификация (поиск 1:N)'),
                               L('Bitta rasm butun baza bilan solishtiriladi; har bir odam uchun eng yuqori ehtimollik olinadi. '
                                 'Taqqoslash himoyalangan shablonlar ustida bajariladi.',
                                 'Одно фото сравнивается со всей базой; для каждого человека берётся наибольшая вероятность. '
                                 'Сравнение выполняется над защищёнными шаблонами.')))
        row = QHBoxLayout()
        row.addWidget(button(L('LFW: bazadagi odamning BOSHQA rasmi', 'LFW: ДРУГОЕ фото человека из базы'),
                             'primary', self.sample_known))
        row.addWidget(button(L('LFW: bazada yo‘q odam', 'LFW: человек не из базы'), slot=self.sample_unknown))
        row.addStretch(1)
        root.addLayout(row)

        main = QHBoxLayout()
        main.setSpacing(16)
        self.slot = ImageSlot(L('So‘rov rasmi', 'Фото-запрос'), 420)
        self.slot.changed.connect(self._manual)
        main.addWidget(self.slot, 2)

        right = QVBoxLayout()
        tiles = QHBoxLayout()
        self.t_result = StatTile('—', L('natija', 'результат'))
        self.t_time = StatTile('—', L('qidiruv vaqti', 'время поиска'), theme.FUSION)
        self.t_size = StatTile('—', L('bazadagi shablonlar', 'шаблонов в базе'), theme.ACCENT2)
        for t in (self.t_result, self.t_time, self.t_size):
            tiles.addWidget(t)
        right.addLayout(tiles)
        c, lay = card(L('Eng yaqin 10 nomzod', '10 ближайших кандидатов'))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['#', L('Ism', 'Имя'), L('Ehtimollik', 'Вероятность')])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(2, 260)
        lay.addWidget(self.table, 1)
        self.note = label('', muted=True)
        lay.addWidget(self.note)
        right.addWidget(c, 1)
        main.addLayout(right, 3)
        root.addLayout(main, 1)

    def _db_lfw_people(self):
        people = data.lfw_people()
        names = {p['name'] for p in self.engine.persons()}
        return people, names

    def sample_known(self):
        if not data.lfw_available():
            return
        people, names = self._db_lfw_people()
        cands = [n for n, imgs in people.items() if len(imgs) >= 2 and data.pretty_name(n) in names]
        if not cands:
            self.note.setText(L('Avval «Baza» bo‘limida «LFW bilan to‘ldirish» tugmasini bosing.',
                                'Сначала нажмите «Заполнить из LFW» в разделе «База».'))
            return
        n = random.choice(cands)
        path = random.choice(people[n][1:])         # enrolled with photo #1, query with another one
        self._load(path, data.pretty_name(n))

    def sample_unknown(self):
        if not data.lfw_available():
            return
        people, names = self._db_lfw_people()
        cands = [n for n in people if data.pretty_name(n) not in names]
        if not cands:
            return
        n = random.choice(cands)
        self._load(random.choice(people[n]), None)

    def _load(self, path, expected):
        img = read_image(path)
        if img is None:
            return
        self._loading = True
        self.slot.set(img, f'LFW: {path.name}')
        self._loading = False
        self.expected = expected if expected else ''
        self.run(img)

    def _manual(self, img, _src):
        if getattr(self, '_loading', False):
            return
        self.expected = None
        self.run(img)

    def run(self, img):
        def work(task):
            face = self.engine.analyze_largest(img)
            return face, (self.engine.identify(face, top_k=10) if face is not None else None)

        run_task(work, lambda r: self.show(img, *r), lambda e: self.note.setText(e[-300:]))

    def show(self, img, face, res):
        if face is None:
            self.slot.view.set_image(img)
            self.t_result.set(L('Yuz yo‘q', 'Нет лица'), theme.WARN)
            return
        thr = self.engine.current_threshold()
        cands = res['candidates']
        top_name, top_p = cands[0] if cands else (None, 0.0)
        known = top_name is not None and top_p >= thr
        col = theme.GOOD if known else theme.BAD
        self.slot.view.set_image(img, [{'landmarks': face['landmarks'], 'bbox': face['bbox'], 'color': col,
                                        'label': f'{top_name if known else L("Noma’lum", "Неизвестный")}'}])
        self.t_result.set(top_name if known else L('Noma’lum', 'Неизвестный'), col)
        self.t_time.set(f'{res["time_ms"]:.1f} ms')
        self.t_size.set(str(res['faces']))
        self.table.setRowCount(len(cands))
        for i, (n, p) in enumerate(cands):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(n))
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(p * 1000))
            bar.setFormat(pct(p, 2))
            bar.setTextVisible(True)
            bar.setStyleSheet(f'QProgressBar::chunk{{background:{theme.GOOD if p >= thr else "#4a5566"}}}')
            self.table.setCellWidget(i, 2, bar)
        if top_name is not None:
            self.engine.log_event(top_name, top_p, known, 'photo')
        msg = L(f'Chegara {num(thr)}. ', f'Порог {num(thr)}. ')
        if self.expected is not None:
            if self.expected:
                ok = known and top_name == self.expected
                msg += L(f'Kutilgan: {self.expected} — {"✅ to‘g‘ri topildi" if ok else "❌ xato"}',
                         f'Ожидался: {self.expected} — {"✅ найден верно" if ok else "❌ ошибка"}')
            else:
                ok = not known
                msg += L(f'Bu odam bazada yo‘q — {"✅ to‘g‘ri rad etildi" if ok else "❌ noto‘g‘ri qabul qilindi"}',
                         f'Этого человека нет в базе — {"✅ верно отклонён" if ok else "❌ ложный допуск"}')
        self.note.setText(msg)
