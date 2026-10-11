"""Recognition journal: who was recognised when (no images are kept)."""

import csv
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QMessageBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .. import theme
from ..i18n import L, pct
from ..widgets import StatTile, button, heading, label


class JournalPage(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.setObjectName('page')
        self.engine = engine
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)
        root.addWidget(heading(L('Tanish jurnali', 'Журнал распознавания'),
                               L('Jonli tanish va identifikatsiya natijalari vaqt bo‘yicha yoziladi (bir odam — 10 soniyada bir marta). '
                                 'Jurnalda ham rasm yo‘q: faqat vaqt, ism va ehtimollik.',
                                 'Результаты живого распознавания и идентификации записываются по времени (один человек — раз в 10 секунд). '
                                 'В журнале тоже нет фото: только время, имя и вероятность.')))
        tiles = QHBoxLayout()
        self.t_total = StatTile('0', L('jami yozuv', 'всего записей'))
        self.t_today = StatTile('0', L('bugun', 'сегодня'), theme.ACCENT)
        self.t_known = StatTile('0', L('tanilgan', 'распознано'), theme.GOOD)
        self.t_unknown = StatTile('0', L('noma’lum', 'неизвестных'), theme.BAD)
        self.t_people = StatTile('0', L('turli odam', 'разных людей'), theme.ACCENT2)
        for t in (self.t_total, self.t_today, self.t_known, self.t_unknown, self.t_people):
            tiles.addWidget(t)
        root.addLayout(tiles)
        row = QHBoxLayout()
        self.filter = QComboBox()
        self.filter.addItem(L('Hammasi', 'Все'), 'all')
        self.filter.addItem(L('Faqat tanilganlar', 'Только распознанные'), 'known')
        self.filter.addItem(L('Faqat noma’lumlar', 'Только неизвестные'), 'unknown')
        self.filter.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.filter)
        row.addStretch(1)
        row.addWidget(button(L('Yangilash', 'Обновить'), slot=self.refresh))
        row.addWidget(button(L('CSV ga eksport…', 'Экспорт в CSV…'), slot=self.export))
        row.addWidget(button(L('Jurnalni tozalash', 'Очистить журнал'), 'danger', self.clear))
        root.addLayout(row)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([L('Vaqt', 'Время'), L('Natija', 'Результат'), L('Eng yaqin nomzod', 'Ближайший кандидат'),
                                              L('Ehtimollik', 'Вероятность'), L('Manba', 'Источник')])
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        root.addWidget(self.table, 1)
        root.addWidget(label(L(f'Fayl: {engine.events_file}', f'Файл: {engine.events_file}'), muted=True))
        self.refresh()

    def refresh(self):
        ev = list(reversed(self.engine.events))
        today = datetime.now().strftime('%Y-%m-%d')
        self.t_total.set(str(len(ev)))
        self.t_today.set(str(sum(1 for e in ev if e['time'].startswith(today))))
        self.t_known.set(str(sum(1 for e in ev if e['name'])))
        self.t_unknown.set(str(sum(1 for e in ev if not e['name'])))
        self.t_people.set(str(len({e['name'] for e in ev if e['name']})))
        mode = self.filter.currentData()
        if mode == 'known':
            ev = [e for e in ev if e['name']]
        elif mode == 'unknown':
            ev = [e for e in ev if not e['name']]
        ev = ev[:1000]
        src = {'camera': L('kamera', 'камера'), 'photo': L('rasm', 'фото')}
        self.table.setRowCount(len(ev))
        for i, e in enumerate(ev):
            known = bool(e['name'])
            vals = [e['time'], e['name'] if known else L('Noma’lum', 'Неизвестный'), e.get('candidate') or '—',
                    pct(e['probability'], 1), src.get(e.get('source'), e.get('source', ''))]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j == 1:
                    it.setForeground(QColor(theme.GOOD if known else theme.BAD))
                if j == 3:
                    it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, it)

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, 'CSV', str(Path.home() / 'Desktop' / 'gfram_jurnal.csv'), 'CSV (*.csv)')
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['time', 'result', 'candidate', 'probability', 'source'])
            for e in self.engine.events:
                w.writerow([e['time'], e['name'] or 'Unknown', e.get('candidate') or '', e['probability'], e.get('source', '')])

    def clear(self):
        if QMessageBox.question(self, 'GFRAM', L('Jurnal tozalansinmi?', 'Очистить журнал?')) == QMessageBox.Yes:
            self.engine.clear_events()
            self.refresh()
