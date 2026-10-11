"""Registration of people (guided camera capture or photos) and management of the enrolled people."""

from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFormLayout, QGridLayout, QHBoxLayout, QHeaderView,
                               QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .. import theme
from ..i18n import L, num, pct
from ..widgets import (FaceView, StatTile, bgr_to_qimage, button, card, heading, label, pick_images, read_image,
                       run_task)
from .live import RecognitionThread, camera_names

MIN_FACE_PX = 100
MIN_SHARP = 2.5
HOLD_FRAMES = 3


def steps():
    return [
        ('front', L('To‘g‘riga, kameraga qarang', 'Смотрите прямо в камеру')),
        ('side1', L('Boshni sekin bir tomonga buring', 'Медленно поверните голову в одну сторону')),
        ('side2', L('Endi boshqa tomonga buring', 'Теперь в другую сторону')),
        ('tilt', L('Boshni biroz yuqoriga yoki pastga', 'Слегка поднимите или опустите голову')),
        ('front2', L('Yana to‘g‘riga qarang', 'Снова смотрите прямо')),
    ]


def card_fields():
    return [('position', L('Lavozimi', 'Должность')), ('department', L('Bo‘lim', 'Отдел')),
            ('code', L('Tabel / ID raqami', 'Табельный / ID номер')), ('note', L('Izoh', 'Примечание'))]


class Thumb(QLabel):
    def __init__(self, crop):
        super().__init__()
        self.setPixmap(QPixmap.fromImage(bgr_to_qimage(crop)).scaled(84, 84, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setFixedSize(88, 88)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f'border:1px solid {theme.BORDER}; border-radius:6px;')


class PeoplePage(QWidget):
    def __init__(self, engine, on_db_changed):
        super().__init__()
        self.setObjectName('page')
        self.engine = engine
        self.on_db_changed = on_db_changed
        self.thread: Optional[RecognitionThread] = None
        self.captured: List[Dict] = []
        self.step = 0
        self.hold = 0
        self.ref_pose = None
        self.side_sign = 0
        self.duplicate = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(12)
        root.addWidget(heading(L('Ro‘yxatga olish va boshqaruv', 'Регистрация и управление'),
                               L('Odamni kamera orqali (yo‘naltirilgan, 5 holat) yoki rasmlardan ro‘yxatga olish; kartochka, shablonlar, '
                                 'tahrirlash va o‘chirish. Rasmlar faqat xotirada ishlanadi va diskka yozilmaydi.',
                                 'Регистрация человека через камеру (с подсказками, 5 положений) или по фото; карточка, шаблоны, '
                                 'редактирование и удаление. Фото обрабатываются только в памяти и не записываются на диск.')))
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_register(), L('➕  Yangi odamni ro‘yxatga olish', '➕  Регистрация нового человека'))
        self.tabs.addTab(self._build_manage(), L('👥  Odamlar ro‘yxati', '👥  Список людей'))
        self.tabs.currentChanged.connect(lambda i: (self.stop_camera(), self.refresh()) if i == 1 else None)
        root.addWidget(self.tabs, 1)
        self.refresh()

    # ================================================================== registration

    def _build_register(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 12, 0, 0)
        lay.setSpacing(16)

        left = QVBoxLayout()
        row = QHBoxLayout()
        self.cam_box = QComboBox()
        names = camera_names()
        for i in range(max(1, len(names))):
            self.cam_box.addItem(f'📷 {names[i] if i < len(names) else L("Kamera", "Камера")} ({i})', i)
        for i, n in enumerate(names):
            if 'iphone' not in n.lower():
                self.cam_box.setCurrentIndex(i)
                break
        row.addWidget(self.cam_box, 1)
        self.cam_btn = button(L('▶  Kamerani yoqish', '▶  Включить камеру'), 'primary', self.toggle_camera)
        row.addWidget(self.cam_btn)
        row.addWidget(button(L('Rasmlardan…', 'Из фото…'), slot=self.add_from_photos))
        left.addLayout(row)
        self.view = FaceView(L('Kamerani yoqing yoki rasmlardan qo‘shing', 'Включите камеру или добавьте фото'), 380)
        self.view.show_mesh = False
        left.addWidget(self.view, 1)
        self.hint = label('', size=20, bold=True, color=theme.ACCENT)
        self.hint.setAlignment(Qt.AlignCenter)
        left.addWidget(self.hint)
        self.quality = label('', muted=True)
        self.quality.setAlignment(Qt.AlignCenter)
        left.addWidget(self.quality)
        lay.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(12)
        c, cl = card(L('Yo‘naltirilgan suratga olish', 'Съёмка с подсказками'),
                     L('Har bir holat avtomatik olinadi: yuz yetarlicha katta, tiniq va kadrda bitta bo‘lishi kerak. '
                       'Turli burchaklar keyingi tanishni ishonchliroq qiladi.',
                       'Каждое положение снимается автоматически: лицо достаточно крупное, резкое и одно в кадре. '
                       'Разные ракурсы делают дальнейшее распознавание надёжнее.'))
        self.step_labels = []
        for _, text in steps():
            lb = label('○  ' + text, muted=True)
            cl.addWidget(lb)
            self.step_labels.append(lb)
        self.dup_label = label('', bold=True)
        cl.addWidget(self.dup_label)
        thumbs_row = QHBoxLayout()
        self.thumbs = QHBoxLayout()
        thumbs_row.addLayout(self.thumbs)
        thumbs_row.addStretch(1)
        cl.addLayout(thumbs_row)
        cl.addWidget(label(L('Yuqoridagi kesimlar faqat xotirada — diskka hech qachon yozilmaydi.',
                             'Кадры выше — только в памяти, на диск никогда не записываются.'), muted=True))
        r = QHBoxLayout()
        r.addWidget(button(L('Qaytadan', 'Заново'), slot=self.reset_capture))
        cl.addLayout(r)
        right.addWidget(c)

        c, cl = card(L('Kartochka', 'Карточка'))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(L('Familiya Ism Sharif', 'Фамилия Имя Отчество'))
        form.addRow(L('F.I.Sh. *', 'Ф.И.О. *'), self.name_edit)
        self.fields = {}
        for k, t in card_fields():
            e = QLineEdit()
            form.addRow(t, e)
            self.fields[k] = e
        cl.addLayout(form)
        self.save_btn = button(L('💾  Bazaga saqlash', '💾  Сохранить в базу'), 'primary', self.save_person)
        cl.addWidget(self.save_btn)
        self.save_status = label('', muted=True)
        cl.addWidget(self.save_status)
        right.addWidget(c)
        right.addStretch(1)
        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(430)
        lay.addWidget(rw)
        self._update_steps()
        return w

    def toggle_camera(self):
        if self.thread:
            self.stop_camera()
            return
        self.thread = RecognitionThread(self.engine, self.cam_box.currentData())
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.failed.connect(lambda _: (self.stop_camera(), QMessageBox.warning(self, 'GFRAM', L(
            'Kamerani ochib bo‘lmadi. Tizim sozlamalari → Maxfiylik → Kamera → Terminal ga ruxsat bering.',
            'Не удалось открыть камеру. Системные настройки → Конфиденциальность → Камера → разрешите Terminal.'))))
        self.thread.start()
        self.cam_btn.setText(L('■  O‘chirish', '■  Выключить'))

    def stop_camera(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait(2000)
            self.thread = None
        self.cam_btn.setText(L('▶  Kamerani yoqish', '▶  Включить камеру'))

    def leave(self):
        self.stop_camera()

    def reset_capture(self):
        self.captured, self.step, self.hold, self.ref_pose, self.side_sign, self.duplicate = [], 0, 0, None, 0, None
        while self.thumbs.count():
            it = self.thumbs.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.dup_label.setText('')
        self.save_status.setText('')
        self._update_steps()

    def _update_steps(self):
        for i, lb in enumerate(self.step_labels):
            text = steps()[i][1]
            if i < self.step:
                lb.setText('✅  ' + text)
                lb.setStyleSheet(f'color:{theme.GOOD}')
            elif i == self.step:
                lb.setText('➜  ' + text)
                lb.setStyleSheet(f'color:{theme.TEXT};font-weight:700')
            else:
                lb.setText('○  ' + text)
                lb.setStyleSheet(f'color:{theme.MUTED}')
        if self.step >= len(steps()):
            self.hint.setText(L('✅ Tayyor — kartochkani to‘ldirib «Saqlash» ni bosing',
                                '✅ Готово — заполните карточку и нажмите «Сохранить»'))
            self.hint.setStyleSheet(f'font-size:18px;font-weight:700;color:{theme.GOOD}')
        elif self.thread:
            self.hint.setText(steps()[self.step][1])
            self.hint.setStyleSheet(f'font-size:20px;font-weight:700;color:{theme.ACCENT}')

    def _condition(self, face) -> bool:
        yaw, pitch, _ = face['pose']
        key = steps()[self.step][0]
        if key in ('front', 'front2'):
            return abs(yaw) < 10
        dy = yaw - self.ref_pose[0]
        if key == 'side1':
            return abs(dy) > 15
        if key == 'side2':
            return abs(dy) > 15 and np.sign(dy) == -self.side_sign
        if key == 'tilt':
            return abs(pitch - self.ref_pose[1]) > 8
        return False

    def _on_frame(self, frame, faces, info):
        if not faces:
            self.view.set_image(frame)
            self.quality.setText(L('Yuz topilmadi', 'Лицо не найдено'))
            return
        f = faces[0]
        w = f['bbox'][2]
        problems = []
        if len(faces) > 1:
            problems.append(L('kadrda bir nechta yuz', 'в кадре несколько лиц'))
        if w < MIN_FACE_PX:
            problems.append(L('yaqinroq keling', 'подойдите ближе'))
        if f['sharp'] < MIN_SHARP:
            problems.append(L('tasvir xira', 'изображение размыто'))
        ok = not problems
        yaw, pitch, _ = f['pose']
        self.quality.setText(f'yaw {yaw:+.0f}° · pitch {pitch:+.0f}° · ' + L('o‘lcham', 'размер') + f' {w}px · '
                             + L('tiniqlik', 'резкость') + f' {num(f["sharp"], 1)}'
                             + ('' if ok else '   ⚠ ' + ', '.join(problems)))
        color = theme.GOOD if ok else theme.WARN
        self.view.set_image(frame, [{'landmarks': f['landmarks'], 'bbox': f['bbox'], 'pose': f['pose'], 'color': color}])
        if self.step >= len(steps()) or not ok:
            self.hold = 0
            return
        if self._condition(f):
            self.hold += 1
            if self.hold >= HOLD_FRAMES:
                self._capture(f)
        else:
            self.hold = 0

    def _capture(self, face, source='camera'):
        self.hold = 0
        if self.ref_pose is None:
            self.ref_pose = face['pose'].copy()
        if self.step < len(steps()) and steps()[self.step][0] == 'side1':
            self.side_sign = np.sign(face['pose'][0] - self.ref_pose[0])
        self.captured.append(dict(face, source=source))
        self.thumbs.addWidget(Thumb(face['crop']))
        self.step = min(self.step + 1, len(steps())) if source == 'camera' else self.step
        if len(self.captured) == 1:
            self._check_duplicate(face)
        self._update_steps()

    def _check_duplicate(self, face):
        res = self.engine.identify(face, top_k=1)['candidates']
        thr = self.engine.current_threshold()
        if res and res[0][1] >= thr:
            self.duplicate = res[0][0]
            self.dup_label.setText(L(f'⚠ Bu odam bazada allaqachon bor: «{res[0][0]}» ({pct(res[0][1], 1)}). '
                                     f'Saqlansa, shablonlar shu odamga qo‘shiladi.',
                                     f'⚠ Этот человек уже есть в базе: «{res[0][0]}» ({pct(res[0][1], 1)}). '
                                     f'При сохранении шаблоны добавятся к нему.'))
            self.dup_label.setStyleSheet(f'color:{theme.WARN};font-weight:700')
            self.name_edit.setText(res[0][0])
            m = self.engine.person(res[0][0])
            for k, e in self.fields.items():
                e.setText(m.get(k, ''))
        else:
            self.duplicate = None
            self.dup_label.setText(L('✓ Bazada o‘xshash odam yo‘q — yangi shaxs.', '✓ Похожих в базе нет — новый человек.'))
            self.dup_label.setStyleSheet(f'color:{theme.GOOD};font-weight:700')

    def add_from_photos(self):
        files = pick_images(self, multiple=True)
        if not files:
            return
        engine = self.engine

        def work(task):
            out = []
            for p in files:
                img = read_image(p)
                f = engine.analyze_largest(img) if img is not None else None
                out.append((p, img, f))
            return out

        def done(results):
            missing = 0
            for p, img, f in results:
                if f is None:
                    missing += 1
                    continue
                self.view.set_image(img, [{'landmarks': f['landmarks'], 'bbox': f['bbox'], 'color': theme.GOOD}])
                self._capture(f, source='photo')
            self.hint.setText(L(f'{len(results) - missing} ta rasmdan yuz olindi' + (f', {missing} tasida yuz yo‘q' if missing else ''),
                                f'Лицо взято из {len(results) - missing} фото' + (f', на {missing} лица нет' if missing else '')))
            self.hint.setStyleSheet(f'font-size:16px;font-weight:700;color:{theme.GOOD}')

        run_task(work, done)

    def save_person(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.information(self, 'GFRAM', L('F.I.Sh. ni kiriting.', 'Введите Ф.И.О.'))
            return
        if not self.captured:
            QMessageBox.information(self, 'GFRAM', L('Avval yuzni suratga oling yoki rasm qo‘shing.',
                                                     'Сначала снимите лицо или добавьте фото.'))
            return
        for f in self.captured:
            self.engine.enroll(name, f, save=False, source=f.get('source', 'camera'))
        self.engine.update_person(name, **{k: e.text() for k, e in self.fields.items()})
        self.engine.save()
        n = len(self.captured)
        total = len(self.engine.face_indices(name))
        self.reset_capture()
        self.save_status.setText(L(f'✅ «{name}» saqlandi: +{n} shablon (jami {total}). Rasm saqlanmadi.',
                                   f'✅ «{name}» сохранён: +{n} шаблонов (всего {total}). Фото не сохранено.'))
        self.save_status.setStyleSheet(f'color:{theme.GOOD};font-weight:700')
        self.name_edit.clear()
        for e in self.fields.values():
            e.clear()
        self.on_db_changed()

    # ================================================================== management

    def _build_manage(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 12, 0, 0)
        lay.setSpacing(16)

        left = QVBoxLayout()
        tiles = QHBoxLayout()
        self.t_people = StatTile('0', L('odam', 'человек'), theme.ACCENT)
        self.t_templates = StatTile('0', L('shablon', 'шаблонов'), theme.ACCENT2)
        self.t_photos = StatTile('0', L('saqlangan rasm', 'сохранённых фото'), theme.GOOD)
        for t in (self.t_people, self.t_templates, self.t_photos):
            tiles.addWidget(t)
        left.addLayout(tiles)
        self.search = QLineEdit()
        self.search.setPlaceholderText(L('Ism, lavozim, bo‘lim yoki ID bo‘yicha qidirish…', 'Поиск по имени, должности, отделу или ID…'))
        self.search.textChanged.connect(self.refresh)
        left.addWidget(self.search)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([L('F.I.Sh.', 'Ф.И.О.'), L('Lavozimi', 'Должность'), L('Bo‘lim', 'Отдел'),
                                              'ID', L('Shablon', 'Шабл.'), L('Qo‘shilgan', 'Добавлен'),
                                              L('Oxirgi tanilgan', 'Последнее распознавание')])
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        h.setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self.show_person)
        left.addWidget(self.table, 1)
        lay.addLayout(left, 3)

        right = QVBoxLayout()
        c, cl = card(L('Kartochka', 'Карточка'))
        self.d_name = label('—', size=20, bold=True)
        cl.addWidget(self.d_name)
        self.d_info = label('', muted=True)
        cl.addWidget(self.d_info)
        form = QFormLayout()
        self.d_fields = {}
        for k, t in card_fields():
            e = QLineEdit()
            form.addRow(t, e)
            self.d_fields[k] = e
        cl.addLayout(form)
        r = QHBoxLayout()
        r.addWidget(button(L('Saqlash', 'Сохранить'), 'primary', self.save_card))
        r.addWidget(button(L('Nomini o‘zgartirish', 'Переименовать'), slot=self.rename))
        r.addWidget(button(L('O‘chirish', 'Удалить'), 'danger', self.delete_person))
        cl.addLayout(r)
        right.addWidget(c)

        c, cl = card(L('Shablonlar', 'Шаблоны'),
                     L('Har bir shablon — bitta suratdan olingan himoyalangan vektorlar. Rasmning o‘zi yo‘q.',
                       'Каждый шаблон — защищённые векторы из одного снимка. Самого фото нет.'))
        self.t_table = QTableWidget(0, 6)
        self.t_table.setHorizontalHeaderLabels(['#', L('Qo‘shilgan', 'Добавлен'), L('Manba', 'Источник'), 'Yaw',
                                                L('Sifat', 'Качество'), ''])
        self.t_table.verticalHeader().setVisible(False)
        self.t_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.t_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.t_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.t_table.setMinimumHeight(220)
        cl.addWidget(self.t_table)
        cl.addWidget(button(L('Rasmdan shablon qo‘shish…', 'Добавить шаблон из фото…'), slot=self.add_template))
        right.addWidget(c, 1)
        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(470)
        lay.addWidget(rw)
        return w

    def refresh(self):
        persons = self.engine.persons()
        q = self.search.text().strip().lower()
        rows = []
        for p in persons:
            m = self.engine.person(p['name'])
            text = ' '.join([p['name'], m.get('position', ''), m.get('department', ''), m.get('code', '')]).lower()
            if q and q not in text:
                continue
            rows.append((p, m))
        self.t_people.set(str(len(persons)))
        self.t_templates.set(str(sum(p['faces'] for p in persons)))
        rep = self.engine.storage_report()
        n_img = rep['image_files'] + rep['embedded_images']
        self.t_photos.set(str(n_img), theme.GOOD if n_img == 0 else theme.BAD)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for i, (p, m) in enumerate(rows):
            vals = [p['name'], m.get('position', ''), m.get('department', ''), m.get('code', ''), str(p['faces']),
                    (m.get('created') or '')[:16], (m.get('last_seen') or '—')[:16]]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j == 4:
                    it.setData(Qt.DisplayRole, p['faces'])
                    it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, it)
        self.table.setSortingEnabled(True)

    def selected(self) -> Optional[str]:
        rows = self.table.selectionModel().selectedRows()
        return self.table.item(rows[0].row(), 0).text() if rows else None

    def show_person(self):
        name = self.selected()
        if not name:
            return
        m = self.engine.person(name)
        self.d_name.setText(name)
        self.d_info.setText(L(f'Qo‘shilgan: {m.get("created") or "—"} · tanilgan: {m.get("seen", 0)} marta · oxirgi: {m.get("last_seen") or "—"}',
                              f'Добавлен: {m.get("created") or "—"} · распознан: {m.get("seen", 0)} раз · последний: {m.get("last_seen") or "—"}'))
        for k, e in self.d_fields.items():
            e.setText(m.get(k, ''))
        t = m.get('templates', [])
        self.t_table.setRowCount(len(t))
        src = {'camera': L('kamera', 'камера'), 'photo': L('rasm', 'фото'), 'lfw': 'LFW', 'folder': L('papka', 'папка')}
        for i, tm in enumerate(t):
            vals = [str(i + 1), (tm.get('added') or '—')[:16], src.get(tm.get('source'), tm.get('source', '?')),
                    f'{tm["yaw"]:+.0f}°' if 'yaw' in tm else '—', num(tm['quality'], 1) if 'quality' in tm else '—']
            for j, v in enumerate(vals):
                self.t_table.setItem(i, j, QTableWidgetItem(v))
            b = QPushButton('×')
            b.setFixedSize(30, 24)
            b.setStyleSheet(f'padding:0;font-size:16px;font-weight:700;color:{theme.BAD}')
            b.setToolTip(L('Shablonni o‘chirish', 'Удалить шаблон'))
            b.clicked.connect(lambda _=False, k=i: self.delete_template(name, k))
            self.t_table.setCellWidget(i, 5, b)

    def save_card(self):
        name = self.selected()
        if name:
            self.engine.update_person(name, **{k: e.text() for k, e in self.d_fields.items()})
            self.refresh()
            self._select(name)

    def _select(self, name):
        for i in range(self.table.rowCount()):
            if self.table.item(i, 0).text() == name:
                self.table.selectRow(i)
                return

    def rename(self):
        name = self.selected()
        if not name:
            return
        new, ok = QInputDialog.getText(self, 'GFRAM', L('Yangi F.I.Sh.:', 'Новое Ф.И.О.:'), text=name)
        if ok and new.strip() and new.strip() != name:
            if not self.engine.rename_person(name, new):
                QMessageBox.warning(self, 'GFRAM', L('Bunday nom bazada bor.', 'Такое имя уже есть в базе.'))
                return
            self.refresh()
            self._select(new.strip())
            self.on_db_changed()

    def delete_person(self):
        name = self.selected()
        if name and QMessageBox.question(self, 'GFRAM', L(f'«{name}» va uning barcha shablonlari o‘chirilsinmi?',
                                                          f'Удалить «{name}» и все его шаблоны?')) == QMessageBox.Yes:
            self.engine.remove(name)
            self.refresh()
            self.on_db_changed()

    def delete_template(self, name, k):
        self.engine.delete_template(name, k)
        self.refresh()
        self._select(name)
        self.on_db_changed()

    def add_template(self):
        name = self.selected()
        if not name:
            return
        files = pick_images(self, multiple=True)
        added = 0
        for p in files:
            img = read_image(p)
            f = self.engine.analyze_largest(img) if img is not None else None
            if f is not None:
                self.engine.enroll(name, f, save=False, source='photo')
                added += 1
        self.engine.save()
        self.refresh()
        self._select(name)
        self.on_db_changed()
        if files and added < len(files):
            QMessageBox.information(self, 'GFRAM', L(f'{len(files) - added} ta rasmda yuz topilmadi.',
                                                     f'На {len(files) - added} фото лицо не найдено.'))
