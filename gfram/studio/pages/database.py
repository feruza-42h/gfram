"""
Database on disk: every stored file, a byte-level scan proving no photos are kept,
a step-by-step proof that recognition works without photos, and the key-rotation demo.
"""

import shutil
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QHeaderView, QInputDialog, QMessageBox, QPlainTextEdit,
                               QProgressBar, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import data, theme
from ..i18n import L, num, pct
from ..widgets import FaceView, StatTile, button, card, heading, label, pick_images, read_image, run_task  # noqa: F401


def lfw_ordered_people():
    """LFW identities with 2+ photos first (so 1:N can be tested on unseen photos), then the rest."""
    people = data.lfw_people()
    multi = [n for n, imgs in people.items() if len(imgs) >= 2]
    single = [n for n, imgs in people.items() if len(imgs) < 2]
    return multi + single, people


def file_role(path: str) -> str:
    name = Path(path).name
    return {
        'persons.pkl': L('Shablonlar bazasi: himoyalangan vektorlar + shifrlangan nuqtalar',
                         'База шаблонов: защищённые векторы + зашифрованные точки'),
        'template.key': L('Maxfiy kalit (256 bit) — amalda bazadan alohida saqlanadi',
                          'Секретный ключ (256 бит) — на практике хранится отдельно от базы'),
        'people.json': L('Kartochkalar: ism, lavozim, bo‘lim (matn)', 'Карточки: имя, должность, отдел (текст)'),
        'events.jsonl': L('Tanish jurnali: vaqt, ism, ehtimollik (matn)', 'Журнал: время, имя, вероятность (текст)'),
    }.get(name, '')


def key_role(key: str) -> str:
    return {
        'format': L('baza formati (4 = himoyalangan)', 'формат базы (4 = защищённая)'),
        'persons': L('ism → raqam', 'имя → номер'),
        'next_id': L('keyingi raqam', 'следующий номер'),
        'sample_names': L('har bir shablon kimga tegishli', 'кому принадлежит каждый шаблон'),
        'key_fingerprint': L('kalit izi (kalitning o‘zi emas)', 'отпечаток ключа (не сам ключ)'),
        'sample_landmarks_enc': L('478 nuqta, AES-256-GCM bilan shifrlangan', '478 точек, зашифрованы AES-256-GCM'),
        'sample_app': L('ko‘rinish vektorlari R_a·a (512), kalit bilan burilgan', 'векторы внешности R_a·a (512), повёрнуты ключом'),
        'sample_quality': L('sifat ko‘rsatkichlari (2 son)', 'показатели качества (2 числа)'),
        'appearance_model': L('neyron tarmoq nomi', 'имя нейросети'),
        'geo': L('geometrik vektorlar R_g·g (128), kalit bilan burilgan', 'геометрические векторы R_g·g (128), повёрнуты ключом'),
        'pose': L('bosh holati (3 burchak)', 'поза головы (3 угла)'),
        'geo_version': L('geometrik model versiyasi', 'версия геометрической модели'),
        'total_recognitions': L('tanishlar soni', 'число распознаваний'),
    }.get(key, '')


def human_size(n: int) -> str:
    if n >= 1 << 20:
        return f'{n / (1 << 20):.2f} MB'.replace('.', ',')
    if n >= 1024:
        return f'{n / 1024:.1f} KB'.replace('.', ',')
    return f'{n} B'


def make_table(headers, stretch=()):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setSelectionMode(QTableWidget.NoSelection)
    h = t.horizontalHeader()
    h.setSectionResizeMode(QHeaderView.ResizeToContents)
    for j in stretch:
        h.setSectionResizeMode(j, QHeaderView.Stretch)
    return t


class DatabasePage(QWidget):
    def __init__(self, engine, on_db_changed):
        super().__init__()
        self.setObjectName('page')
        self.engine = engine
        self.on_db_changed = on_db_changed
        self.task = None
        self.proof = {}

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
        root.addWidget(heading(L('Baza: fayllar, isbot va himoya', 'База: файлы, доказательство и защита'),
                               L('Bu yerda diskda aynan nima saqlanishi ko‘rinadi. Rasm saqlanmaydi: faqat kalit bilan himoyalangan '
                                 'raqamli shablonlar va shifrlangan nuqtalar. Tanish faqat shu fayldan ishlaydi.',
                                 'Здесь видно, что именно хранится на диске. Фото не хранятся: только защищённые ключом числовые '
                                 'шаблоны и зашифрованные точки. Распознавание работает только по этому файлу.')))

        tiles = QHBoxLayout()
        self.t_persons = StatTile('0', L('odam', 'человек'), theme.ACCENT)
        self.t_faces = StatTile('0', L('shablon', 'шаблонов'), theme.ACCENT2)
        self.t_size = StatTile('0', L('baza fayli', 'файл базы'), theme.FUSION)
        self.t_per = StatTile('0', L('bitta shablonga', 'на один шаблон'), theme.TEXT)
        self.t_images = StatTile('0', L('rasm fayllari / ichida rasm', 'файлов-фото / фото внутри'), theme.GOOD)
        self.t_key = StatTile('—', L('kalit izi', 'отпечаток ключа'), theme.WARN)
        for t in (self.t_persons, self.t_faces, self.t_size, self.t_per, self.t_images, self.t_key):
            tiles.addWidget(t)
        root.addLayout(tiles)

        # ---------------- files on disk
        c, lay = card(L('1. Diskdagi barcha fayllar', '1. Все файлы на диске'),
                      L('Har bir fayl bayt-bayt tekshiriladi: JPEG, PNG, HEIC, WebP, GIF, TIFF imzolari qidiriladi.',
                        'Каждый файл проверяется побайтно: ищутся сигнатуры JPEG, PNG, HEIC, WebP, GIF, TIFF.'))
        row = QHBoxLayout()
        self.root_label = label('', muted=True)
        row.addWidget(self.root_label, 1)
        row.addWidget(button(L('🔍  Qayta skanerlash', '🔍  Просканировать заново'), 'primary', self.refresh))
        row.addWidget(button(L('📂  Finder’da ochish', '📂  Открыть в Finder'), slot=self.open_folder))
        lay.addLayout(row)
        self.files = make_table([L('Fayl', 'Файл'), L('Hajm', 'Размер'), L('Vazifasi', 'Назначение'), L('Rasm bormi?', 'Есть фото?')], (2,))
        self.files.setMinimumHeight(190)
        lay.addWidget(self.files)
        self.scan_verdict = label('', size=17, bold=True)
        lay.addWidget(self.scan_verdict)
        root.addWidget(c)

        mid = QVBoxLayout()
        mid.setSpacing(16)
        c, lay = card(L('2. persons.pkl ichida nima bor', '2. Что внутри persons.pkl'))
        self.structure = make_table([L('Maydon', 'Поле'), L('Turi', 'Тип'), L('Soni', 'Кол-во'), L('Hajm', 'Размер'),
                                     L('Ma’nosi', 'Смысл')], (4,))
        self.structure.setMinimumHeight(470)
        lay.addWidget(self.structure)
        mid.addWidget(c, 1)
        c, lay = card(L('3. Bitta odamning yozuvi', '3. Запись одного человека'))
        self.person_box = QComboBox()
        self.person_box.currentIndexChanged.connect(self.show_record)
        lay.addWidget(self.person_box)
        self.record = QPlainTextEdit()
        self.record.setReadOnly(True)
        self.record.setStyleSheet('font-family: Menlo, monospace; font-size: 12px;')
        self.record.setMinimumHeight(300)
        lay.addWidget(self.record)
        lay.addWidget(button(L('Fayl boshidagi xom baytlar', 'Сырые байты начала файла'), slot=self.show_hex))
        mid.addWidget(c, 1)
        root.addLayout(mid)

        # ---------------- proof
        c, lay = card(L('4. Isbot: rasm o‘chirilgan — tanish baribir ishlaydi', '4. Доказательство: фото удалено — распознавание работает'),
                      L('Rasm vaqtinchalik papkaga nusxalanadi, undan odam qo‘shiladi, keyin rasm o‘chiriladi, disk skanerlanadi, '
                        'baza xotiradan tashlab yuborilib fayldan qayta o‘qiladi va shu odamning BOSHQA rasmi tekshiriladi.',
                        'Фото копируется во временную папку, по нему добавляется человек, затем фото удаляется, диск сканируется, '
                        'база выбрасывается из памяти и читается заново из файла, и проверяется ДРУГОЕ фото этого человека.'))
        steps = QHBoxLayout()
        self.p_btns = [
            button(L('① Rasm → bazaga', '① Фото → в базу'), 'primary', self.proof_enroll),
            button(L('② Rasmni o‘chirish', '② Удалить фото'), slot=self.proof_delete),
            button(L('③ Diskni skanerlash', '③ Сканировать диск'), slot=self.proof_scan),
            button(L('④ Bazani fayldan qayta yuklash', '④ Перечитать базу из файла'), slot=self.proof_reload),
            button(L('⑤ Boshqa rasm bilan tanish', '⑤ Распознать по другому фото'), slot=self.proof_recognize),
        ]
        for b in self.p_btns:
            steps.addWidget(b)
        lay.addLayout(steps)
        pr = QHBoxLayout()
        self.p_view_a = FaceView(L('Qo‘shilgan rasm', 'Добавленное фото'), 210)
        self.p_view_b = FaceView(L('Tekshiruv rasmi', 'Фото для проверки'), 210)
        for v in (self.p_view_a, self.p_view_b):
            v.setFixedWidth(250)
        pr.addWidget(self.p_view_a)
        self.p_log = label('', wrap=True)
        self.p_log.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        pr.addWidget(self.p_log, 1)
        pr.addWidget(self.p_view_b)
        lay.addLayout(pr)
        root.addWidget(c)
        self._proof_state(0)

        # ---------------- key rotation
        c, lay = card(L('5. Kalitni almashtirish (shablonlarni bekor qilish)', '5. Смена ключа (отзыв шаблонов)'),
                      L('Hujumchi baza faylini o‘g‘irladi deylik. Egasi kalitni almashtiradi — baza rasmlarsiz qayta himoyalanadi. '
                        'Shu odamning yangi rasmi joriy bazada taniladi, o‘g‘irlangan nusxada esa — yo‘q.',
                        'Допустим, злоумышленник украл файл базы. Владелец меняет ключ — база перезащищается без фото. '
                        'Новое фото того же человека узнаётся в текущей базе, а в украденной копии — нет.'))
        steps = QHBoxLayout()
        steps.addWidget(button(L('① Bazani «o‘g‘irlash»', '① «Украсть» базу'), slot=self.steal))
        steps.addWidget(button(L('② Kalitni almashtirish', '② Сменить ключ'), 'primary', self.rotate))
        steps.addWidget(button(L('③ Rasm bilan tekshirish…', '③ Проверить по фото…'), slot=self.check))
        lay.addLayout(steps)
        row = QHBoxLayout()
        self.check_view = FaceView(L('Tekshiruv rasmi', 'Фото для проверки'), 200)
        self.check_view.setFixedWidth(260)
        row.addWidget(self.check_view)
        self.demo_log = label('', wrap=True)
        self.demo_log.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        row.addWidget(self.demo_log, 1)
        lay.addLayout(row)
        root.addWidget(c)

        # ---------------- bulk
        c, lay = card(L('6. Bazani to‘ldirish va tozalash', '6. Наполнение и очистка базы'),
                      L('Katta bazada qidiruvni ko‘rsatish uchun LFW dan odamlar qo‘shiladi (har biridan 1 ta rasm).',
                        'Чтобы показать поиск по большой базе, добавляются люди из LFW (по 1 фото каждого).'))
        r = QHBoxLayout()
        self.lfw_count = QComboBox()
        for n in (100, 500, 1680, 5749):
            self.lfw_count.addItem(L(f'{n} ta odam', f'{n} человек'), n)
        self.lfw_count.setCurrentIndex(1)
        r.addWidget(self.lfw_count)
        r.addWidget(button(L('LFW bilan to‘ldirish', 'Заполнить из LFW'), slot=self.import_lfw))
        r.addWidget(button(L('Papkadan import…', 'Импорт из папки…'), slot=self.import_folder))
        r.addStretch(1)
        r.addWidget(button(L('Bazani tozalash', 'Очистить базу'), 'danger', self.clear))
        lay.addLayout(r)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)
        self.status = label('', muted=True)
        lay.addWidget(self.status)
        root.addWidget(c)
        self.refresh()

    # ------------------------------------------------------------------ inspection

    def refresh(self):
        st = self.engine.stats()
        rep = self.engine.storage_report()
        self.t_persons.set(str(st['persons']))
        self.t_faces.set(str(st['faces']))
        self.t_size.set(human_size(st['db_bytes']))
        self.t_per.set(human_size(st['db_bytes'] // st['faces']) if st['faces'] else '—')
        self.t_key.set(st['fingerprint'][:8])
        imgs = rep['image_files'] + rep['embedded_images']
        self.t_images.set(f'{rep["image_files"]} / {rep["embedded_images"]}', theme.GOOD if imgs == 0 else theme.BAD)
        self.root_label.setText(rep['root'])

        self.files.setRowCount(len(rep['files']))
        for i, f in enumerate(rep['files']):
            has = f['images'] or f['image_ext']
            vals = [f['path'], human_size(f['bytes']), file_role(f['path']),
                    (', '.join(f['images']) or L('rasm fayli', 'файл-фото')) if has else L('yo‘q ✓', 'нет ✓')]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j == 3:
                    it.setForeground(QColor(theme.BAD if has else theme.GOOD))
                self.files.setItem(i, j, it)
        total = sum(f['bytes'] for f in rep['files'])
        if imgs == 0:
            self.scan_verdict.setText(L(f'✅ {len(rep["files"])} ta fayl ({human_size(total)}) tekshirildi — birorta ham rasm yo‘q.',
                                        f'✅ Проверено {len(rep["files"])} файлов ({human_size(total)}) — ни одного фото.'))
            self.scan_verdict.setStyleSheet(f'font-size:17px;font-weight:700;color:{theme.GOOD}')
        else:
            self.scan_verdict.setText(L(f'⚠ Rasm topildi: {imgs}', f'⚠ Найдены фото: {imgs}'))
            self.scan_verdict.setStyleSheet(f'font-size:17px;font-weight:700;color:{theme.BAD}')

        self.structure.setRowCount(len(rep['structure']))
        for i, s in enumerate(rep['structure']):
            vals = [s['key'], s['type'], '' if s['items'] is None else str(s['items']), human_size(s['bytes']), key_role(s['key'])]
            for j, v in enumerate(vals):
                self.structure.setItem(i, j, QTableWidgetItem(v))

        current = self.person_box.currentText()
        names = sorted(p['name'] for p in self.engine.persons())
        self.person_box.blockSignals(True)
        self.person_box.clear()
        self.person_box.addItems(names)
        if current in names:
            self.person_box.setCurrentText(current)
        self.person_box.blockSignals(False)
        self.show_record()

    def show_record(self):
        name = self.person_box.currentText()
        idx = self.engine.face_indices(name) if name else []
        if not idx:
            self.record.setPlainText(L('Baza bo‘sh.', 'База пуста.'))
            return
        r = self.engine.stored_record(idx[0])
        m = self.engine.person(name)
        enc, app, geo = r['landmarks_enc'], r['app_protected'], r['geo_protected']
        fmt = lambda v: '[' + ', '.join(f'{x:+.3f}' for x in v[:5]) + ', …]'
        lines = [
            f'{L("Ism", "Имя")}: {name}    {L("shablonlar", "шаблонов")}: {len(idx)}',
            f'{L("Lavozim / bo‘lim", "Должность / отдел")}: {m.get("position") or "—"} / {m.get("department") or "—"}',
            '',
            f'{L("Shifrlangan 478 nuqta", "Зашифрованные 478 точек")} (AES-256-GCM): {len(enc)} {L("bayt", "байт")}',
            '  ' + enc[:30].hex(' '),
            '  ' + enc[30:60].hex(' ') + ' …',
            '',
            f'R_g·g (128): {fmt(geo) if geo is not None else "—"}',
            f'R_a·a (512): {fmt(app)}',
            f'{L("Bosh holati", "Поза")}: ' + ', '.join(f'{x:+.1f}°' for x in r['pose']),
            f'{L("Sifat", "Качество")}: {num(r["quality"]["app_norm"], 1)} · {L("tiniqlik", "резкость")} {num(r["quality"]["sharp"], 2)}',
            '',
            f'{L("Rasm", "Фото")}: ' + L('YO‘Q — rasm hech qayerda saqlanmaydi', 'НЕТ — фото нигде не хранится'),
            L('Vektorlar maxfiy kalit bilan burilgan: boshqa kalitdagi bazalar bilan bog‘lab bo‘lmaydi (AUC≈0,5), '
              'kalit almashtirilsa eski nusxa yaroqsiz.',
              'Векторы повёрнуты секретным ключом: их нельзя связать с базами под другим ключом (AUC≈0,5), '
              'после смены ключа старая копия бесполезна.'),
        ]
        self.record.setPlainText('\n'.join(lines))

    def show_hex(self):
        db = self.engine.rec._db_file
        if not db.exists():
            return
        raw = db.read_bytes()[:768]
        out = []
        for i in range(0, len(raw), 16):
            chunk = raw[i:i + 16]
            out.append(f'{i:06x}  {chunk.hex(" "):<48}  ' + ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk))
        self.record.setPlainText(f'persons.pkl — {L("birinchi 768 bayt", "первые 768 байт")}\n\n' + '\n'.join(out))

    def open_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.engine.home)))

    # ------------------------------------------------------------------ proof without photos

    def _proof_state(self, step):
        for i, b in enumerate(self.p_btns):
            b.setEnabled(i <= step)

    def _plog(self, text, reset=False):
        self.p_log.setText(text if reset else (self.p_log.text() + '<br>' + text))

    def proof_enroll(self):
        files = pick_images(self)
        if not files:
            return
        name, ok = QInputDialog.getText(self, 'GFRAM', L('Bu odamning ismi:', 'Имя этого человека:'),
                                        text=Path(files[0]).parent.name.replace('_', ' '))
        if not ok or not name.strip():
            return
        tmp_dir = Path(tempfile.mkdtemp(prefix='gfram_proof_'))
        copy = tmp_dir / Path(files[0]).name
        shutil.copy2(files[0], copy)
        img = read_image(copy)
        face = self.engine.analyze_largest(img) if img is not None else None
        if face is None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._plog(L('Yuz topilmadi — boshqa rasm tanlang.', 'Лицо не найдено — выберите другое фото.'), True)
            return
        self.engine.enroll(name.strip(), face, source='photo')
        self.proof = {'dir': tmp_dir, 'file': copy, 'name': name.strip(), 'source': files[0]}
        self.p_view_a.set_image(img, [{'landmarks': face['landmarks'], 'bbox': face['bbox'], 'color': theme.ACCENT}])
        self._plog(L(f'<b>①</b> Rasm nusxasi: <code>{copy}</code><br>&nbsp;&nbsp;&nbsp;«{name.strip()}» bazaga qo‘shildi '
                     f'(faqat himoyalangan shablon yozildi).',
                     f'<b>①</b> Копия фото: <code>{copy}</code><br>&nbsp;&nbsp;&nbsp;«{name.strip()}» добавлен в базу '
                     f'(записан только защищённый шаблон).'), True)
        self._proof_state(1)
        self.refresh()
        self.on_db_changed()

    def proof_delete(self):
        f = self.proof.get('file')
        if not f:
            return
        shutil.rmtree(self.proof['dir'], ignore_errors=True)
        self.p_view_a.set_image(None)
        self.p_view_a.placeholder = L('Rasm o‘chirildi', 'Фото удалено')
        self._plog(L(f'<b>②</b> Rasm o‘chirildi. Fayl mavjudmi? <b style="color:{theme.GOOD}">{f.exists()}</b>',
                     f'<b>②</b> Фото удалено. Файл существует? <b style="color:{theme.GOOD}">{f.exists()}</b>'))
        self._proof_state(2)

    def proof_scan(self):
        rep = self.engine.storage_report()
        n = rep['image_files'] + rep['embedded_images']
        col = theme.GOOD if n == 0 else theme.BAD
        self._plog(L(f'<b>③</b> {len(rep["files"])} ta fayl bayt-bayt tekshirildi: rasm topilmadi — '
                     f'<b style="color:{col}">{n} ta</b>.',
                     f'<b>③</b> Побайтно проверено файлов: {len(rep["files"])}. Найдено фото: '
                     f'<b style="color:{col}">{n}</b>.'))
        self.refresh()
        self._proof_state(3)

    def proof_reload(self):
        ms = self.engine.reload_from_disk()
        st = self.engine.stats()
        self._plog(L(f'<b>④</b> Xotiradagi baza tashlab yuborildi va <code>persons.pkl</code> dan qayta o‘qildi '
                     f'({ms:.0f} ms): {st["persons"]} odam, {st["faces"]} shablon.',
                     f'<b>④</b> База в памяти выброшена и перечитана из <code>persons.pkl</code> '
                     f'({ms:.0f} мс): {st["persons"]} человек, {st["faces"]} шаблонов.'))
        self._proof_state(4)
        self.on_db_changed()

    def proof_recognize(self):
        start = str(Path(self.proof.get('source', str(Path.home()))).parent)
        from PySide6.QtWidgets import QFileDialog
        from ..widgets import IMAGE_FILTER
        f, _ = QFileDialog.getOpenFileName(self, L('Shu odamning BOSHQA rasmi', 'ДРУГОЕ фото этого человека'), start, IMAGE_FILTER)
        if not f:
            return
        img = read_image(f)
        face = self.engine.analyze_largest(img) if img is not None else None
        if face is None:
            self._plog(L('Yuz topilmadi.', 'Лицо не найдено.'))
            return
        res = self.engine.identify(face, top_k=1)['candidates']
        thr = self.engine.current_threshold()
        name, p = res[0] if res else ('—', 0.0)
        ok = p >= thr
        col = theme.GOOD if ok else theme.BAD
        self.p_view_b.set_image(img, [{'landmarks': face['landmarks'], 'bbox': face['bbox'], 'color': col,
                                       'label': name if ok else L('Noma’lum', 'Неизвестный')}])
        self._plog(L(f'<b>⑤</b> Natija: <b style="color:{col}">{name if ok else "Noma’lum"}</b> — {pct(p, 2)}. ',
                     f'<b>⑤</b> Результат: <b style="color:{col}">{name if ok else "Неизвестный"}</b> — {pct(p, 2)}. '))
        if ok and name == self.proof.get('name'):
            self._plog(L(f'<b style="color:{theme.GOOD}">✅ ISBOTLANDI: rasm o‘chirilgan, diskda rasm yo‘q, baza fayldan o‘qildi — '
                         f'odam baribir tanildi.</b>',
                         f'<b style="color:{theme.GOOD}">✅ ДОКАЗАНО: фото удалено, на диске фото нет, база прочитана из файла — '
                         f'человек всё равно распознан.</b>'))
        self.engine.log_event(name, p, ok, 'photo')

    # ------------------------------------------------------------------ key rotation

    def steal(self):
        if not self.engine.steal_copy():
            self.demo_log.setText(L('Baza bo‘sh — avval odam qo‘shing.', 'База пуста — сначала добавьте людей.'))
            return
        fp = self.engine.stolen_copy['fingerprint']
        self.demo_log.setText(L(f'🕵️ Nusxa olindi: {len(self.engine.stolen_copy["names"])} ta shablon, kalit {fp[:8]}.',
                                f'🕵️ Копия снята: {len(self.engine.stolen_copy["names"])} шаблонов, ключ {fp[:8]}.'))

    def rotate(self):
        if not self.engine.stats()['faces']:
            return
        old = self.engine.stats()['fingerprint']
        t = time.perf_counter()
        new = self.engine.rotate_key()
        dt = (time.perf_counter() - t) * 1000
        self.demo_log.setText(self.demo_log.text() + '\n\n' + L(
            f'🔑 Kalit almashtirildi: {old[:8]} → {new[:8]} ({dt:.0f} ms, rasmlarsiz).',
            f'🔑 Ключ заменён: {old[:8]} → {new[:8]} ({dt:.0f} мс, без фото).'))
        self.refresh()
        self.on_db_changed()

    def check(self):
        files = pick_images(self)
        if not files:
            return
        img = read_image(files[0])
        face = self.engine.analyze_largest(img) if img is not None else None
        if face is None:
            self.demo_log.setText(self.demo_log.text() + '\n\n' + L('Yuz topilmadi.', 'Лицо не найдено.'))
            return
        self.check_view.set_image(img, [{'landmarks': face['landmarks'], 'bbox': face['bbox'], 'color': theme.ACCENT}])
        thr = self.engine.current_threshold()
        cur = self.engine.identify(face, top_k=1)['candidates']
        lines = []
        if cur:
            n, p = cur[0]
            ok = p >= thr
            lines.append(L(f'Joriy baza (yangi kalit): {"✅ " + n if ok else "❌ topilmadi"} — {pct(p, 1)}',
                           f'Текущая база (новый ключ): {"✅ " + n if ok else "❌ не найден"} — {pct(p, 1)}'))
        st = self.engine.match_against_stolen(face)
        if st:
            ok = st['probability'] >= thr
            lines.append(L(f'O‘g‘irlangan nusxa (eski kalit {st["fingerprint"][:8]}): '
                           f'{"⚠️ mos keldi " + st["name"] if ok else "🔒 hech kim topilmadi"} — eng yuqori {pct(st["probability"], 2)}',
                           f'Украденная копия (старый ключ {st["fingerprint"][:8]}): '
                           f'{"⚠️ совпало " + st["name"] if ok else "🔒 никто не найден"} — максимум {pct(st["probability"], 2)}'))
            if st['fingerprint'] != self.engine.stats()['fingerprint'] and not ok:
                lines.append(L('Natija: kalit almashtirilgach, o‘g‘irlangan shablonlar yaroqsiz.',
                               'Итог: после смены ключа украденные шаблоны бесполезны.'))
        else:
            lines.append(L('(O‘g‘irlangan nusxa yo‘q — ① qadamni bajaring.)', '(Украденной копии нет — выполните шаг ①.)'))
        self.demo_log.setText(self.demo_log.text() + '\n\n' + '\n'.join(lines))

    # ------------------------------------------------------------------ bulk

    def _run_import(self, jobs, label_text, source):
        if self.task:
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, len(jobs))
        engine = self.engine

        def work(task):
            added = skipped = 0
            t0 = time.perf_counter()
            for i, (name, path) in enumerate(jobs):
                img = read_image(path)
                face = engine.analyze_largest(img) if img is not None else None
                if face is None:
                    skipped += 1
                else:
                    engine.enroll(name, face, save=False, source=source)
                    added += 1
                if i % 20 == 0 or i == len(jobs) - 1:
                    task.report((i + 1, added, skipped))
            engine.save()
            return added, skipped, time.perf_counter() - t0

        def progress(v):
            i, added, skipped = v
            self.progress.setValue(i)
            self.status.setText(f'{label_text}: {i}/{len(jobs)}')

        def done(r):
            added, skipped, dt = r
            self.task = None
            self.progress.setVisible(False)
            self.status.setText(L(f'✅ Qo‘shildi: {added} ta shablon, yuz topilmadi: {skipped}. Vaqt {dt:.0f} s. Rasmlar saqlanmadi.',
                                  f'✅ Добавлено: {added} шаблонов, без лица: {skipped}. Время {dt:.0f} с. Фото не сохранены.'))
            self.refresh()
            self.on_db_changed()

        def fail(e):
            self.task = None
            self.progress.setVisible(False)
            self.status.setText(e[-300:])

        self.task = run_task(work, done, fail, progress)

    def import_lfw(self):
        if not data.lfw_available():
            return
        n = self.lfw_count.currentData()
        order, people = lfw_ordered_people()
        existing = {p['name'] for p in self.engine.persons()}
        jobs = [(data.pretty_name(nm), people[nm][0]) for nm in order[:n] if data.pretty_name(nm) not in existing]
        self._run_import(jobs, 'LFW', 'lfw')

    def import_folder(self):
        from PySide6.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(self, L('Papka: har bir odam — alohida ichki papka',
                                                     'Папка: каждый человек — отдельная подпапка'), str(Path.home()))
        if not d:
            return
        jobs = [(sub.name.replace('_', ' '), f) for sub in sorted(Path(d).iterdir()) if sub.is_dir()
                for f in sorted(sub.iterdir()) if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')]
        if jobs:
            self._run_import(jobs, Path(d).name, 'folder')

    def clear(self):
        if QMessageBox.question(self, 'GFRAM', L('Butun baza o‘chirilsinmi?', 'Удалить всю базу?')) == QMessageBox.Yes:
            self.engine.clear()
            self.demo_log.setText('')
            self.refresh()
            self.on_db_changed()
