"""Live recognition from a camera or a video file, with the full pipeline visible."""

import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QThread, QWaitCondition, Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QProgressBar, QVBoxLayout, QWidget)

from .. import theme
from ..i18n import L, num, pct
from ..widgets import FaceView, button, card, heading, label

MAX_WIDTH = 960


def camera_names() -> List[str]:
    try:
        from PySide6.QtMultimedia import QMediaDevices
        return [d.description() for d in QMediaDevices.videoInputs()]
    except Exception:
        return []


class RecognitionThread(QThread):
    """Grabs frames (camera or video file), runs detection + hybrid recognition, emits results."""

    frame_ready = Signal(object, list, dict)   # image, faces, info
    failed = Signal(str)

    def __init__(self, engine, source, mirror=True):
        super().__init__()
        self.engine = engine
        self.source = source            # int: camera (Qt device index), str: video file
        self.mirror = mirror
        self._running = True
        self.tracks: List[Dict] = []
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._pending = None
        self._camera = None

    # ---------------- camera through Qt Multimedia (exact device, native macOS permission)

    def start(self):
        if isinstance(self.source, int):
            if not self._start_camera():
                QTimer.singleShot(0, lambda: self.failed.emit('open'))
                return
        super().start()

    def _start_camera(self) -> bool:
        from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices, QVideoSink
        devices = QMediaDevices.videoInputs()
        if not devices:
            return False
        dev = devices[self.source] if 0 <= self.source < len(devices) else devices[0]
        cam = QCamera(dev)
        formats = [f for f in dev.videoFormats() if f.resolution().width() >= f.resolution().height()]
        if formats:
            best = min(formats, key=lambda f: (abs(f.resolution().width() - 1280), -f.maxFrameRate()))
            cam.setCameraFormat(best)
        session = QMediaCaptureSession()
        sink = QVideoSink()
        session.setCamera(cam)
        session.setVideoSink(sink)
        sink.videoFrameChanged.connect(self._on_video_frame)
        cam.errorOccurred.connect(lambda err, msg: self.failed.emit(msg or 'camera'))
        self._camera = (cam, session, sink)
        cam.start()
        return True

    def _on_video_frame(self, frame):
        self._mutex.lock()
        self._pending = frame               # keep only the newest frame
        self._cond.wakeOne()
        self._mutex.unlock()

    def _next_camera_frame(self, timeout_ms=6000):
        self._mutex.lock()
        if self._pending is None and self._running:
            self._cond.wait(self._mutex, timeout_ms)
        vf, self._pending = self._pending, None
        self._mutex.unlock()
        if vf is None:
            return None
        img = vf.toImage()
        if img.isNull():
            return None
        if img.width() > MAX_WIDTH:
            img = img.scaledToWidth(MAX_WIDTH, Qt.SmoothTransformation)
        from ..widgets import qimage_to_bgr
        return qimage_to_bgr(img)

    def stop(self):
        self._running = False
        self._mutex.lock()
        self._cond.wakeAll()
        self._mutex.unlock()
        if self._camera:
            cam, _, sink = self._camera
            try:
                sink.videoFrameChanged.disconnect(self._on_video_frame)
            except Exception:
                pass
            cam.stop()
            self._camera = None

    def run(self):
        camera = isinstance(self.source, int)
        cap = None
        frame_period = 0
        if not camera:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.failed.emit('open')
                return
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            frame_period = 1.0 / fps_video if fps_video and fps_video < 120 else 0
        times = []
        waited = 0
        while self._running:
            t_frame = time.perf_counter()
            if camera:
                frame = self._next_camera_frame(1000)
                if frame is None:
                    waited += 1
                    if waited >= 8 and self._running:      # no picture for 8 s
                        self.failed.emit('read')
                        break
                    continue
                waited = 0
                if self.mirror:
                    frame = cv2.flip(frame, 1)
            else:
                ok, frame = cap.read()
                if not ok:                                  # loop the video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                h, w = frame.shape[:2]
                if w > MAX_WIDTH:
                    frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
            try:
                faces, info = self.process(frame)
            except Exception as e:  # keep the demo alive whatever happens on one frame
                faces, info = [], {'error': str(e)}
            now = time.perf_counter()
            times = [t for t in times if now - t < 1.0] + [now]
            info['fps'] = len(times)
            if self._running:
                self.frame_ready.emit(frame, faces, info)
            if frame_period:
                rest = frame_period - (time.perf_counter() - t_frame)
                if rest > 0:
                    self.msleep(int(rest * 1000))
        if cap is not None:
            cap.release()

    def process(self, frame):
        t0 = time.perf_counter()
        result = self.engine.analyze(frame)
        faces = result['faces']
        t_match = time.perf_counter()
        for f in faces:
            f['match'] = self.engine.identify(f, top_k=3)
        timing = dict(result['timing'], match=(time.perf_counter() - t_match) * 1000,
                      total=(time.perf_counter() - t0) * 1000)
        self._smooth(faces)
        return faces, {'timing': timing}

    def _smooth(self, faces):
        """Stabilise the displayed probability across frames (per face track)."""
        new_tracks = []
        for f in faces:
            x, y, w, h = f['bbox']
            c = np.array([x + w / 2, y + h / 2])
            cand = f['match']['candidates']
            name, p = cand[0] if cand else (None, 0.0)
            prev = None
            for t in self.tracks:
                if np.linalg.norm(t['c'] - c) < max(w, h) * 0.6:
                    prev = t
                    break
            if prev is not None and prev['name'] == name:
                p_s = 0.6 * prev['p'] + 0.4 * p
            else:
                p_s = p
            f['name'], f['p_smooth'] = name, p_s
            new_tracks.append({'c': c, 'name': name, 'p': p_s})
        self.tracks = new_tracks


class LivePage(QWidget):
    def __init__(self, engine, on_db_changed):
        super().__init__()
        self.setObjectName('page')
        self.engine = engine
        self.on_db_changed = on_db_changed
        self.thread: Optional[RecognitionThread] = None
        self.last_frame = None
        self.last_faces: List[Dict] = []
        self.burst_left = 0
        self.last_logged = {}

        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(18)

        # ---------------- left: video
        left = QVBoxLayout()
        left.setSpacing(12)
        left.addWidget(heading(L('Jonli tanish', 'Распознавание в реальном времени'),
                               L('Kamera → 478 nuqta → geometrik modul + neyron tarmoq → adaptiv birlashtirish → '
                                 'himoyalangan baza bo‘yicha qidiruv (1:N). Har bir kadr to‘liq qayta ishlanadi.',
                                 'Камера → 478 точек → геометрический модуль + нейросеть → адаптивное слияние → '
                                 'поиск по защищённой базе (1:N). Каждый кадр обрабатывается полностью.')))
        controls = QHBoxLayout()
        self.source_box = QComboBox()
        names = camera_names()
        for i in range(max(1, len(names))):
            self.source_box.addItem(f'📷 {names[i] if i < len(names) else L("Kamera", "Камера")} ({i})', i)
        self.source_box.addItem(L('🎞 Video fayl…', '🎞 Видеофайл…'), 'file')
        for i, n in enumerate(names):
            if 'iphone' not in n.lower():
                self.source_box.setCurrentIndex(i)
                break
        controls.addWidget(self.source_box, 1)
        self.start_btn = button(L('▶  Boshlash', '▶  Старт'), 'primary', self.toggle)
        controls.addWidget(self.start_btn)
        self.shot_btn = button(L('Kadrni saqlash', 'Сохранить кадр'), slot=self.save_frame)
        controls.addWidget(self.shot_btn)
        left.addLayout(controls)

        self.view = FaceView(L('Kamerani yoqish uchun «Boshlash» tugmasini bosing',
                               'Нажмите «Старт», чтобы включить камеру'), 420)
        self.view.show_mesh = False
        left.addWidget(self.view, 1)

        toggles = QHBoxLayout()
        for text, attr, default in ((L('Nuqtalar (478)', 'Точки (478)'), 'show_points', True),
                                    (L('Uchburchak to‘r', 'Сетка треугольников'), 'show_mesh', False),
                                    (L('Konturlar', 'Контуры'), 'show_contours', True),
                                    (L('Bosh holati o‘qlari', 'Оси позы головы'), 'show_pose', True),
                                    (L('Ramka', 'Рамка'), 'show_box', True)):
            cb = QCheckBox(text)
            cb.setChecked(default)
            setattr(self.view, attr, default)
            cb.toggled.connect(lambda v, a=attr: (setattr(self.view, a, v), self.view.update()))
            toggles.addWidget(cb)
        toggles.addStretch(1)
        left.addLayout(toggles)
        root.addLayout(left, 3)

        # ---------------- right: panels
        right = QVBoxLayout()
        right.setSpacing(12)

        c, lay = card(L('Natija', 'Результат'))
        self.result_name = label('—', size=30, bold=True)
        self.result_prob = label('', size=15)
        lay.addWidget(self.result_name)
        lay.addWidget(self.result_prob)
        self.prob_bar = QProgressBar()
        self.prob_bar.setRange(0, 1000)
        self.prob_bar.setTextVisible(False)
        lay.addWidget(self.prob_bar)
        self.candidates = label('', muted=True)
        lay.addWidget(self.candidates)
        mode = QHBoxLayout()
        mode.addWidget(label(L('Rejim:', 'Режим:'), muted=True, wrap=False))
        self.mode_box = QComboBox()
        th = engine.calibrated_thresholds()
        self.mode_box.addItem(L(f'Muvozanatli (chegara {num(th["best_accuracy"])})',
                                f'Сбалансированный (порог {num(th["best_accuracy"])})'), th['best_accuracy'])
        self.mode_box.addItem(L(f'Qat’iy, FAR≈0,1% (chegara {num(th["far_1e-3"])})',
                                f'Строгий, FAR≈0,1% (порог {num(th["far_1e-3"])})'), th['far_1e-3'])
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        mode.addWidget(self.mode_box, 1)
        lay.addLayout(mode)
        right.addWidget(c)

        c, lay = card(L('Geometrik tahlil (asosiy yuz)', 'Геометрический анализ (главное лицо)'))
        grid = QGridLayout()
        self.geo_labels = {}
        rows = [('yaw', L('Burilish (yaw)', 'Поворот (yaw)')), ('pitch', L('Egilish (pitch)', 'Наклон (pitch)')),
                ('roll', L('Qiyshayish (roll)', 'Крен (roll)')), ('sharp', L('Tiniqlik', 'Резкость')),
                ('norm', L('Sifat (vektor normasi)', 'Качество (норма вектора)')), ('faces', L('Yuzlar kadrda', 'Лиц в кадре'))]
        for i, (k, t) in enumerate(rows):
            grid.addWidget(label(t, muted=True, wrap=False), i, 0)
            v = label('—', bold=True, wrap=False)
            grid.addWidget(v, i, 1, alignment=Qt.AlignRight)
            self.geo_labels[k] = v
        lay.addLayout(grid)
        right.addWidget(c)

        c, lay = card(L('Tezlik (bitta kadr)', 'Скорость (один кадр)'))
        self.fps_label = label('—', size=22, bold=True, wrap=False)
        self.fps_label.setMinimumHeight(34)
        lay.addWidget(self.fps_label)
        self.stage_bars = {}
        for k, t, col in (('detect', L('478 nuqtani topish', 'Поиск 478 точек'), theme.ACCENT),
                          ('geometry', L('Geometrik modul', 'Геометрический модуль'), theme.ACCENT),
                          ('appearance', L('Neyron tarmoq (tashqi ko‘rinish)', 'Нейросеть (внешность)'), theme.ACCENT2),
                          ('match', L('Birlashtirish + 1:N qidiruv', 'Слияние + поиск 1:N'), theme.FUSION)):
            row = QHBoxLayout()
            row.addWidget(label(t, muted=True, wrap=False), 1)
            v = label('—', wrap=False)
            v.setMinimumWidth(70)
            v.setAlignment(Qt.AlignRight)
            row.addWidget(v)
            lay.addLayout(row)
            bar = QProgressBar()
            bar.setRange(0, 400)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            bar.setStyleSheet(f'QProgressBar::chunk{{background:{col}}}')
            lay.addWidget(bar)
            self.stage_bars[k] = (v, bar)
        right.addWidget(c)

        c, lay = card(L('Tezkor qo‘shish', 'Быстрое добавление'),
                      L('Kadrdagi eng katta yuz qo‘shiladi. To‘liq kartochka bilan — «Ro‘yxatga olish» bo‘limida. Rasm saqlanmaydi.',
                        'Добавляется самое крупное лицо в кадре. С полной карточкой — в разделе «Регистрация». Фото не сохраняется.'))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(L('Ism Familiya', 'Имя Фамилия'))
        lay.addWidget(self.name_edit)
        row = QHBoxLayout()
        row.addWidget(button(L('Qo‘shish (1 kadr)', 'Добавить (1 кадр)'), 'primary', self.enroll_one))
        row.addWidget(button(L('5 kadr (boshni burang)', '5 кадров (повороты)'),
                             slot=self.enroll_burst))
        lay.addLayout(row)
        self.enroll_status = label('', muted=True)
        lay.addWidget(self.enroll_status)
        right.addWidget(c)
        right.addStretch(1)

        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(400)
        root.addWidget(rw)

        self.burst_timer = QTimer(self)
        self.burst_timer.setInterval(600)
        self.burst_timer.timeout.connect(self._burst_step)
        self._mode_changed()

    # ------------------------------------------------------------ control

    def _mode_changed(self):
        self.engine.threshold = self.mode_box.currentData()

    def toggle(self):
        if self.thread:
            self.stop()
            return
        src = self.source_box.currentData()
        if src == 'file':
            path, _ = QFileDialog.getOpenFileName(self, L('Video fayl', 'Видеофайл'), str(Path.home()),
                                                  'Video (*.mp4 *.mov *.avi *.mkv *.m4v)')
            if not path:
                return
            src = path
        self.thread = RecognitionThread(self.engine, src)
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.failed.connect(self._on_fail)
        self.thread.start()
        self.start_btn.setText(L('■  To‘xtatish', '■  Стоп'))

    def stop(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait(2000)
            self.thread = None
        self.burst_timer.stop()
        self.start_btn.setText(L('▶  Boshlash', '▶  Старт'))

    def _on_fail(self, why):
        self.stop()
        QMessageBox.warning(self, 'GFRAM', L(
            'Kamerani ochib bo‘lmadi.\n\nmacOS: Tizim sozlamalari → Maxfiylik va xavfsizlik → Kamera → '
            'Terminal (yoki dasturni ishga tushirgan ilova) uchun ruxsat bering va dasturni qayta oching.\n\n'
            'Boshqa kamera raqamini yoki «Video fayl» manbasini ham tanlash mumkin.',
            'Не удалось открыть камеру.\n\nmacOS: Системные настройки → Конфиденциальность и безопасность → Камера → '
            'разрешите доступ для Terminal (или приложения, из которого запущена программа) и перезапустите программу.\n\n'
            'Можно выбрать другой номер камеры или источник «Видеофайл».'))

    # ------------------------------------------------------------ frames

    def _on_frame(self, frame, faces, info):
        self.last_frame, self.last_faces = frame, faces
        thr = self.engine.current_threshold()
        overlays = []
        for i, f in enumerate(faces):
            p = f.get('p_smooth', 0.0)
            known = f.get('name') is not None and p >= thr
            name = f['name'] if known else L('Noma’lum', 'Неизвестный')
            color = theme.GOOD if known else (theme.WARN if f.get('name') is None else theme.BAD)
            overlays.append({'landmarks': f['landmarks'], 'bbox': f['bbox'], 'pose': f['pose'],
                             'color': color, 'label': f'{name}  {p * 100:.1f}%' if f.get('name') else name})
        self.view.set_image(frame, overlays)
        self._journal(faces, thr)

        timing = info.get('timing', {})
        self.fps_label.setText(f'{info.get("fps", 0)} FPS · {timing.get("total", 0):.0f} ms')
        for k, (v, bar) in self.stage_bars.items():
            ms = timing.get(k, 0.0)
            v.setText(f'{ms:.1f} ms')
            bar.setValue(int(min(400, ms * 10)))

        self.geo_labels['faces'].setText(str(len(faces)))
        if not faces:
            self.result_name.setText(L('Yuz topilmadi', 'Лицо не найдено'))
            self.result_name.setStyleSheet(f'font-size:26px;font-weight:700;color:{theme.MUTED}')
            self.result_prob.setText('')
            self.prob_bar.setValue(0)
            self.candidates.setText('')
            return
        f = faces[0]
        yaw, pitch, roll = f['pose']
        self.geo_labels['yaw'].setText(f'{yaw:+.1f}°')
        self.geo_labels['pitch'].setText(f'{pitch:+.1f}°')
        self.geo_labels['roll'].setText(f'{roll:+.1f}°')
        self.geo_labels['sharp'].setText(num(f['sharp'], 2))
        self.geo_labels['norm'].setText(num(f['app_norm'], 1))
        p = f.get('p_smooth', 0.0)
        known = f.get('name') is not None and p >= thr
        if f.get('name') is None:
            self.result_name.setText(L('Baza bo‘sh', 'База пуста'))
            col = theme.WARN
        else:
            self.result_name.setText(f['name'] if known else L('Noma’lum shaxs', 'Неизвестный'))
            col = theme.GOOD if known else theme.BAD
        self.result_name.setStyleSheet(f'font-size:30px;font-weight:700;color:{col}')
        self.result_prob.setText(L(f'Bir shaxs ekanligi ehtimoli: {pct(p, 1)} · chegara {num(thr)}',
                                   f'Вероятность совпадения: {pct(p, 1)} · порог {num(thr)}'))
        self.prob_bar.setValue(int(p * 1000))
        self.prob_bar.setStyleSheet(f'QProgressBar::chunk{{background:{col}}}')
        m = f['match']
        if m['candidates']:
            lines = [f'{i + 1}. {n} — {pct(q, 1)}' for i, (n, q) in enumerate(m['candidates'])]
            self.candidates.setText(L(f'Bazada {m["faces"]} ta shablon, qidiruv {m["time_ms"]:.1f} ms:\n',
                                      f'В базе {m["faces"]} шаблонов, поиск {m["time_ms"]:.1f} мс:\n') + '\n'.join(lines))
        else:
            self.candidates.setText(L('Avval bazaga odam qo‘shing (pastda).', 'Сначала добавьте человека в базу (ниже).'))

    def _journal(self, faces, thr):
        """One journal entry per person (or per unknown face) at most every 10 s."""
        now = time.time()
        for f in faces:
            if f.get('name') is None:
                continue
            p = f.get('p_smooth', 0.0)
            known = p >= thr
            key = f['name'] if known else '?unknown'
            if now - self.last_logged.get(key, 0) > 10:
                self.last_logged[key] = now
                self.engine.log_event(f['name'], p, known, 'camera')

    # ------------------------------------------------------------ enrolment

    def _name(self) -> Optional[str]:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.information(self, 'GFRAM', L('Ismni kiriting.', 'Введите имя.'))
            return None
        if not self.thread:
            QMessageBox.information(self, 'GFRAM', L('Avval kamerani yoqing.', 'Сначала включите камеру.'))
            return None
        return name

    def enroll_one(self):
        name = self._name()
        if name and self._enroll_current(name):
            self.on_db_changed()

    def _enroll_current(self, name) -> bool:
        if not self.last_faces:
            self.enroll_status.setText(L('Kadrda yuz yo‘q.', 'В кадре нет лица.'))
            return False
        n = self.engine.enroll(name, self.last_faces[0])
        self.enroll_status.setText(L(f'✅ {name}: bazada {n} ta shablon', f'✅ {name}: в базе {n} шаблонов'))
        return True

    def enroll_burst(self):
        name = self._name()
        if name:
            self.burst_name = name
            self.burst_left = 5
            self.burst_timer.start()

    def _burst_step(self):
        if self.burst_left <= 0 or not self.thread:
            self.burst_timer.stop()
            self.on_db_changed()
            return
        if self._enroll_current(self.burst_name):
            self.burst_left -= 1
            self.enroll_status.setText(self.enroll_status.text() + L(f' · yana {self.burst_left}', f' · ещё {self.burst_left}'))

    def save_frame(self):
        if self.last_frame is None:
            return
        out = Path.home() / 'Desktop' / f'gfram_kadr_{time.strftime("%Y%m%d_%H%M%S")}.png'
        pix = self.view.grab()
        pix.save(str(out))
        self.enroll_status.setText(L(f'Kadr saqlandi: {out.name}', f'Кадр сохранён: {out.name}'))

    def leave(self):
        self.stop()
