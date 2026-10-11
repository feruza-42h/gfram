"""Versions and updates: the library + desktop (PyPI) and the geometric model (gfram.uz)."""

import sys

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QPlainTextEdit, QScrollArea, QVBoxLayout,
                               QWidget)

from .. import theme, updates
from ..i18n import L
from ..widgets import StatTile, button, card, heading, label, run_task


class UpdatesPage(QWidget):
    def __init__(self, engine, on_update_found=lambda v: None):
        super().__init__()
        self.setObjectName('page')
        self.engine = engine
        self.on_update_found = on_update_found
        self.latest = None
        self.proc = None

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
        root.addWidget(heading(L('Versiya va yangilanishlar', 'Версия и обновления'),
                               L('Kutubxona va desktop dasturi bitta paket — gfram (PyPI). Uni yangilash ikkalasini ham yangilaydi. '
                                 'Geometrik model alohida — gfram.uz serveridan, odamlarni qayta ro‘yxatga olishsiz.',
                                 'Библиотека и десктоп — один пакет gfram (PyPI): его обновление обновляет оба. '
                                 'Геометрическая модель — отдельно, с сервера gfram.uz, без перерегистрации людей.')))
        info = updates.installed_info(engine)
        tiles = QHBoxLayout()
        self.t_cur = StatTile(info['version'], L('o‘rnatilgan versiya (gfram + Studio)', 'установленная версия (gfram + Studio)'), theme.ACCENT)
        self.t_new = StatTile('…', L('PyPI dagi so‘nggi versiya', 'последняя версия на PyPI'), theme.MUTED)
        self.t_model = StatTile(info['geometry_model'], L('geometrik model', 'геометрическая модель'), theme.ACCENT2)
        self.t_server = StatTile('…', L('gfram.uz serveridagi model', 'модель на сервере gfram.uz'), theme.MUTED)
        for t in (self.t_cur, self.t_new, self.t_model, self.t_server):
            tiles.addWidget(t)
        root.addLayout(tiles)

        c, lay = card(L('Kutubxona va desktop dasturini yangilash', 'Обновление библиотеки и десктопа'))
        self.status = label('', size=16, bold=True)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        row.addWidget(button(L('🔄  Tekshirish', '🔄  Проверить'), slot=self.check))
        self.update_btn = button(L('⬇  Yangilash', '⬇  Обновить'), 'primary', self.run_update)
        self.update_btn.setEnabled(False)
        row.addWidget(self.update_btn)
        self.restart_btn = button(L('↻  Qayta ishga tushirish', '↻  Перезапустить'), slot=self.restart)
        self.restart_btn.setVisible(False)
        row.addWidget(self.restart_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(170)
        self.log.setStyleSheet('font-family: Menlo, Consolas, monospace; font-size: 12px;')
        self.log.setPlaceholderText(L('Yangilash jurnali shu yerda ko‘rinadi', 'Здесь будет журнал обновления'))
        lay.addWidget(self.log)
        root.addWidget(c)

        c, lay = card(L('Geometrik model (gfram.uz)', 'Геометрическая модель (gfram.uz)'),
                      L('Kutubxona serverni kuniga bir marta o‘zi tekshiradi. Yangi model yuklansa, bazadagi geometrik shablonlar '
                        'saqlangan shifrlangan nuqtalardan avtomatik qayta hisoblanadi.',
                        'Библиотека сама проверяет сервер раз в сутки. Если загружена новая модель, геометрические шаблоны '
                        'базы автоматически пересчитываются из сохранённых зашифрованных точек.'))
        row = QHBoxLayout()
        row.addWidget(button(L('Modelni hozir yangilash', 'Обновить модель сейчас'), slot=self.update_model))
        row.addStretch(1)
        lay.addLayout(row)
        self.model_status = label('', muted=True)
        lay.addWidget(self.model_status)
        root.addWidget(c)

        c, lay = card(L('O‘rnatish ma’lumotlari', 'Сведения об установке'))
        grid = QGridLayout()
        rows = [(L('Versiya', 'Версия'), info['version']),
                (L('Geometrik model', 'Геометрическая модель'), info['geometry_model']),
                (L('Neyron tarmoq', 'Нейросеть'), info['appearance_model']),
                ('Python', info['python']),
                (L('Tizim', 'Система'), info['system']),
                (L('Rejim', 'Режим'), L('manba kod (ishlab chiquvchi)', 'исходный код (разработчик)') if info['source']
                 else L('o‘rnatilgan paket (pip)', 'установленный пакет (pip)')),
                (L('Kutubxona joyi', 'Расположение библиотеки'), info['location']),
                (L('Baza va kalit', 'База и ключ'), info['data'])]
        for i, (k, v) in enumerate(rows):
            grid.addWidget(label(k, muted=True, wrap=False), i, 0)
            grid.addWidget(label(v), i, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)
        root.addWidget(c)
        root.addStretch(1)
        self.info = info
        self.check()

    # ------------------------------------------------------------------

    def check(self):
        self.status.setText(L('PyPI tekshirilmoqda…', 'Проверка PyPI…'))
        self.status.setStyleSheet(f'font-size:16px;font-weight:700;color:{theme.MUTED}')

        def work(task):
            out = {}
            try:
                out['pypi'] = updates.latest_pypi()
            except Exception as e:
                out['pypi_error'] = str(e)
            out['server'] = updates.model_server_status()
            return out

        run_task(work, self._checked, lambda e: self.status.setText(e[-200:]))

    def _checked(self, r):
        if self.proc is not None or self.restart_btn.isVisible():
            return                      # an update is running or finished: keep its message
        cur = self.info['version']
        if 'pypi' in r:
            latest = r['pypi']['version']
            self.latest = latest
            newer = updates.is_newer(latest, cur)
            self.t_new.set(latest, theme.WARN if newer else theme.GOOD)
            if newer:
                self.status.setText(L(f'⬆ Yangi versiya mavjud: {latest} ({r["pypi"]["date"]}). Joriy: {cur}.',
                                      f'⬆ Доступна новая версия: {latest} ({r["pypi"]["date"]}). Текущая: {cur}.'))
                self.status.setStyleSheet(f'font-size:16px;font-weight:700;color:{theme.WARN}')
                self.on_update_found(latest)
            else:
                self.status.setText(L(f'✅ Eng so‘nggi versiya o‘rnatilgan ({cur}).', f'✅ Установлена последняя версия ({cur}).'))
                self.status.setStyleSheet(f'font-size:16px;font-weight:700;color:{theme.GOOD}')
            if self.info['source']:
                self.update_btn.setEnabled(False)
                self.update_btn.setToolTip(L('Manba koddan ishlayapti: yangilash git orqali.', 'Запуск из исходного кода: обновление через git.'))
                if newer:
                    self.status.setText(self.status.text() + L(' (manba kod rejimi — git pull)', ' (режим исходного кода — git pull)'))
            else:
                self.update_btn.setEnabled(newer)
        else:
            self.t_new.set('—', theme.MUTED)
            self.status.setText(L('Internet yoki PyPI mavjud emas — dastur oflayn ishlashda davom etadi.',
                                  'Нет интернета или PyPI недоступен — программа продолжает работать офлайн.'))
            self.status.setStyleSheet(f'font-size:16px;font-weight:700;color:{theme.MUTED}')
        s = r.get('server', {})
        if s.get('online') and s.get('version'):
            newer = updates.is_newer(s['version'], self.info['geometry_model'])
            self.t_server.set(s['version'], theme.WARN if newer else theme.GOOD)
            self.model_status.setText(L(f'Server ishlayapti. Serverdagi model: {s["version"]}, ishlatilayotgani: {self.info["geometry_model"]}.',
                                        f'Сервер работает. Модель на сервере: {s["version"]}, используется: {self.info["geometry_model"]}.'))
        else:
            self.t_server.set('—', theme.MUTED)
            self.model_status.setText(L('Server javob bermadi — o‘rnatilgan model ishlatiladi.',
                                        'Сервер не ответил — используется установленная модель.'))

    def run_update(self):
        if self.proc:
            return
        cmd = updates.pip_command()
        self.log.setPlainText('$ ' + ' '.join(cmd) + '\n')
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(
            lambda: self.log.appendPlainText(bytes(self.proc.readAllStandardOutput()).decode(errors='replace').rstrip()))
        self.proc.finished.connect(self._update_finished)
        self.update_btn.setEnabled(False)
        self.status.setText(L('⬇ Yuklab olinmoqda va o‘rnatilmoqda…', '⬇ Загрузка и установка…'))
        self.proc.start(cmd[0], cmd[1:])

    def _update_finished(self, code, _status):
        self.proc = None
        if code == 0:
            self.status.setText(L('✅ Yangilandi. Yangi versiyani ishga tushirish uchun dasturni qayta oching.',
                                  '✅ Обновлено. Перезапустите программу, чтобы запустить новую версию.'))
            self.status.setStyleSheet(f'font-size:16px;font-weight:700;color:{theme.GOOD}')
            self.restart_btn.setVisible(True)
        else:
            self.status.setText(L(f'❌ Yangilash bajarilmadi (kod {code}). Jurnalni ko‘ring.', f'❌ Обновление не выполнено (код {code}). См. журнал.'))
            self.status.setStyleSheet(f'font-size:16px;font-weight:700;color:{theme.BAD}')
            self.update_btn.setEnabled(True)

    def restart(self):
        QProcess.startDetached(sys.executable, ['-m', 'gfram.studio'])
        QApplication.quit()

    def update_model(self):
        self.model_status.setText(L('gfram.uz bilan bog‘lanilmoqda…', 'Связь с gfram.uz…'))

        def done(v):
            self.model_status.setText(L(f'Ishlatilayotgan model: {v or "—"}. O‘zgarish bo‘lsa, dastur qayta ochilganda qo‘llanadi.',
                                        f'Используемая модель: {v or "—"}. Если она изменилась, применится после перезапуска.'))

        run_task(lambda task: updates.update_model(), done, lambda e: self.model_status.setText(e[-200:]))
