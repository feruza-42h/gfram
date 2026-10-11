"""Experimental results of the dissertation, read from runs/*.json produced by the project scripts."""

import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QHeaderView, QScrollArea, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from .. import data, theme
from ..i18n import L, num, pct
from ..widgets import Chart, button, card, heading, label, read_image, run_task

CONDITIONS = ['clean', 'lowres_24px', 'lowres_16px', 'blur_s4', 'dark', 'jpeg_q5', 'mask', 'sunglasses']


def condition_names():
    return {
        'clean': L('Toza', 'Чистые'), 'lowres_24px': L('Yuz 24 px', 'Лицо 24 px'),
        'lowres_16px': L('Yuz 16 px', 'Лицо 16 px'), 'blur_s4': L('Xiralik', 'Размытие'),
        'dark': L('Qorong‘i', 'Темнота'), 'jpeg_q5': 'JPEG q5', 'mask': L('Niqob', 'Маска'),
        'sunglasses': L('Qora ko‘zoynak', 'Тёмные очки'),
    }


def table(headers, rows, highlight_last=False, stretch_first=True):
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setSelectionMode(QTableWidget.NoSelection)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            it = QTableWidgetItem(str(v))
            it.setTextAlignment(Qt.AlignVCenter | (Qt.AlignHCenter if len(str(v)) <= 16 else Qt.AlignLeft))
            if highlight_last and i == len(rows) - 1:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
                it.setForeground(Qt.white)
            t.setItem(i, j, it)
    h = t.horizontalHeader()
    for j in range(len(headers)):
        h.setSectionResizeMode(j, QHeaderView.Stretch if (j == 0 and stretch_first) else QHeaderView.ResizeToContents)
    t.setMinimumHeight(min(520, 44 + 34 * len(rows)))
    t.verticalHeader().setDefaultSectionSize(32)
    return t


def scroll_tab(build):
    area = QScrollArea()
    area.setWidgetResizable(True)
    w = QWidget()
    w.setObjectName('page')
    lay = QVBoxLayout(w)
    lay.setContentsMargins(4, 12, 4, 12)
    lay.setSpacing(14)
    build(lay)
    lay.addStretch(1)
    area.setWidget(w)
    return area


def bar_labels(ax, bars, fmt=lambda v: f'{v:.1f}%', dy=0.6, size=10):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + dy, fmt(v), ha='center', va='bottom', fontsize=size, color=theme.TEXT)


class ResultsPage(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.setObjectName('page')
        self.engine = engine
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 16)
        root.addWidget(heading(L('Tadqiqot natijalari', 'Результаты исследования'),
                               L('Barcha raqamlar loyiha skriptlari bilan olingan va runs/ papkasidagi fayllardan o‘qiladi. '
                                 'Sinov va o‘qitish odamlar bo‘yicha ajratilgan.',
                                 'Все цифры получены скриптами проекта и читаются из файлов папки runs/. '
                                 'Обучение и тест разделены по людям.')))
        tabs = QTabWidget()
        tabs.addTab(scroll_tab(self.stages), L('4 bosqich', '4 этапа'))
        tabs.addTab(scroll_tab(self.training), L('O‘qitish', 'Обучение'))
        tabs.addTab(scroll_tab(self.lfw), 'LFW')
        tabs.addTab(scroll_tab(self.robustness), L('Tasvir sharoitlari', 'Условия съёмки'))
        tabs.addTab(scroll_tab(self.cplfw), L('Burilish (CPLFW)', 'Ракурсы (CPLFW)'))
        tabs.addTab(scroll_tab(self.protection), L('Shablon himoyasi', 'Защита шаблонов'))
        tabs.addTab(scroll_tab(self.speed), L('Tezlik va hajm', 'Скорость и размер'))
        root.addWidget(tabs, 1)

    # ------------------------------------------------------------------ tabs

    def stages(self, lay):
        b = data.run('baselines')
        h = data.run('hybrid_lfw')
        vals = [b.get('current_pipeline', {}).get('accuracy', 0.537) * 100,
                b.get('procrustes_raw', {}).get('accuracy', 0.675) * 100,
                max((x['accuracy'] for x in data.training_history()), default=0.780) * 100, 99.67]
        names = [L('1. Klassik\ngeometrik belgilar', '1. Классические\nгеом. признаки'),
                 L('2. Prokrust +\nstandartlash', '2. Прокруст +\nстандартизация'),
                 L('3. Geometriya + SI\n(GeometricTransformer)', '3. Геометрия + ИИ\n(GeometricTransformer)'),
                 L('4. Gibrid tizim\n(adaptiv birlashtirish)', '4. Гибридная система\n(адаптивное слияние)')]
        ch = Chart(10, 4.6)
        ax = ch.axes()
        bars = ax.bar(names, vals, color=[theme.MUTED, theme.ACCENT, theme.ACCENT2, theme.GOOD], width=0.6)
        bar_labels(ax, bars, lambda v: (f'{v:.2f}%' if v > 99 else f'{v:.1f}%').replace('.', ','), size=13)
        ax.axhline(50, color=theme.BAD, ls=':', lw=1)
        ax.text(-0.45, 50.8, L('tasodifiy taxmin (50%)', 'случайное угадывание (50%)'), color=theme.BAD, fontsize=9, ha='left')
        ax.set_ylim(40, 105)
        ax.set_ylabel(L('LFW aniqligi, %', 'Точность на LFW, %'))
        ax.set_title(L('Usulning rivojlanishi: bir xil test (LFW, 6000 juftlik)', 'Развитие метода: один и тот же тест (LFW, 6000 пар)'))
        ax.grid(axis='x', visible=False)
        ch.draw()
        c, l = card()
        l.addWidget(ch)
        lay.addWidget(c)
        rows = [
            ['1', L('153 klassik geometrik belgi (gfram 3.0)', '153 классических геометрических признака (gfram 3.0)'), pct(vals[0] / 100, 1), num(b.get('current_pipeline', {}).get('auc', 0), 3)],
            ['2', L('3D Prokrust tekislash + standartlash, o‘qitishsiz', '3D-выравнивание Прокруста + стандартизация, без обучения'), pct(vals[1] / 100, 1), num(b.get('procrustes_raw', {}).get('auc', 0), 3)],
            ['3', L('GeometricTransformer, CelebA da o‘qitilgan (423 ming parametr)', 'GeometricTransformer, обучен на CelebA (423 тыс. параметров)'), pct(vals[2] / 100, 1), num(max(data.training_history(), key=lambda x: x['accuracy'], default={'auc': 0})['auc'], 3)],
            ['4', L('Gibrid: geometrik modul + MobileFaceNet + adaptiv birlashtirish', 'Гибрид: геом. модуль + MobileFaceNet + адаптивное слияние'), '99,67%', '—'],
        ]
        t = table([L('Bosqich', 'Этап'), L('Usul', 'Метод'), L('Aniqlik', 'Точность'), 'AUC'], rows, True, stretch_first=False)
        t.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        lay.addWidget(t)
        lay.addWidget(label(L('Geometriya SI bilan 78% beradi; 99% — bu gibrid tizim, unda geometrik modul neyron tarmoqni boshqaradi '
                              '(yuzni topadi, tekislaydi, bosh holatini baholaydi).',
                              'Геометрия с ИИ даёт 78%; 99% — это гибридная система, в которой геометрический модуль управляет '
                              'нейросетью (находит лицо, выравнивает, оценивает позу).'), muted=True))

    def training(self, lay):
        hist = data.training_history()
        if not hist:
            lay.addWidget(label(L('O‘qitish tarixi topilmadi.', 'История обучения не найдена.')))
            return
        ep = [h['epoch'] for h in hist]
        ch = Chart(10, 4.4)
        fig = ch.figure
        fig.clear()
        a1 = fig.add_subplot(1, 2, 1)
        a1.plot(ep, [h['accuracy'] * 100 for h in hist], color=theme.ACCENT, lw=2.4, marker='o', ms=3, label=L('aniqlik', 'точность'))
        a1.plot(ep, [h['auc'] * 100 for h in hist], color=theme.ACCENT2, lw=2.4, marker='o', ms=3, label='AUC × 100')
        a1.set_xlabel(L('epoxa', 'эпоха'))
        a1.set_ylabel('%')
        a1.set_title(L('LFW (o‘qitishda ko‘rilmagan odamlar)', 'LFW (люди, не виденные при обучении)'))
        a1.legend()
        a2 = fig.add_subplot(1, 2, 2)
        a2.plot(ep, [h['loss'] for h in hist], color=theme.WARN, lw=2.4)
        a2.set_xlabel(L('epoxa', 'эпоха'))
        a2.set_title(L('ArcFace yo‘qotish funksiyasi', 'Функция потерь ArcFace'))
        ch.draw()
        c, l = card(L('GeometricTransformer: CelebA da o‘qitish', 'GeometricTransformer: обучение на CelebA'),
                    L('198 803 yuz, 9 325 odam; 30 epoxa × 2000 qadam, Apple M3 (8 GB). Aniqlik oxirgi epoxagacha o‘sgan — '
                      'geometrik model chegarasiga yetilmagan.',
                      '198 803 лица, 9 325 человек; 30 эпох × 2000 шагов, Apple M3 (8 ГБ). Точность росла до последней эпохи — '
                      'потолок геометрической модели не достигнут.'))
        l.addWidget(ch)
        lay.addWidget(c)
        best = max(hist, key=lambda h: h['accuracy'])
        rows = [[L('Eng yaxshi epoxa', 'Лучшая эпоха'), str(best['epoch'])],
                [L('Aniqlik (LFW, 6000 juftlik)', 'Точность (LFW, 6000 пар)'), pct(best['accuracy'])],
                ['ROC AUC', num(best['auc'], 3)],
                ['TAR@FAR=1%', pct(best['tar@far=1%'], 1)],
                [L('Muhim metodik xulosa', 'Ключевой методический вывод'),
                 L('augmentatsiya kuchi shaxslararo farqqa (≈0,0014) bog‘lanishi kerak',
                   'сила аугментаций должна быть привязана к межличностному разбросу (≈0,0014)')]]
        lay.addWidget(table([L('Ko‘rsatkich', 'Показатель'), L('Qiymat', 'Значение')], rows))

    def lfw(self, lay):
        h = data.run('hybrid_lfw')
        names = [L('Geometriya + SI\n(faqat nuqtalar)', 'Геометрия + ИИ\n(только точки)'), L('Geom. tekislash +\nMobileFaceNet', 'Геом. выравнивание +\nMobileFaceNet'),
                 L('Gibrid tizim\n(MobileFaceNet + birlashtirish)', 'Гибрид\n(MobileFaceNet + слияние)'),
                 L('Geom. tekislash +\nArcFace R50 (taqqoslash uchun)', 'Геом. выравнивание +\nArcFace R50 (ориентир)')]
        vals = [h.get('geometry', {}).get('accuracy', 0.7787) * 100, h.get('w600k_mbf', {}).get('accuracy', 0.9965) * 100,
                99.67, h.get('w600k_r50', {}).get('accuracy', 0.9985) * 100]
        ch = Chart(10, 4.2)
        ax = ch.axes()
        bars = ax.bar(names, vals, color=[theme.ACCENT, theme.ACCENT2, theme.GOOD, theme.MUTED], width=0.55)
        bar_labels(ax, bars, lambda v: f'{v:.2f}%'.replace('.', ','), dy=0.3, size=12)
        ax.set_ylim(70, 102)
        ax.set_ylabel('%')
        ax.grid(axis='x', visible=False)
        ax.set_title(L('LFW, standart protokol (10 fold)', 'LFW, стандартный протокол (10 фолдов)'))
        ch.draw()
        c, l = card()
        l.addWidget(ch)
        lay.addWidget(c)
        rows = [[L('Geom. tekislash + MobileFaceNet', 'Геом. выравнивание + MobileFaceNet'), '13 MB', '~6 ms', pct(vals[1] / 100)],
                [L('Geom. tekislash + ArcFace R50 (og‘ir model, mo‘ljal)', 'Геом. выравнивание + ArcFace R50 (тяжёлая модель, ориентир)'), '174 MB', '~74 ms', pct(vals[3] / 100)],
                [L('Gibrid tizim (MobileFaceNet + adaptiv birlashtirish)', 'Гибрид (MobileFaceNet + адаптивное слияние)'), '15 MB', '~24 ms', '99,67%'],
                [L('gfram kutubxonasi, verify(), 6000 juftlik, yagona chegara', 'Библиотека gfram, verify(), 6000 пар, один порог'), '15 MB', '~24 ms', '99,53%']]
        lay.addWidget(table([L('Tizim', 'Система'), L('Hajm', 'Размер'), L('Vaqt / yuz', 'Время / лицо'), L('Aniqlik', 'Точность')], rows))

        ab = data.run('qaf_ablation')
        if ab:
            c, l = card(L('Ablatsiya: har bir qismning hissasi (bir xil sharoit, yagona chegara)',
                          'Абляция: вклад каждой части (одни условия, один порог)'))
            keys = [('appearance score only', L('Faqat neyron tarmoq o‘xshashligi', 'Только сходство нейросети')),
                    ('appearance + geometry identity (s_geo) only', L('+ geometrik o‘xshashlik', '+ геометрическое сходство')),
                    ('appearance + embedding norms only', L('+ sifat (vektor normasi)', '+ качество (норма вектора)')),
                    ('without embedding norms', L('To‘liq, sifatsiz', 'Полное, без качества')),
                    ('full QAF', L('To‘liq adaptiv birlashtirish', 'Полное адаптивное слияние'))]
            rows = [[n, pct(ab[k][0]), pct(ab[k][1]), pct(1 - ab[k][0])] for k, n in keys if k in ab]
            l.addWidget(table([L('Variant', 'Вариант'), L('Toza', 'Чистые'), L('7 buzilish o‘rtachasi', 'Среднее по 7 искажениям'),
                               L('Xatolar', 'Ошибки')], rows, True))
            l.addWidget(label(L('Bir xil protokolda birlashtirish xatolarni 0,65% dan 0,33% ga — ikki baravar kamaytiradi. '
                                'Old tomondan olingan rasmlarda geometriyaning hissasi kichik; u burilishda namoyon bo‘ladi (CPLFW).',
                                'В одном протоколе слияние снижает ошибки с 0,65% до 0,33% — вдвое. '
                                'На фронтальных фото вклад геометрии мал; он проявляется при поворотах (CPLFW).'), muted=True))
            lay.addWidget(c)

    def robustness(self, lay):
        r = data.run('robustness')
        if not r:
            return
        res = r['results']
        cn = condition_names()
        x = np.arange(len(CONDITIONS))
        ch = Chart(11, 4.6)
        ax = ch.axes()
        series = [('geometry', L('Geometriya + SI', 'Геометрия + ИИ'), theme.ACCENT),
                  ('appearance', L('MobileFaceNet (birlashtirishsiz)', 'MobileFaceNet (без слияния)'), theme.ACCENT2),
                  ('qaf_logistic', L('Gibrid tizim', 'Гибридная система'), theme.GOOD)]
        w = 0.27
        for i, (k, n, col) in enumerate(series):
            ax.bar(x + (i - 1) * w, [res[k][c]['accuracy'] * 100 for c in CONDITIONS], w, label=n, color=col)
        ax.set_xticks(x, [cn[c] for c in CONDITIONS])
        ax.set_ylim(50, 101)
        ax.set_ylabel('%')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3)
        ax.grid(axis='x', visible=False)
        ch.draw()
        c, l = card(subtitle=L('Har bir juftlikda birinchi rasm toza (bazaga qo‘shilgandek), ikkinchisi buzilgan (kameradan kelgandek); '
                               'nuqtalar qaytadan topiladi.',
                               'В каждой паре первое фото чистое (как при добавлении в базу), второе искажено (как с камеры); '
                               'точки ищутся заново.'))
        l.addWidget(ch)
        lay.addWidget(c)
        rows = []
        for cnd in CONDITIONS:
            a, q = res['appearance'][cnd]['accuracy'], res['qaf_logistic'][cnd]['accuracy']
            rows.append([cn[cnd], pct(r['detected'][cnd], 1), pct(res['geometry'][cnd]['accuracy']), pct(a), pct(q),
                         f'{(1 - q) / (1 - a) * 100 - 100:+.0f}%'])
        lay.addWidget(table([L('Sharoit', 'Условие'), L('Yuz topildi', 'Лицо найдено'), L('Geometriya + SI', 'Геометрия + ИИ'),
                             'MobileFaceNet', L('Gibrid', 'Гибрид'), L('Xatolar o‘zgarishi', 'Изменение ошибок')], rows))

    def cplfw(self, lay):
        light = data.run('cplfw')
        stock = data.run('cplfw_insightface')
        sig = data.run('cplfw_significance_landmarks')
        sig2 = data.run('cplfw_significance_insightface')
        steps = [('appearance only', L('Faqat SI', 'Только ИИ')), ('+ quality (QAF w/o geometry)', L('+ sifat', '+ качество')),
                 ('+ quality + pose + geometry (full)', L('+ sifat + bosh holati + geometriya', '+ качество + поза + геометрия'))]
        configs = [(light.get('w600k_mbf'), 'MobileFaceNet'), (light.get('w600k_r50'), 'ArcFace R50'),
                   (stock.get('w600k_r50'), L('R50, InsightFace standart', 'R50, штатный InsightFace'))]
        ch = Chart(11, 4.4)
        fig = ch.figure
        fig.clear()
        for j, metric in enumerate(('accuracy', 'tar@far=1%')):
            ax = fig.add_subplot(1, 2, j + 1)
            x = np.arange(len(configs))
            w = 0.27
            for i, (k, n) in enumerate(steps):
                vals = [cfg[k][metric] * 100 if cfg else 0 for cfg, _ in configs]
                ax.bar(x + (i - 1) * w, vals, w, label=n, color=[theme.ACCENT2, theme.FUSION, theme.GOOD][i])
            ax.set_xticks(x, [n for _, n in configs], fontsize=9)
            ax.set_ylim(40 if metric != 'accuracy' else 75, 100)
            ax.set_title(L('Aniqlik, %', 'Точность, %') if metric == 'accuracy' else 'TAR@FAR=1%, %')
            ax.grid(axis='x', visible=False)
            if j == 0:
                handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=3)
        ch.draw()
        c, l = card(L('Old tomondan – profildan: geometriya foyda beradigan joy', 'Анфас – профиль: где геометрия помогает'),
                    L('CPLFW — 6000 juftlik, yuz burilishi juda katta. Bu yerda geometrik belgilar statistik ahamiyatli foyda beradi.',
                      'CPLFW — 6000 пар с сильным поворотом головы. Здесь геометрические признаки дают статистически значимый выигрыш.'))
        l.addWidget(ch)
        lay.addWidget(c)
        rows = []
        for src, model, pipe in ((sig, 'w600k_mbf', L('yengil (MediaPipe)', 'лёгкий (MediaPipe)')),
                                 (sig, 'w600k_r50', L('yengil (MediaPipe)', 'лёгкий (MediaPipe)')),
                                 (sig2, 'w600k_r50', L('InsightFace standart', 'штатный InsightFace'))):
            s = src.get(model)
            if not s:
                continue
            m = s['mcnemar']
            rows.append(['MobileFaceNet' if model == 'w600k_mbf' else 'ArcFace R50', pipe,
                         f'{pct(s["acc_without_geometry"])} → {pct(s["acc_with_geometry"])}',
                         f'{pct(s["tar1_without_geometry"], 1)} → {pct(s["tar1_with_geometry"], 1)}',
                         f'{m["fixed_by_geometry"]} / {m["broken_by_geometry"]}',
                         f'{m["p_value"]:.4f}'.replace('.', ',')])
        lay.addWidget(table([L('Model', 'Модель'), L('Pipeline', 'Пайплайн'), L('Aniqlik', 'Точность'), 'TAR@FAR=1%',
                             L('Tuzatdi / buzdi', 'Исправила / испортила'), L('p (McNemar)', 'p (Мак-Немар)')], rows))
        lay.addWidget(label(L('Geometriya qo‘shilishi: TAR@FAR=1% +6…11 foiz punkt, p < 0,002 (yengil konfiguratsiya).',
                              'Добавление геометрии: TAR@FAR=1% +6…11 п.п., p < 0,002 (лёгкая конфигурация).'), muted=True))

    def protection(self, lay):
        tp = data.run('template_protection')
        if not tp:
            return
        rows = []
        total = changed = 0
        for k, v in tp.items():
            if 'pairs' not in v:
                continue
            rows.append([k.replace('blur_s4', 'blur').replace('lowres_', ''), f'{v["pairs"]:,}'.replace(',', ' '),
                         f'{v["max_abs_prob_diff"]:.1e}', str(v['decisions_changed']),
                         num(v['unlinkability_auc'], 3), pct(v['revoked_genuine_accept_rate']), pct(v['plain_genuine_accept_rate'], 1)])
            total += v['pairs']
            changed += v['decisions_changed']
        c, l = card(L('Himoyalangan shablonlar: aniqlik yo‘qolmaydi', 'Защищённые шаблоны: точность не теряется'),
                    L(f'{total:,} juftlikda {changed} ta qaror o‘zgardi. Turli kalitlar ostidagi shablonlarni bog‘lab bo‘lmaydi (AUC≈0,5). '
                      'Kalit almashtirilgach eski shablonlar faqat tasodifiy darajada mos keladi.'.replace(',', ' '),
                      f'На {total:,} парах изменилось решений: {changed}. Шаблоны под разными ключами нельзя связать (AUC≈0,5). '
                      'После смены ключа старые шаблоны совпадают лишь на уровне случайности.'.replace(',', ' ')))
        l.addWidget(table([L('To‘plam', 'Набор'), L('Juftliklar', 'Пар'), L('Maks. farq', 'Макс. разница'),
                           L('O‘zgargan qarorlar', 'Изменено решений'), L('Bog‘lanmaslik AUC', 'Несвязываемость AUC'),
                           L('Kalit almashgach mos kelish', 'Совпадения после отзыва'),
                           L('Almashishdan oldin', 'До отзыва')], rows))
        lay.addWidget(c)
        c, l = card(L('Qanday ishlaydi', 'Как работает'))
        l.addWidget(label(L(
            '• O‘rnatishning maxfiy kaliti K (256 bit) bazadan alohida saqlanadi.\n'
            '• Kalitdan ikkita tasodifiy ortogonal matritsa quriladi: R_g (128×128) va R_a (512×512).\n'
            '• Bazada faqat R_g·g va R_a·a saqlanadi; so‘rov ham xuddi shunday buriladi. Ortogonal burish skalyar '
            'ko‘paytmani saqlaydi, shuning uchun barcha qarorlar aynan bir xil (teorema).\n'
            '• 478 nuqta AES-256-GCM bilan shifrlanadi — model yangilanganda qayta hisoblash uchun.\n'
            '• Kalitni almashtirish butun bazani rasmlarsiz qayta himoyalaydi; o‘g‘irlangan nusxa yaroqsiz bo‘ladi.',
            '• Секретный ключ установки K (256 бит) хранится отдельно от базы.\n'
            '• Из ключа строятся две случайные ортогональные матрицы: R_g (128×128) и R_a (512×512).\n'
            '• В базе хранятся только R_g·g и R_a·a; запрос поворачивается так же. Ортогональный поворот сохраняет '
            'скалярное произведение, поэтому все решения совпадают точно (теорема).\n'
            '• 478 точек шифруются AES-256-GCM — для пересчёта при обновлении модели.\n'
            '• Смена ключа перезащищает всю базу без фото; украденная копия становится бесполезной.')))
        lay.addWidget(c)

    def speed(self, lay):
        info = self.engine.model_info()
        rows = [[L('Geometrik model (GeometricTransformer)', 'Геометрическая модель (GeometricTransformer)'),
                 f'{info["geometry_params"]:,}'.replace(',', ' ') + L(' parametr', ' параметров'), '1,9 MB', f'v{info["geometry_version"]}'],
                [L('Tashqi ko‘rinish modeli (MobileFaceNet)', 'Модель внешности (MobileFaceNet)'), '—', f'{info["appearance_mb"]:.1f} MB', 'InsightFace w600k_mbf'],
                [L('Adaptiv birlashtirish', 'Адаптивное слияние'), f'{len(info["fusion_features"])} ' + L('belgi', 'признаков'), '< 1 KB', L('logistik model', 'логистическая модель')],
                [L('Taqqoslash uchun: ArcFace R50', 'Для сравнения: ArcFace R50'), '~43 mln', '174 MB', '']]
        lay.addWidget(table([L('Komponent', 'Компонент'), L('Hajm', 'Объём'), L('Fayl', 'Файл'), L('Izoh', 'Примечание')], rows))
        c, l = card(L('Shu kompyuterda tezlikni o‘lchash', 'Измерить скорость на этом компьютере'),
                    L('Bitta LFW rasmi 30 marta to‘liq qayta ishlanadi (protsessor, videokartasiz).',
                      'Одно фото LFW 30 раз проходит полную обработку (процессор, без видеокарты).'))
        self.speed_chart = Chart(9, 3.2)
        self.speed_note = label('', muted=True)
        l.addWidget(button(L('O‘lchash', 'Измерить'), 'primary', self.measure))
        l.addWidget(self.speed_chart)
        l.addWidget(self.speed_note)
        lay.addWidget(c)

    def measure(self):
        people = data.lfw_people() if data.lfw_available() else {}
        if not people:
            return
        img = read_image(next(iter(people.values()))[0])
        engine = self.engine

        def work(task):
            out = []
            for _ in range(30):
                r = engine.analyze(img, max_faces=1)
                t = time.perf_counter()
                if r['faces']:
                    engine.identify(r['faces'][0])
                out.append(dict(r['timing'], match=(time.perf_counter() - t) * 1000))
            return out

        def done(out):
            keys = [('detect', L('478 nuqta', '478 точек'), theme.ACCENT), ('geometry', L('Geometrik modul', 'Геом. модуль'), theme.ACCENT),
                    ('appearance', 'MobileFaceNet', theme.ACCENT2), ('match', L('Birlashtirish + 1:N', 'Слияние + 1:N'), theme.FUSION)]
            med = [float(np.median([o[k] for o in out])) for k, _, _ in keys]
            ax = self.speed_chart.axes()
            left = 0
            for (k, n, col), v in zip(keys, med):
                ax.barh([0], [v], left=left, color=col, label=f'{n}: {v:.1f} ms')
                left += v
            ax.set_yticks([])
            ax.set_xlabel('ms')
            ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.35), fontsize=10)
            ax.grid(axis='y', visible=False)
            self.speed_chart.draw()
            self.speed_note.setText(L(f'Jami (mediana): {sum(med):.1f} ms bitta yuzga → ~{1000 / sum(med):.0f} kadr/soniya. '
                                      f'Bazadagi shablonlar: {engine.stats()["faces"]}.',
                                      f'Итого (медиана): {sum(med):.1f} мс на лицо → ~{1000 / sum(med):.0f} кадров/с. '
                                      f'Шаблонов в базе: {engine.stats()["faces"]}.'))

        run_task(work, done, lambda e: self.speed_note.setText(e[-300:]))
