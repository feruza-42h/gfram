"""Scientific novelty, defence positions and the software / data used."""

from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from .. import theme
from ..i18n import L
from ..widgets import card, heading, label


class AboutPage(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.setObjectName('page')
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
        root.addWidget(heading(L('Ilmiy yangilik', 'Научная новизна')))

        positions = [
            L('3D yuz geometriyasini normallash usuli (umumlashgan Prokrust tahlili + koordinatalar bo‘yicha standartlash): '
              'geometrik tanish aniqligi o‘qitishsiz 53,7% dan 67,5% ga oshadi.',
              'Метод нормализации трёхмерной геометрии лица (обобщённый анализ Прокруста + покоординатная стандартизация): '
              'точность геометрического распознавания без обучения растёт с 53,7% до 67,5%.'),
            L('478 ta 3D nuqta uchun GeometricTransformer neyron tarmoq modeli va augmentatsiyalarni shaxslararo farq bo‘yicha '
              'kalibrlash tamoyili: LFW da 78,0%, 423 ming parametr.',
              'Нейросетевая модель GeometricTransformer для 478 трёхмерных точек лица и принцип калибровки аугментаций по '
              'межличностному разбросу: 78,0% на LFW при 423 тыс. параметров.'),
            L('Gibrid arxitektura: geometrik modul neyron tarmoq uchun yuzni topadi, tekislaydi va pozani baholaydi; sifat va pozani '
              'hisobga oluvchi adaptiv birlashtirish: LFW da 99,67%, CPLFW da 96,28%.',
              'Гибридная архитектура: геометрический модуль выполняет для нейросети детекцию, выравнивание и оценку позы; '
              'адаптивное слияние с учётом качества и позы: 99,67% на LFW, 96,28% на CPLFW.'),
            L('Yuz burilishi katta bo‘lganda geometrik belgilar SI ishonchliligini statistik ahamiyatli oshirishi isbotlangan: '
              'TAR@FAR=1% +6…11 foiz punkt, p < 0,002.',
              'Экспериментально доказано, что при большой разнице ракурсов геометрические признаки статистически значимо '
              'повышают надёжность ИИ: TAR@FAR=1% +6…11 п.п., p < 0,002.'),
            L('Gibrid biometrik shablonni o‘rnatish kaliti bilan himoyalash usuli (maxfiy ortogonal o‘zgartirishlar + nuqtalarni '
              'shifrlash), qarorlar saqlanishi isbotlangan: 50 156 juftlikda 0 ta o‘zgarish, bog‘lanmaslik AUC 0,49–0,51, '
              'rasmlarsiz bekor qilish.',
              'Метод защиты гибридного биометрического шаблона ключом установки (секретные ортогональные преобразования + '
              'шифрование точек) с доказанным сохранением решений: 0 изменений на 50 156 парах, несвязываемость AUC 0,49–0,51, '
              'отзыв без исходных фото.'),
            L('Dasturiy amalga oshirish — gfram kutubxonasi (PyPI), server orqali model yangilanishi va himoyalangan shablonlar: '
              'LFW da 99,53%, protsessorda ~24 ms bitta yuzga.',
              'Программная реализация — библиотека gfram (PyPI) с серверным обновлением модели и защищёнными шаблонами: '
              '99,53% на LFW, ~24 мс на лицо на процессоре.'),
        ]
        c, lay = card(L('Himoyaga olib chiqiladigan holatlar', 'Положения, выносимые на защиту'))
        for i, t in enumerate(positions, 1):
            lay.addWidget(label(f'<b style="color:{theme.ACCENT}">{i}.</b> {t}'))
        root.addWidget(c)

        c, lay = card(L('Afzalliklar', 'Преимущества'))
        for t in (
            L('Kalibrlangan ehtimollik, oddiy «o‘xshashlik» emas; tayyor chegaralar (0,526 va 0,90).',
              'Калиброванная вероятность, а не «сходство»; готовые пороги (0,526 и 0,90).'),
            L('Bitta geometrik modul hamma ishni qiladi: yuzni topish, tekislash, poza, mustaqil shaxs kanali.',
              'Один геометрический модуль делает всё: поиск лица, выравнивание, поза, независимый канал личности.'),
            L('Yengil: 15 MB model, videokartasiz real vaqt.', 'Лёгкость: 15 МБ моделей, реальное время без видеокарты.'),
            L('Model yangilanganda odamlarni qayta ro‘yxatdan o‘tkazish shart emas.', 'Обновление модели без перерегистрации людей.'),
            L('Rasmlar saqlanmaydi; kalit almashtirilsa eski shablonlar yaroqsiz.', 'Фото не хранятся; после смены ключа старые шаблоны бесполезны.'),
        ):
            lay.addWidget(label('• ' + t))
        root.addWidget(c)

        c, lay = card(L('Ma’lumotlar va komponentlar', 'Данные и компоненты'))
        for t in (
            L('O‘qitish: CelebA — 198 803 yuz, 9 325 odam. Sinov: LFW — 13 233 rasm, 6 000 juftlik; CPLFW — 6 000 juftlik.',
              'Обучение: CelebA — 198 803 лица, 9 325 человек. Тест: LFW — 13 233 фото, 6 000 пар; CPLFW — 6 000 пар.'),
            L('MediaPipe FaceMesh (478 nuqta); tashqi ko‘rinish modeli — oldindan o‘qitilgan InsightFace MobileFaceNet (WebFace600K), '
              'ish davomida o‘qitilmagan.',
              'MediaPipe FaceMesh (478 точек); модель внешности — предобученная InsightFace MobileFaceNet (WebFace600K), '
              'в рамках работы не обучалась.'),
            L('Muallif ishlanmasi: geometrik modul (Prokrust normallash, GeometricTransformer, poza, tekislash), adaptiv '
              'birlashtirish, himoyalangan shablonlar, gfram kutubxonasi va server.',
              'Разработка автора: геометрический модуль (нормализация Прокруста, GeometricTransformer, поза, выравнивание), '
              'адаптивное слияние, защищённые шаблоны, библиотека gfram и сервер.'),
        ):
            lay.addWidget(label('• ' + t))
        info = engine.model_info()
        lay.addWidget(label(L(f'Yuklangan modellar: geometrik v{info["geometry_version"]}, {info["appearance_model"]}, '
                              f'birlashtirish — {len(info["fusion_features"])} belgi.',
                              f'Загруженные модели: геометрическая v{info["geometry_version"]}, {info["appearance_model"]}, '
                              f'слияние — {len(info["fusion_features"])} признаков.'), muted=True))
        root.addWidget(c)
        root.addStretch(1)
