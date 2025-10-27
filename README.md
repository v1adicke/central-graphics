# CentralGraphics

Библиотека для создания стилизованных графиков в дизайне Центрального Университета с использованием matplotlib и seaborn.

## Требования

```python
pandas
seaborn
matplotlib
```

## Цветовая палитра

По умолчанию используется палитра Центрального Университета:

```python
DEFAULT_PALETTE = [    '#775AFF',  # Фиолетовый
    '#FF662C',  # Оранжевый
    '#00A651',  # Зелёный
    '#FE68B9',  # Розовый
    '#FFDD2D'   # Жёлтый
]
```

## Настройка параметров класса

При создании экземпляра класса можно настроить глобальные параметры оформления:

```python
g = CentralGraphics(
    text_color='#000000',           # Цвет текста
    background_color='#FFFFFF',     # Цвет фона
    tick_fontsize=18,               # Размер шрифта меток осей
    label_fontsize=24,              # Размер шрифта названий осей
    palette=['#FF0000', '#00FF00']  # Пользовательская палитра
)
```

## Структура проекта

```
CentralGraphics/
│
├── central_graphics.py                  # Основной модуль с классом
├── fonts/                               
│   ├── Inter-VariableFont_slnt,wght.ttf # Классический шрифт ЦУ
├── requirements.txt                     # Требования
└── README.md                            # Документация
```

## Автор

tg: @v1adicke14
