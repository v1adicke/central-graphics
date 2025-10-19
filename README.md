# CentralGraphics

Библиотека для создания стилизованных графиков в дизайне Центрального Университета с использованием matplotlib и seaborn.

## 📦 Требования

```python
pandas
seaborn
matplotlib
```

## 📊 Типы графиков

### 1. Scatter Plot (Диаграмма рассеяния)

Используется для визуализации взаимосвязи между двумя числовыми переменными.

```python
g = CentralGraphics()

# Простой scatter plot
fig, ax = g.scatter(df, x='experience', y='salary')

# С группировкой по категории
fig, ax = g.scatter(
    df, 
    x='experience', 
    y='salary', 
    hue='department',
    alpha=0.7,
    figsize=(12, 8),
    x_rot=45
)
plt.show()
```

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `x` (str): Название столбца для оси X
- `y` (str): Название столбца для оси Y
- `hue` (str, optional): Столбец для цветовой группировки
- `x_rot` (float): Угол поворота меток оси X (градусы)
- `y_rot` (float): Угол поворота меток оси Y (градусы)
- `figsize` (tuple): Размер графика (ширина, высота)
- `legend_fontsize` (int): Размер шрифта легенды (по умолчанию 20)
- `alpha` (float): Прозрачность точек от 0 до 1 (по умолчанию 0.8)

### 2. Histogram (Гистограмма)

Отображает распределение одной числовой переменной.

```python
g = CentralGraphics()

# Простая гистограмма
fig, ax = g.histplot(df, x='age', bins=30)

# С группировкой и пользовательскими bins
fig, ax = g.histplot(
    df,
    x='age',
    hue='gender',
    bins=20,
    figsize=(12, 8),
    x_rot=0
)
plt.show()

# С заданной шириной столбцов
fig, ax = g.histplot(df, x='salary', binwidth=5000)
plt.show()
```

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `x` (str): Название столбца для гистограммы
- `hue` (str, optional): Столбец для цветовой группировки
- `x_rot` (float): Угол поворота меток оси X
- `y_rot` (float): Угол поворота меток оси Y
- `legend_fontsize` (int): Размер шрифта легенды (по умолчанию 20)
- `bins` (int или list, optional): Количество столбцов или список границ
- `binwidth` (float, optional): Ширина одного столбца
- `figsize` (tuple): Размер графика

### 3. Box Plot (Коробчатая диаграмма)

Показывает распределение данных через квартили.

```python
g = CentralGraphics()

# Простой box plot
fig, ax = g.boxplot(df, y='salary')

# Box plot с группировкой
fig, ax = g.boxplot(
    df,
    x='department',
    y='salary',
    hue='gender',
    x_rot=45,
    figsize=(14, 8)
)
plt.show()
```

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `x` (str, optional): Название столбца для оси X
- `y` (str, optional): Название столбца для оси Y
- `hue` (str, optional): Столбец для цветовой группировки
- `x_rot` (float): Угол поворота меток оси X
- `y_rot` (float): Угол поворота меток оси Y
- `figsize` (tuple): Размер графика
- `legend_fontsize` (int): Размер шрифта легенды (по умолчанию 20)

### 4. Pie Plot (Круговая диаграмма)

Отображает пропорции категориальных данных.

```python
g = CentralGraphics()

# Простая круговая диаграмма
fig, ax = g.pieplot(df, values='category')

# С дополнительными параметрами
fig, ax = g.pieplot(
    df,
    values='department',
    figsize=(10, 10),
    autopct='%1.2f%%',
    startangle=45,
    explode=(0, 0.1, 0, 0)  # Выдвинуть второй сектор
)
plt.show()
```

**Параметры:**
- `df` (pd.DataFrame): DataFrame с данными
- `values` (str): Название столбца со значениями
- `figsize` (tuple): Размер графика
- `autopct` (str): Формат процентов (по умолчанию '%1.1f%%')
- `startangle` (float): Начальный угол первого сектора (по умолчанию 90)
- `label_fontsize` (int, optional): Размер шрифта для меток
- `explode` (tuple, optional): Кортеж со смещениями секторов

## 🎨 Цветовая палитра

По умолчанию используется палитра Центрального Университета:

```python
DEFAULT_PALETTE = [
    '#775AFF',  # Фиолетовый
    '#FF662C',  # Оранжевый
    '#00A651',  # Зелёный
    '#FE68B9',  # Розовый
    '#FFDD2D'   # Жёлтый
]
```

### Кастомизация цветов

```python
# Создание экземпляра с пользовательской палитрой
custom_palette = ['#FF5733', '#33FF57', '#3357FF', '#F033FF']
g = CentralGraphics(palette=custom_palette)

fig, ax = g.scatter(df, x='x', y='y', hue='category')
plt.show()
```

## ⚙️ Настройка параметров класса

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

### Параметры по умолчанию

```python
DEFAULT_TEXT_COLOR = '#141414'
DEFAULT_BACKGROUND_COLOR = '#E6E6E6'
DEFAULT_TICK_FONTSIZE = 22
DEFAULT_LABEL_FONTSIZE = 30
DEFAULT_FIGSIZE = (10, 10)
```

## 🔧 Работа со шрифтами

Класс автоматически загружает классический шрифт Центрального Университета Inter из директории `fonts/`:

```python
font_dirs = ["fonts/"]
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)

if font_files:
    prop = fm.FontProperties(fname=font_files[0])
    plt.rcParams["font.family"] = prop.get_name()
```

## 📂 Структура проекта

```
CentralGraphics/
│
├── central_graphics.py                  # Основной модуль с классом
├── fonts/                               
│   ├── Inter-VariableFont_slnt,wght.ttf # Классический шрифт ЦУ
├── requirements.txt                     # Требования
└── README.md                            # Документация
```

## 🎯 Советы по использованию

1. **Размер графиков**: Для презентаций используйте `figsize=(14, 10)` или больше
2. **Сохранение**: Всегда указывайте `facecolor=g.background_color` при сохранении
3. **DPI**: Для публикаций используйте `dpi=300` при сохранении
4. **Прозрачность**: Для плотных данных уменьшите `alpha` до 0.5-0.7
5. **Легенда**: Если легенда перекрывает данные, используйте `sns.move_legend()`

## 👤 Автор

tg: @v1adicke14

**Версия:** 1.0.0  
**Последнее обновление:** Октябрь 2025