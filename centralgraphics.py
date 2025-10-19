import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from typing import Optional, List, Tuple, Union

sns.set_style('white')

font_dirs = ["fonts/"]
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)

if font_files:
    prop = fm.FontProperties(fname=font_files[0])
    plt.rcParams["font.family"] = prop.get_name()


class CentralGraphics:
    """
    Класс для создания стилизованных графиков в дизайне Центрального Университета.
    
    Поддерживает создание различных типов визуализаций (scatter, histplot, 
    boxplot, pieplot) с общими настройками стиля, цветовой палитрой и шрифтами.
    
    Attributes:
        text_color (str): Цвет текста для меток и подписей.
        background_color (str): Цвет фона графиков.
        tick_fontsize (int): Размер шрифта для меток осей.
        label_fontsize (int): Размер шрифта для названий осей.
        palette (List[str]): Цветовая палитра для графиков.
    
    Example:
        >>> g = Graphics()
        >>> fig, ax = g.scatter(df, x='column1', y='column2', hue='category')
        >>> plt.show()
    """
    
    DEFAULT_TEXT_COLOR = '#141414'
    DEFAULT_BACKGROUND_COLOR = '#E6E6E6'
    DEFAULT_TICK_FONTSIZE = 22
    DEFAULT_LABEL_FONTSIZE = 30
    DEFAULT_PALETTE = ['#775AFF', '#FF662C', '#00A651', '#FE68B9', '#FFDD2D']
    DEFAULT_FIGSIZE = (10, 10)
    
    def __init__(
        self,
        text_color: str = DEFAULT_TEXT_COLOR,
        background_color: str = DEFAULT_BACKGROUND_COLOR,
        tick_fontsize: int = DEFAULT_TICK_FONTSIZE,
        label_fontsize: int = DEFAULT_LABEL_FONTSIZE,
        palette: List[str] = None
    ):
        """
        Инициализация класса Graphics с настройками стиля.
        
        Args:
            text_color: Цвет текста (hex-формат).
            background_color: Цвет фона (hex-формат).
            tick_fontsize: Размер шрифта для меток осей.
            label_fontsize: Размер шрифта для названий осей.
            palette: Список цветов для графиков (hex-формат).
        """
        self.text_color = text_color
        self.background_color = background_color
        self.tick_fontsize = tick_fontsize
        self.label_fontsize = label_fontsize
        self.palette = palette if palette is not None else self.DEFAULT_PALETTE.copy()
    
    def _setup_axes(
        self,
        ax: plt.Axes,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        x_rot: float = 0,
        y_rot: float = 0
    ) -> None:
        """
        Применяет общие настройки к осям графика.
        
        Args:
            ax: Объект осей matplotlib.
            x_label: Название оси X.
            y_label: Название оси Y.
            x_rot: Угол поворота меток оси X.
            y_rot: Угол поворота меток оси Y.
        """
        ax.set_facecolor(self.background_color)
        
        plt.xticks(
            fontsize=self.tick_fontsize,
            color=self.text_color,
            rotation=x_rot
        )
        plt.yticks(
            fontsize=self.tick_fontsize,
            color=self.text_color,
            rotation=y_rot
        )
        
        if x_label:
            plt.xlabel(x_label, fontsize=self.label_fontsize, color=self.text_color)
        
        if y_label:
            plt.ylabel(y_label, fontsize=self.label_fontsize, color=self.text_color)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    def _setup_legend(
        self,
        ax: plt.Axes,
        hue: str,
        legend_fontsize: int
    ) -> None:
        """
        Настраивает легенду графика.
        
        Args:
            ax: Объект осей matplotlib.
            hue: Название переменной для группировки.
            legend_fontsize: Размер шрифта легенды.
        """
        ax.legend(
            fontsize=legend_fontsize,
            title=hue,
            title_fontsize=legend_fontsize + 2
        )
    
    def scatter(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: Optional[str] = None,
        x_rot: float = 0,
        y_rot: float = 0,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        legend_fontsize: int = 20,
        alpha: float = 0.8
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Создает диаграмму рассеяния (scatter plot).
        
        Args:
            df: DataFrame с данными.
            x: Название столбца для оси X.
            y: Название столбца для оси Y.
            hue: Название столбца для цветовой группировки (опционально).
            x_rot: Угол поворота меток оси X в градусах.
            y_rot: Угол поворота меток оси Y в градусах.
            figsize: Размер графика (ширина, высота).
            legend_fontsize: Размер шрифта легенды.
            alpha: Прозрачность точек (0-1).
        
        Returns:
            Кортеж (figure, axes) объектов matplotlib.
        
        Example:
            >>> g = Graphics()
            >>> fig, ax = g.scatter(df, x='age', y='income', hue='gender')
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.background_color)
        
        if hue:
            sns.scatterplot(
                data=df,
                x=x,
                y=y,
                palette=self.palette,
                hue=hue,
                edgecolor=None,
                alpha=alpha,
                ax=ax
            )
            self._setup_legend(ax, hue, legend_fontsize)
        else:
            sns.scatterplot(
                data=df,
                x=x,
                y=y,
                color=self.palette[0],
                edgecolor=None,
                alpha=alpha,
                ax=ax
            )
        
        self._setup_axes(ax, x_label=x, y_label=y, x_rot=x_rot, y_rot=y_rot)
        
        return fig, ax
    
    def histplot(
        self,
        df: pd.DataFrame,
        x: str,
        hue: Optional[str] = None,
        x_rot: float = 0,
        y_rot: float = 0,
        legend_fontsize: int = 20,
        bins: Optional[Union[int, List]] = None,
        binwidth: Optional[float] = None,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Создает гистограмму распределения.
        
        Args:
            df: DataFrame с данными.
            x: Название столбца для гистограммы.
            hue: Название столбца для цветовой группировки (опционально).
            x_rot: Угол поворота меток оси X в градусах.
            y_rot: Угол поворота меток оси Y в градусах.
            legend_fontsize: Размер шрифта легенды.
            bins: Количество столбцов или список границ.
            binwidth: Ширина одного столбца.
            figsize: Размер графика (ширина, высота).
        
        Returns:
            Кортеж (figure, axes) объектов matplotlib.
        
        Example:
            >>> g = Graphics()
            >>> fig, ax = g.histplot(df, x='age', bins=20, hue='gender')
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.background_color)
        
        hist_params = {
            'data': df,
            'x': x,
            'edgecolor': 'none',
            'ax': ax
        }
        
        if bins is not None:
            hist_params['bins'] = bins
        elif binwidth is not None:
            hist_params['binwidth'] = binwidth
        
        if hue:
            hist_params['palette'] = self.palette
            hist_params['hue'] = hue
            sns.histplot(**hist_params)
            sns.move_legend(
                ax,
                "best",
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize + 2
            )
        else:
            hist_params['color'] = self.palette[0]
            sns.histplot(**hist_params)
        
        self._setup_axes(ax, x_label=x, y_label='Count', x_rot=x_rot, y_rot=y_rot)
        
        return fig, ax
    

    def boxplot(
        self,
        df: pd.DataFrame,
        x: Optional[str] = None,
        y: Optional[str] = None,
        hue: Optional[str] = None,
        x_rot: float = 0,
        y_rot: float = 0,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        legend_fontsize: int = 20
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Создает коробчатую диаграмму (box plot).
        
        Args:
            df: DataFrame с данными.
            x: Название столбца для оси X (опционально).
            y: Название столбца для оси Y (опционально).
            hue: Название столбца для цветовой группировки (опционально).
            x_rot: Угол поворота меток оси X в градусах.
            y_rot: Угол поворота меток оси Y в градусах.
            figsize: Размер графика (ширина, высота).
            legend_fontsize: Размер шрифта легенды.
        
        Returns:
            Кортеж (figure, axes) объектов matplotlib.
        
        Example:
            >>> g = Graphics()
            >>> fig, ax = g.boxplot(df, x='category', y='value', hue='group')
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.background_color)
        
        if hue:
            sns.boxplot(data=df, x=x, y=y, hue=hue, palette=self.palette, ax=ax)
            self._setup_legend(ax, hue, legend_fontsize)
        else:
            sns.boxplot(data=df, x=x, y=y, color=self.palette[0], ax=ax)
        
        self._setup_axes(ax, x_label=x, y_label=y, x_rot=x_rot, y_rot=y_rot)
        
        return fig, ax
    

    def pieplot(
        self,
        df: pd.DataFrame,
        values: str,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        autopct: str = '%1.1f%%',
        startangle: float = 90,
        label_fontsize: Optional[int] = None,
        explode: Optional[Tuple[float, ...]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Создает круговую диаграмму (pie chart).
        
        Args:
            df: DataFrame с данными.
            values: Название столбца со значениями.
            figsize: Размер графика (ширина, высота).
            autopct: Формат процентов (например, '%1.1f%%').
            startangle: Начальный угол первого сектора в градусах.
            label_fontsize: Размер шрифта для меток.
            explode: Кортеж со смещениями секторов (опционально).
        
        Returns:
            Кортеж (figure, axes) объектов matplotlib.
        
        Example:
            >>> g = Graphics()
            >>> fig, ax = g.pieplot(df, values='category')
        """
        if label_fontsize is None:
            label_fontsize = self.tick_fontsize
        
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.background_color)
        ax.set_facecolor(self.background_color)
        
        pie_values = df[values].value_counts().sort_index(ascending=True)
        pie_labels = pie_values.index
        
        num_colors = len(pie_values)
        colors = (self.palette * ((num_colors // len(self.palette)) + 1))[:num_colors]
        
        text_props = {'fontsize': label_fontsize, 'color': self.text_color}
        
        ax.pie(
            pie_values,
            labels=pie_labels,
            colors=colors,
            autopct=autopct,
            startangle=startangle,
            explode=explode,
            textprops=text_props
        )
        
        ax.axis('equal')
        
        return fig, ax
    

    def barplot(
        self,
        pivot_data: pd.DataFrame,
        figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
        legend_fontsize: int = 20,
        stacked: bool = False,
        percentage_labels: bool = False,
        percentage_fontsize: int = 20,
        percentage_color: str = None,
        x_rot: float = 0,
        y_rot: float = 0,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Создает столбчатую диаграмму (bar plot) на основе pivot таблицы.

        Args:
            pivot_data: Pivot таблица с данными (индексы = категории, колонки = группы).
            figsize: Размер графика (ширина, высота).
            legend_fontsize: Размер шрифта легенды.
            stacked: Создать стековую диаграмму.
            percentage_labels: Добавить процентные подписи на стековую диаграмму.
            percentage_fontsize: Размер шрифта для процентных подписей.
            percentage_color: Цвет процентных подписей (по умолчанию используется text_color класса).
            x_rot: Угол поворота меток оси X в градусах.
            y_rot: Угол поворота меток оси Y в градусах.
            x_label: Название оси X (по умолчанию - имя индекса pivot таблицы).
            y_label: Название оси Y (по умолчанию - 'Value').

        Returns:
            Кортеж (figure, axes) объектов matplotlib.

        Example:
            >>> # Создание pivot таблицы
            >>> pivot_data = df.pivot_table(index='category', columns='group',
            ...                            values='value', aggfunc='sum', fill_value=0)
            >>> g = CentralGraphics()
            >>> # Стековая диаграмма с белыми процентными подписями
            >>> fig, ax = g.barplot(pivot_data, stacked=True, percentage_labels=True,
            ...                     percentage_color='white')
            >>> # Стековая диаграмма с красными процентными подписями
            >>> fig, ax = g.barplot(pivot_data, stacked=True, percentage_labels=True,
            ...                     percentage_color='#FF0000')
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.background_color)

        num_colors = len(pivot_data.columns)
        colors = self.palette[:num_colors] if num_colors <= len(self.palette) else (
            self.palette * ((num_colors // len(self.palette)) + 1)
        )[:num_colors]

        percent_text_color = percentage_color if percentage_color is not None else self.text_color

        if stacked:
            pivot_data.plot(
                kind='bar',
                stacked=True,
                color=colors,
                ax=ax,
                width=0.7,
                legend=False,
                edgecolor='none'
            )

            if percentage_labels:
                pivot_percents = pivot_data.div(pivot_data.sum(axis=1), axis=0)

                for i, group in enumerate(pivot_percents.index):
                    cumulative = 0
                    for j, hue_val in enumerate(pivot_percents.columns):
                        value = pivot_percents.loc[group, hue_val]
                        if value > 0:
                            abs_value = pivot_data.loc[group, hue_val]
                            ax.text(
                                i, cumulative + abs_value / 2,
                                f'{value * 100:.1f}%',
                                ha='center', va='center',
                                color=percent_text_color,
                                fontsize=percentage_fontsize,
                                weight='bold'
                            )
                            cumulative += abs_value
        else:
            pivot_data.plot(
                kind='bar',
                color=colors,
                ax=ax,
                width=0.7,
                legend=False,
                edgecolor='none'
            )

        if len(pivot_data.columns) > 1:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                fontsize=legend_fontsize,
                title=pivot_data.columns.name or 'Groups',
                title_fontsize=legend_fontsize + 2,
                loc='best'
            )

        x_axis_label = x_label or pivot_data.index.name or 'Categories'
        y_axis_label = y_label or 'Value'

        self._setup_axes(ax, x_label=x_axis_label, y_label=y_axis_label,
                        x_rot=x_rot, y_rot=y_rot)

        return fig, ax