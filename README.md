# larek
<img src="larek.png" alt="larek" style="zoom:50%;" />

Репозиторий с кодом для ВКР "Адаптивное управление ассортиментом и запасами автономной торговой точки" (Митяй Герман Витальевич)

#### Пример работы
<img src="work_gif.gif" alt="work_gif" style="zoom:67%;" />

#### Необходимые данные для работы

* [Ссылка](https://drive.google.com/drive/folders/1DqqlrWkxWvYZdCnmdz8kgCH_AjtobbmQ?usp=share_link) на скачивание данных для обучения, а также результатов: `.csv` файлов для вывода в Streamlit. Вес более 2.5 ГБ
* [Ссылка](https://drive.google.com/drive/folders/18RTrKjNZ89JyEzjpcaj6CPqXgxlKrZT7?usp=share_link) на скачивание историй оптимизиации моделей, а также на обученные модели с лучшими найденными параметрами. Вес более 0.3 ГБ

Для корректной работы необходимо скаченные папки `optimized_models` и `data` разместить в папке с кодом

#### Устройство репозитория

```
├── data - содержит необходимые данные и результаты работы
├── optimized_models - содержит предобученные модели и истории оптимизации
├── tools
│   ├── tools_forecasting.py - вспомогательные классы и функции для прогнозирования
│   └── tools_recommendation.py - вспомогательные классы и функции для прогнозирования
├── create_forecast.py - генерируем прогноз
├── create_recommendations.py - генерируем рекомендации
├── main.py - запуск сервиса
├── optimization_forcasting.ipynb - оптимизация и сравнение моделей прогнозирования
└── optimization_recommendations.ipynb - оптимизация, обучение моделей рекомендаций и сравнение
```
#### Схема запуска сервиса

* В случае предварительной загрузки результатов - только запускаем из консоли `main.py` с помощью команды `streamlit run main.py`
* Иначе
  * Оптимизируем и обучаем модели в ноутбуках `optimization_forcasting.ipynb`, `optimization_recommendations.ipynb`, сохраняя модели в `./optimization_models/`
  * Генерируем прогноз и рекомендации `create_forecast.py`, `create_recommendations.py`, сохраняя результаты в `./data/results/`
  * Запускаем из консоли `main.py` с помощью команды `streamlit run main.py`
