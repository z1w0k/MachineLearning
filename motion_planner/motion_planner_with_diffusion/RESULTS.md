# Результаты diffusion ego planning

Для каждой validation-сцены генерировалось 6 траекторий.

| Метрика | Значение |
|---|---:|
| minADE | 1.8132 м |
| minFDE | 1.9414 м |
| Collision rate | 0.0238 |
| Route deviation | 1.2081 м |
| Mean acceleration | 95.9666 |
| Mean jerk | 889.8535 |
| Endpoint spread | 3.2355 м |

Высокие acceleration и jerk показывают, что координатной diffusion-модели не
хватает ограничения на гладкость. Это важное ограничение текущей версии, а не
ошибка вычисления метрик.

## По сценариям

| Сценарий | minADE, м | minFDE, м |
|---|---:|---:|
| Straight | 1.6934 | 1.6642 |
| Acceleration or braking | 1.9482 | 2.4830 |
| Left or right turn | 1.7563 | 1.6621 |
| Leader following | 1.8549 | 1.9302 |
