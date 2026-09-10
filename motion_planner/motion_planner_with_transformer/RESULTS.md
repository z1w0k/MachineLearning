# Результаты ego planning

Одинаковая модель обучалась на одном train/validation split. Отличался только
способ выбора train-сцен.

| Sampling | ADE, м | FDE, м | Collision rate | Route deviation, м |
|---|---:|---:|---:|---:|
| Random | 1.1726 | 2.6354 | 0.0250 | 0.3139 |
| Hard mining | 1.1424 | 2.6011 | 0.0350 | 0.3112 |

Hard mining немного улучшил ADE и FDE, но collision rate вырос.
Значит, текущий difficulty score полезен для точности, но его нужно улучшить с
учётом безопасности.

## Hard mining по сценариям

| Сценарий | ADE, м | FDE, м |
|---|---:|---:|
| Straight | 0.7838 | 1.6292 |
| Acceleration or braking | 1.5952 | 3.6640 |
| Left or right turn | 0.9996 | 2.3910 |
| Leader following | 1.2509 | 2.8606 |
