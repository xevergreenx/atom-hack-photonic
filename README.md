# Нейросетевой синтез многоплоскостного фазового ДОЭ

Хакатон НИЯУ МИФИ, задача лаборатории «Фотоника и информационная оптика»:
4 изображения 128×128 → одна фазовая маска (ДОЭ) 128×128, которая по формуле Френеля
восстанавливает каждое изображение на своём расстоянии (24.1 / 25.3 / 26.5 / 27.7 мм, λ = 532 нм, пиксель 10 мкм).

Вся логика решений, вопросы, эксперименты и план перебора — в [research_log.md](research_log.md).

## Файлы

| Файл | Что делает |
|---|---|
| `doe_baseline.py` | Физика (Френель), метрики (КК, SSIM), классические методы: суперпозиция, GS, Adam по фазе |
| `data.py` | Генератор обучающих четвёрок: фото + карта глубины → 4 слоя |
| `research_log.md` | Журнал исследования, из него пишем статью |
| `out/` | Картинки и результаты экспериментов |

## Как воспроизвести

```bash
# бейзлайны без нейросети
python3 doe_baseline.py

# данные: скачать COCO val2017 и DIV2K valid HR в data/raw/, затем собрать пулы сцен
python3 -I data.py prepare data/raw/coco/val2017 data/pool_coco.npz
python3 -I data.py prepare data/raw/div2k/DIV2K_valid_HR data/pool_div2k.npz
python3 -I data.py preview data/pool_coco.npz out/layers_preview.png
```

Источники данных: [COCO 2017](https://cocodataset.org) (val2017.zip), [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (DIV2K_valid_HR.zip),
модель глубины `depth-anything/Depth-Anything-V2-Small-hf` (Apache-2.0, скачивается автоматически).
Данные и пулы в репозиторий не входят (см. `.gitignore`).
