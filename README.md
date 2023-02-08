# Тестовое задание (RecSys)

## Цель

**Онлайн-магазин накопил данные по взаимодействию покупателей с товарами за несколько месяцев. Цель — рекомендовать товар, который вызовет взаимодействие покупателя с ним.**

## Вводные

1. Задача — рекомендовать каждому покупателю список из 10 потенциально релевантных товаров.
2. Для оценки качества рекомендаций используется метрика MAP@10
3. Вы можете использовать любой алгоритм рекомендательной системы (коллаборативная фильтрация, content-based, гибридная итп). Плюсом будет демонстрация использования сразу нескольких алгоритмов.
4. Для разбиения датасета на train и test используйте рандомный сплит 80/20

## Описание данных

**Все данные хранятся в CSV-файлах, разделенных ";".**

interactions.csv — файл хранит данные по взаимодействию товаров и покупателей. Среди данных есть "холодные" товары и покупатели. В колонке row хранятся идентификаторы покупателя. В колонке col идентификаторы товара. В колонке data - значение взаимодействия.

**Данные по товарам**

item_asset.csv - файл хранит качественную характеристику товара. row - идентификатор товара, data - значение характеристики. col - порядковый номер фичи при выгрузке данных (смысла не несет, можно избавиться от этого столбца)

item_price.csv - файл хранит цену товара (уже нормализована). row - идентификатор товара, data - нормализованное значение цены. col - порядковый номер фичи при выгрузке данных (смысла не несет, можно избавиться от этого столбца)

item_subclass.csv - файл хранит значения категорий, к которым относится товар. row - идентификатор товара, col - номер категории, data - признак отношения к категории

**Данные по пользователям**

user_age.csv - файл хранит данные по возрасту пользователей. row - идентификатор пользователя, data - значение возраста (уже нормализованное), col - порядковый номер фичи при выгрузке данных (смысла не несет, можно избавиться от этого столбца)

user_region.csv - файл хранит one-hot encoded значения региона пользователя. row - идентификатор пользователя, col - номер one-hot feature региона, data - признак региона.
