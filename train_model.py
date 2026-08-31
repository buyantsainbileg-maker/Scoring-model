# -*- coding: utf-8 -*-
"""
Байрны үнэ таамаглах загварыг сургах скрипт.

Дараах алхмуудыг хийнэ:
1. osdata.csv файлыг уншина.
2. "Үнэ" баганыг цэвэрлэнэ (хоосон, 0, болон 10 дахин их бичигдсэн утгууд).
3. "Дүүрэг" баганын кодыг (0-5) хүн ойлгомжтой нэртэй харгалзуулж,
   тэр харгалзааг JSON файлд хадгална (веб апп яг ижил харгалзааг ашиглана).
4. RandomForest регрессор сургана.
5. Загварын чанарыг (MAE, R2) тест дата дээр хэмжиж, файлд хадгална.
6. Загвар болон бүх шаардлагатай мэдээллийг диск рүү бичнэ.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Дата болон гаралтын файлуудын нэр
DATA_FILE = "osdata.csv"
MODEL_FILE = "price_model.joblib"
DISTRICT_MAP_FILE = "district_mapping.json"
METRICS_FILE = "metrics.json"

# Үнэ хэт өндөр (аравтаар алдаатай бичигдсэн) гэж үзэх босго.
# Дата дээр шинжилгээ хийхэд 300 сая-с дээш үнэ яг 30 мөрд тохиолдож байсан бөгөөд
# энэ нь бусад цэвэр өгөгдлийн 99 перцентилээс мэдэгдэхүйц давж, харин 10-т
# хуваахад ердийн үнийн мужид (20-140 сая) тохирч байгаа тул сонгосон.
HIGH_PRICE_THRESHOLD = 300_000_000

# Дата дотор дүүрэг зөвхөн 0-5 гэсэн кодоор бичигдсэн, нэргүй байдаг.
# Улаанбаатарын хамгийн олон байртай 6 дүүргийн нэрийг код бүрт харгалзуулна.
# Энэ харгалзаа бол таамаглал (дата дотор бодит нэр байхгүй) тул зөвхөн
# веб апп дээр хэрэглэгчид ойлгомжтой сонголт үзүүлэх зорилготой.
DISTRICT_NAMES = {
    0: "Баянгол",
    1: "Баянзүрх",
    2: "Сонгинохайрхан",
    3: "Сүхбаатар",
    4: "Хан-Уул",
    5: "Чингэлтэй",
}


def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """"Үнэ" баганыг цэвэрлэнэ: хоосон/0-г хаяж, хэт өндөр утгыг 10-д хуваана."""

    df = df.copy()

    # Тоо бус (хоосон мөр) утгуудыг NaN болгож хувиргана
    df["Үнэ"] = pd.to_numeric(df["Үнэ"], errors="coerce")

    # Хоосон (NaN) үнэтэй мөрүүдийг хаяна — үнэ мэдэгдэхгүй бол сургалтад ашиглах
    # боломжгүй тул алдаа болгож бус, зүгээр л дата-с хасна.
    df = df.dropna(subset=["Үнэ"])

    # 0 төгрөгтэй мөрүүд бодит байж чадахгүй тул хаяна.
    df = df[df["Үнэ"] > 0]

    # Хэт өндөр (жишээ нь 10 дахин их бичигдсэн) үнийг 10-д хуваан засна.
    too_high = df["Үнэ"] > HIGH_PRICE_THRESHOLD
    df.loc[too_high, "Үнэ"] = df.loc[too_high, "Үнэ"] / 10

    return df


def main():
    # 1. Дата уншиж, цэвэрлэнэ
    df = pd.read_csv(DATA_FILE, encoding="utf-8")
    n_before = len(df)
    df = clean_price_column(df)
    n_after = len(df)
    print(f"Цэвэрлэгээний өмнө: {n_before} мөр, дараа нь: {n_after} мөр")

    # 2. Дүүргийн код -> нэр харгалзааг JSON файлд хадгалж, дараа нь
    # веб апп яг ижил харгалзааг ашиглах боломжтой болгоно.
    with open(DISTRICT_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(DISTRICT_NAMES, f, ensure_ascii=False, indent=2)

    # 3. Загвар сургах өгөгдлийг бэлдэнэ
    feature_columns = ["Талбай", "Давхар", "Ашиглалтын он", "Дүүрэг"]
    X = df[feature_columns]
    y = df["Үнэ"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Загвар сургана
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Тест дата дээр чанарыг хэмжинэ
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE (дундаж андуурал): {mae:,.0f} төгрөг")
    print(f"R2 (тайлбарлах чадвар): {r2:.3f}")

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mae": float(mae),
                "r2": float(r2),
                "n_train": len(X_train),
                "n_test": len(X_test),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 6. Загварыг хадгална
    joblib.dump(model, MODEL_FILE)
    print(f"Загвар хадгалагдлаа: {MODEL_FILE}")


if __name__ == "__main__":
    main()
