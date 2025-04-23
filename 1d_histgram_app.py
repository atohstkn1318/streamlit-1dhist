import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# タイトル
st.title("1Dヒストグラム（エネルギー和,カウント数）")

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, skiprows=45)
    else:
        df = pd.read_excel(uploaded_file, skiprows=45)

    # 必要なカラムがあるか確認
    if {"CH1(ch)", "CH2(ch)", "Counts"}.issubset(df.columns):
        # エネルギー和を計算
        df["sum_energy"] = df["CH1(ch)"] + df["CH2(ch)"]

        # エネルギーごとのCountsを合計
        grouped = df.groupby("sum_energy")["Counts"].sum().reset_index()

        # スライダー（表示範囲調整）
        x_min = int(grouped["sum_energy"].min())
        x_max = int(grouped["sum_energy"].max())
        x_range = st.slider("表示するエネルギー和の範囲", x_min, x_max, (x_min, x_max))

        filtered = grouped[(grouped["sum_energy"] >= x_range[0]) & (grouped["sum_energy"] <= x_range[1])]

        # 描画
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(filtered["sum_energy"], filtered["Counts"], width=1.0, color='steelblue')
        ax.set_xlabel("sum_energy (CH1 + CH2)")
        ax.set_ylabel("counts")
        st.pyplot(fig)

        # 保存とダウンロード
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="この1Dヒストグラム画像を保存",
            data=buf.getvalue(),
            file_name="1d_histogram.png",
            mime="image/png"
        )

