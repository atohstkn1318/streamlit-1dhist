import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import io

# タイトル
st.title("1Dヒストグラムと代表ピーク検出（sum_energy < 400）")

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
        x_range = st.slider("表示するエネルギー和の範囲", 0, x_max, (0, x_max))

        filtered = grouped[(grouped["sum_energy"] >= x_range[0]) & (grouped["sum_energy"] <= x_range[1])]

        # データ準備
        x = grouped["sum_energy"].values
        y = grouped["Counts"].values
        y_smooth = savgol_filter(y, window_length=21, polyorder=3)

        # ピーク検出（緩く検出）
        peaks, props = find_peaks(y_smooth, prominence=0)
        valid = x[peaks] < 400
        peak_xs = x[peaks][valid]
        peak_ys = y_smooth[peaks][valid]
        peak_proms = props["prominences"][valid]

        # スコア計算（height × prominence）
        scores = peak_ys * peak_proms

        # 上位2つ選出
        top_xs, top_ys = np.array([]), np.array([])
        peak_info_text = ""
        if len(scores) >= 2:
            top_indices = np.argsort(scores)[-2:][::-1]
            top_xs = peak_xs[top_indices]
            top_ys = peak_ys[top_indices]
            for i in range(2):
                peak_info_text += f"📌 Peak {i+1}: sum_energy = {top_xs[i]:.1f}, height = {top_ys[i]:.1f}, prom = {peak_proms[top_indices[i]]:.1f}\n"
        elif len(scores) == 1:
            top_xs = np.array([peak_xs[0]])
            top_ys = np.array([peak_ys[0]])
            peak_info_text += f"📌 Only 1 peak: sum_energy = {top_xs[0]:.1f}, height = {top_ys[0]:.1f}, prom = {peak_proms[0]:.1f}\n"
        else:
            peak_info_text += "⚠ No peak found below sum_energy = 400.\n"

        # 描画
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(filtered["sum_energy"], filtered["Counts"], width=1.0, color='steelblue', label="Original")
        ax.plot(x, y_smooth, color='orange', label="Smoothed", linewidth=1.5)
        ax.axvline(400, color="gray", linestyle="--", label="Threshold: 400")
        if len(top_xs) > 0:
            ax.scatter(top_xs, top_ys, color="red", s=100, edgecolors="black", label="Top Peaks")
        ax.set_xlabel("sum_energy (CH1 + CH2)")
        ax.set_ylabel("counts")
        ax.set_title("1Dヒストグラムとトップ2ピーク")
        ax.legend()
        st.pyplot(fig)

        # テキスト出力
        st.markdown("### 検出された代表ピーク")
        st.text(peak_info_text)

        # 画像保存とダウンロード
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="この1Dヒストグラム画像を保存",
            data=buf.getvalue(),
            file_name="1d_histogram_with_peaks.png",
            mime="image/png"
        )
