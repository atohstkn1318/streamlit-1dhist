import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import io

# タイトル
st.title("1Dヒストグラムと代表ピーク検出（上位4つ）")

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv", "xlsx"])

if uploaded_file is not None:
    # ファイル読み込み
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, skiprows=45)
    else:
        df = pd.read_excel(uploaded_file, skiprows=45)

    # 必要なカラムがあるか確認
    if {"CH1(ch)", "CH2(ch)", "Counts"}.issubset(df.columns):
        # エネルギー和を計算して集計
        df["sum_energy"] = df["CH1(ch)"] + df["CH2(ch)"]
        grouped = df.groupby("sum_energy")["Counts"].sum().reset_index()

        # プロット用データ
        x = grouped["sum_energy"].values
        y = grouped["Counts"].values

        # 全体を一度だけスムージング＆ピーク検出
        y_smooth = savgol_filter(y, window_length=21, polyorder=3)
        peaks, props = find_peaks(y_smooth, prominence=0)
        peak_xs = x[peaks]
        peak_ys = y_smooth[peaks]
        peak_proms = props["prominences"]
        scores = peak_ys * peak_proms

        # 上位4つ（グローバル）を選出
        if len(scores) >= 4:
            top_idx = np.argsort(scores)[-4:][::-1]
        else:
            top_idx = np.argsort(scores)[::-1]
        top_xs = peak_xs[top_idx]
        top_ys = peak_ys[top_idx]
        top_proms = peak_proms[top_idx] if len(scores) > 0 else np.array([])
        top_scores = scores[top_idx] if len(scores) > 0 else np.array([])

        # テキスト出力用
        peak_info_text = ""
        for i, idx in enumerate(top_idx):
            peak_info_text += (
                f"Peak {i+1}: sum_energy = {top_xs[i]:.1f}, "
                f"height = {top_ys[i]:.1f}, "
                f"prom = {top_proms[i]:.1f}, "
                f"score = {top_scores[i]:.1f}\n"
            )
        if len(scores) == 0:
            peak_info_text = "⚠ No peak found in the entire range.\n"

        # スライダー：表示範囲（ズーム）設定
        x_min = int(x.min())
        x_max = int(x.max())
        x_range = st.slider("表示するエネルギー和の範囲", x_min, x_max, (x_min, x_max))

        # 描画
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x, y, width=1.0, color='steelblue', label="Original")
        ax.plot(x, y_smooth, color='orange', label="Smoothed", linewidth=1.5)
        if len(top_xs) > 0:
            ax.scatter(top_xs, top_ys, color="red", s=50, edgecolors="black", label="Top 4 Peaks")
        ax.set_xlabel("sum_energy (CH1 + CH2)")
        ax.set_ylabel("counts")
        ax.set_xlim(x_range[0], x_range[1])
        ax.legend()
        st.pyplot(fig)

        # テキスト出力
        st.markdown("### 検出された代表ピーク（全体から上位4つ）")
        st.text(peak_info_text)

        # 画像保存とダウンロード
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button(
            label="この1Dヒストグラム画像を保存",
            data=buf.getvalue(),
            file_name="1d_histogram_top4_peaks.png",
            mime="image/png"
        )
    else:
        st.error("CSV/Excel に必要なカラム（CH1(ch), CH2(ch), Counts）が含まれていません。")
