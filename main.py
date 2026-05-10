import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import detrend

st.set_page_config(page_title="Norma Euclidiana FaceMesh", layout="wide")

st.title("Análise das normas euclidianas dos pontos 3D")

arquivo = st.file_uploader("Carregue o arquivo CSV", type=["csv", "txt"])

if arquivo is not None:

    df = pd.read_csv(arquivo, sep=None, engine="python")

    st.subheader("Pré-visualização dos dados")
    st.dataframe(df.head())

    frame_col = df.columns[0]
    tempo_col = df.columns[1]

    df[tempo_col] = pd.to_numeric(df[tempo_col], errors="coerce")
    df = df.dropna(subset=[tempo_col])

    coord_cols = df.columns[2:]

    if len(coord_cols) % 3 != 0:
        st.error("O número de colunas de coordenadas não é múltiplo de 3.")
        st.stop()

    n_pontos = len(coord_cols) // 3

    st.success(f"Foram identificados {n_pontos} pontos tridimensionais.")

    unidade_tempo = st.radio(
        "Unidade da coluna de tempo",
        ["segundos", "milissegundos"],
        horizontal=True
    )

    janela_ms = st.number_input(
        "Janela para autozero (ms)",
        min_value=1,
        max_value=1000,
        value=50,
        step=1
    )

    tempo = df[tempo_col].astype(float).values

    if unidade_tempo == "segundos":
        janela_autozero = janela_ms / 1000
    else:
        janela_autozero = janela_ms

    tempo_inicial = tempo[0]
    idx_autozero = tempo <= tempo_inicial + janela_autozero

    if np.sum(idx_autozero) < 2:
        st.warning(
            "A janela de autozero tem menos de 2 amostras. "
            "Verifique a unidade do tempo ou aumente a janela."
        )
        idx_autozero = np.arange(min(5, len(tempo)))

    normas = pd.DataFrame()
    normas[frame_col] = df[frame_col].values
    normas[tempo_col] = tempo

    rms_pontos = []
    x_medios = []
    y_medios = []
    z_medios = []

    for i in range(n_pontos):

        x_original = pd.to_numeric(df.iloc[:, 2 + 3*i], errors="coerce").values
        y_original = pd.to_numeric(df.iloc[:, 2 + 3*i + 1], errors="coerce").values
        z_original = pd.to_numeric(df.iloc[:, 2 + 3*i + 2], errors="coerce").values

        x_zero = np.nanmean(x_original[idx_autozero])
        y_zero = np.nanmean(y_original[idx_autozero])
        z_zero = np.nanmean(z_original[idx_autozero])

        x_autozero = x_original - x_zero
        y_autozero = y_original - y_zero
        z_autozero = z_original - z_zero

        x_autozero = pd.Series(x_autozero).interpolate().bfill().ffill().values
        y_autozero = pd.Series(y_autozero).interpolate().bfill().ffill().values
        z_autozero = pd.Series(z_autozero).interpolate().bfill().ffill().values

        x = detrend(x_autozero)
        y = detrend(y_autozero)
        z = detrend(z_autozero)

        norma = np.sqrt(x**2 + y**2 + z**2)

        normas[f"Pt{i}_norma"] = norma

        rms = np.sqrt(np.mean(norma**2))
        rms_pontos.append(rms)

        x_medios.append(np.nanmean(x_original))
        y_medios.append(np.nanmean(y_original))
        z_medios.append(np.nanmean(z_original))

    rms_df = pd.DataFrame({
        "ponto": [f"Pt{i}" for i in range(n_pontos)],
        "x_medio": x_medios,
        "y_medio": y_medios,
        "z_medio": z_medios,
        "RMS_norma": rms_pontos
    })

    st.subheader("Máscara dos pontos colorida pelo RMS da norma")

    mostrar_numeros = st.checkbox("Mostrar número dos pontos na máscara", value=False)

    texto_pontos = [str(i) for i in range(n_pontos)] if mostrar_numeros else None
    modo = "markers+text" if mostrar_numeros else "markers"

    fig_mask = go.Figure()

    fig_mask.add_trace(
        go.Scatter(
            x=rms_df["x_medio"],
            y=rms_df["y_medio"],
            mode=modo,
            marker=dict(
                size=8,
                color=rms_df["RMS_norma"],
                colorscale="Turbo",
                colorbar=dict(title="RMS"),
                showscale=True
            ),
            text=texto_pontos,
            textposition="top center",
            hovertext=rms_df["ponto"],
            hovertemplate=(
                "Ponto: %{hovertext}<br>"
                "X médio: %{x:.4f}<br>"
                "Y médio: %{y:.4f}<br>"
                "RMS: %{marker.color:.6f}<extra></extra>"
            )
        )
    )

    fig_mask.update_layout(
        height=700,
        xaxis_title="X médio",
        yaxis_title="Y médio",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=20, t=40, b=40)
    )

    fig_mask.update_yaxes(scaleanchor="x", scaleratio=1)

    st.plotly_chart(fig_mask, use_container_width=True)

    st.subheader("Selecionar pontos para visualização temporal")

    pontos = [f"Pt{i}_norma" for i in range(n_pontos)]

    pontos_selecionados = st.multiselect(
        "Escolha os pontos",
        pontos,
        default=["Pt0_norma"]
    )

    usar_tempo = st.radio(
        "Eixo X",
        ["Tempo", "Frame"],
        horizontal=True
    )

    eixo_x = tempo_col if usar_tempo == "Tempo" else frame_col

    if pontos_selecionados:

        fig = go.Figure()

        for ponto in pontos_selecionados:
            fig.add_trace(
                go.Scatter(
                    x=normas[eixo_x],
                    y=normas[ponto],
                    mode="lines",
                    name=ponto
                )
            )

        fig.update_layout(
            height=650,
            xaxis_title=eixo_x,
            yaxis_title="Norma euclidiana após autozero e detrend",
            margin=dict(l=40, r=20, t=40, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Ranking dos pontos por RMS")

    rms_df_ordenado = rms_df.sort_values("RMS_norma", ascending=False)
    st.dataframe(rms_df_ordenado)

    st.subheader("Tabela das normas calculadas")
    st.dataframe(normas)

    csv_normas = normas.to_csv(index=False).encode("utf-8")
    csv_rms = rms_df_ordenado.to_csv(index=False).encode("utf-8")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Baixar normas em CSV",
            data=csv_normas,
            file_name="normas_euclidianas_pontos.csv",
            mime="text/csv"
        )

    with col2:
        st.download_button(
            label="Baixar RMS por ponto em CSV",
            data=csv_rms,
            file_name="rms_norma_por_ponto.csv",
            mime="text/csv"
        )
