import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import detrend

st.set_page_config(page_title="Norma Euclidiana FaceMesh", layout="wide")

st.title("Análise das normas euclidianas dos pontos 3D")

arquivo = st.file_uploader("Carregue o arquivo principal CSV", type=["csv", "txt"])
arquivo_parametros = st.file_uploader("Carregue o arquivo de parâmetros normativos", type=["csv", "txt"])

if arquivo is not None and arquivo_parametros is not None:

    df = pd.read_csv(arquivo, sep=None, engine="python", encoding="utf-8-sig")
    parametros = pd.read_csv(arquivo_parametros, sep=None, engine="python", encoding="utf-8-sig")

    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    parametros.columns = parametros.columns.str.strip().str.replace("\ufeff", "", regex=False)

    #st.subheader("Pré-visualização dos dados principais")
    #st.dataframe(df.head())

    #st.subheader("Pré-visualização dos parâmetros normativos")
    #st.dataframe(parametros.head())

    #st.write("Colunas detectadas no arquivo normativo:")
    #st.write(parametros.columns.tolist())

    if not any(col.lower() == "level" for col in parametros.columns):
        st.error(f"Coluna 'Level' não encontrada. Colunas detectadas: {parametros.columns.tolist()}")
        st.stop()

    level_col = [col for col in parametros.columns if col.lower() == "level"][0]

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

    unidade_tempo = True

    janela_ms = 1000

    usar_detrend = True

    tempo = df[tempo_col].astype(float).values

    janela_autozero = janela_ms / 1000 if unidade_tempo == "segundos" else janela_ms

    tempo_inicial = tempo[0]
    idx_autozero = tempo <= tempo_inicial + janela_autozero

    if np.sum(idx_autozero) < 2:
        st.warning("A janela de autozero possui menos de 2 amostras.")
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

        if usar_detrend:
            x_proc = detrend(x_autozero)
            y_proc = detrend(y_autozero)
            z_proc = detrend(z_autozero)
        else:
            x_proc = x_autozero
            y_proc = y_autozero
            z_proc = z_autozero

        norma = np.sqrt(x_proc**2 + y_proc**2 + z_proc**2)

        normas[f"Pt{i}_norma"] = norma
        rms_pontos.append(np.sqrt(np.mean(norma**2)))

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

    st.subheader("Comparação normativa por amostras da norma")

    classes_desvio = []
    descricoes_desvio = []

    contagens = {
        "n_<2DP": [],
        "n_2a3DP": [],
        "n_3a4DP": [],
        "n_4a5DP": [],
        "n_>=5DP": []
    }

    percentuais = {
        "%_<2DP": [],
        "%_2a3DP": [],
        "%_3a4DP": [],
        "%_4a5DP": [],
        "%_>=5DP": []
    }

    for i in range(n_pontos):

        ponto_col = f"Pt{i}_norma"

        if ponto_col not in parametros.columns:
            classes_desvio.append(0)
            descricoes_desvio.append("Sem referência")

            for k in contagens:
                contagens[k].append(np.nan)
            for k in percentuais:
                percentuais[k].append(np.nan)

            continue

        try:
            lim_2sd = float(parametros.loc[parametros[level_col].astype(str).str.strip() == "Mean + 2 SD", ponto_col].values[0])
            lim_3sd = float(parametros.loc[parametros[level_col].astype(str).str.strip() == "Mean + 3 SD", ponto_col].values[0])
            lim_4sd = float(parametros.loc[parametros[level_col].astype(str).str.strip() == "Mean + 4 SD", ponto_col].values[0])
            lim_5sd = float(parametros.loc[parametros[level_col].astype(str).str.strip() == "Mean + 5 SD", ponto_col].values[0])
        except Exception:
            classes_desvio.append(0)
            descricoes_desvio.append("Erro")

            for k in contagens:
                contagens[k].append(np.nan)
            for k in percentuais:
                percentuais[k].append(np.nan)

            continue

        serie_norma = normas[ponto_col].values
        serie_norma = serie_norma[~np.isnan(serie_norma)]
        n_total = len(serie_norma)

        if n_total == 0:
            classes_desvio.append(0)
            descricoes_desvio.append("Sem dados")

            for k in contagens:
                contagens[k].append(np.nan)
            for k in percentuais:
                percentuais[k].append(np.nan)

            continue

        n_menor_2dp = np.sum(serie_norma < lim_2sd)
        n_2a3dp = np.sum((serie_norma >= lim_2sd) & (serie_norma < lim_3sd))
        n_3a4dp = np.sum((serie_norma >= lim_3sd) & (serie_norma < lim_4sd))
        n_4a5dp = np.sum((serie_norma >= lim_4sd) & (serie_norma < lim_5sd))
        n_maior_5dp = np.sum(serie_norma >= lim_5sd)

        lista_contagens = [
            n_menor_2dp,
            n_2a3dp,
            n_3a4dp,
            n_4a5dp,
            n_maior_5dp
        ]

        classe = int(np.argmax(lista_contagens))

        descricoes = [
            "< 2 DP",
            "2–3 DP",
            "3–4 DP",
            "4–5 DP",
            "≥ 5 DP"
        ]

        classes_desvio.append(classe)
        descricoes_desvio.append(descricoes[classe])

        contagens["n_<2DP"].append(n_menor_2dp)
        contagens["n_2a3DP"].append(n_2a3dp)
        contagens["n_3a4DP"].append(n_3a4dp)
        contagens["n_4a5DP"].append(n_4a5dp)
        contagens["n_>=5DP"].append(n_maior_5dp)

        percentuais["%_<2DP"].append(100 * n_menor_2dp / n_total)
        percentuais["%_2a3DP"].append(100 * n_2a3dp / n_total)
        percentuais["%_3a4DP"].append(100 * n_3a4dp / n_total)
        percentuais["%_4a5DP"].append(100 * n_4a5dp / n_total)
        percentuais["%_>=5DP"].append(100 * n_maior_5dp / n_total)

    rms_df["classe_desvio"] = classes_desvio
    rms_df["interpretação"] = descricoes_desvio

    for k, v in contagens.items():
        rms_df[k] = v

    for k, v in percentuais.items():
        rms_df[k] = v

    st.subheader("Máscara colorida pelo estrato dominante")

    mostrar_numeros = st.checkbox("Mostrar número dos pontos", value=False)

    texto_pontos = [str(i) for i in range(n_pontos)] if mostrar_numeros else None
    modo = "markers+text" if mostrar_numeros else "markers"

    corescale_custom = [
        [0.00, "blue"],
        [0.25, "green"],
        [0.50, "yellow"],
        [0.75, "orange"],
        [1.00, "red"]
    ]

    fig_mask = go.Figure()

    fig_mask.add_trace(
        go.Scatter(
            x=rms_df["x_medio"],
            y=rms_df["y_medio"],
            mode=modo,
            marker=dict(
                size=10,
                color=rms_df["classe_desvio"],
                cmin=0,
                cmax=4,
                colorscale=corescale_custom,
                colorbar=dict(
                    title="Estrato dominante",
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=["<2DP", "2–3DP", "3–4DP", "4–5DP", "≥5DP"]
                ),
                showscale=True
            ),
            text=texto_pontos,
            textposition="top center",
            customdata=np.stack(
                [
                    rms_df["ponto"],
                    rms_df["RMS_norma"],
                    rms_df["interpretação"],
                    rms_df["%_<2DP"],
                    rms_df["%_2a3DP"],
                    rms_df["%_3a4DP"],
                    rms_df["%_4a5DP"],
                    rms_df["%_>=5DP"]
                ],
                axis=-1
            ),
            hovertemplate=(
                "Ponto: %{customdata[0]}<br>"
                "RMS: %{customdata[1]:.6f}<br>"
                "Estrato dominante: %{customdata[2]}<br>"
                "% <2DP: %{customdata[3]:.1f}%<br>"
                "% 2–3DP: %{customdata[4]:.1f}%<br>"
                "% 3–4DP: %{customdata[5]:.1f}%<br>"
                "% 4–5DP: %{customdata[6]:.1f}%<br>"
                "% ≥5DP: %{customdata[7]:.1f}%<br>"
                "X: %{x:.4f}<br>"
                "Y: %{y:.4f}<br>"
                "<extra></extra>"
            )
        )
    )

    fig_mask.update_layout(
        height=750,
        xaxis_title="X médio",
        yaxis_title="Y médio",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=20, t=40, b=40)
    )

    fig_mask.update_yaxes(scaleanchor="x", scaleratio=1)

    st.plotly_chart(fig_mask, use_container_width=True)

    st.subheader("Resumo dos estratos dominantes")
    resumo_classes = rms_df["interpretação"].value_counts().reset_index()
    resumo_classes.columns = ["Estrato dominante", "Número de pontos"]
    st.dataframe(resumo_classes)

    st.subheader("Visualização temporal")

    pontos = [f"Pt{i}_norma" for i in range(n_pontos)]

    pontos_selecionados = st.multiselect(
        "Escolha os pontos",
        pontos,
        default=["Pt0_norma"]
    )

    usar_tempo = st.radio("Eixo X", ["Tempo", "Frame"], horizontal=True)

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

        y_label = "Norma após autozero + detrend" if usar_detrend else "Norma após autozero"

        fig.update_layout(
            height=650,
            xaxis_title=eixo_x,
            yaxis_title=y_label,
            margin=dict(l=40, r=20, t=40, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

    #st.subheader("Ranking dos pontos")

    rms_df_ordenado = rms_df.sort_values("RMS_norma", ascending=False)
    #st.dataframe(rms_df_ordenado)

    #st.subheader("Tabela das normas")
    #st.dataframe(normas)

    csv_normas = normas.to_csv(index=False).encode("utf-8")
    csv_rms = rms_df_ordenado.to_csv(index=False).encode("utf-8")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Baixar normas",
            data=csv_normas,
            file_name="normas.csv",
            mime="text/csv"
        )

    with col2:
        st.download_button(
            label="Baixar resumo por ponto",
            data=csv_rms,
            file_name="resumo_pontos.csv",
            mime="text/csv"
        )

else:
    st.info("Carregue o arquivo principal e o arquivo normativo.")
