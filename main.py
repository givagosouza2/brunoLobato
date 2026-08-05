import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import detrend

st.set_page_config(page_title="Norma Euclidiana FaceMesh", layout="wide")

st.title("Análise dos deslocamentos euclidianos dos pontos 2D")

arquivo = st.file_uploader("Carregue o arquivo principal CSV", type=["csv", "txt"])
arquivo_parametros = "Parameters.csv"

if arquivo is not None and arquivo_parametros is not None:

    df = pd.read_csv(arquivo, sep=None, engine="python", encoding="utf-8-sig")
    parametros = pd.read_csv(arquivo_parametros, sep=None, engine="python", encoding="utf-8-sig")

    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    parametros.columns = parametros.columns.str.strip().str.replace("\ufeff", "", regex=False)

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

    st.subheader("Normalização geométrica")

    usar_procrustes = st.checkbox(
        "Remover translação, rotação e escala global da face (Procrustes)",
        value=True,
        help=(
            "Alinha cada frame a uma configuração facial de referência. "
            "Isso reduz a influência da distância face-câmera e dos movimentos rígidos da cabeça."
        )
    )

    unidade_tempo = st.selectbox("Unidade da coluna de tempo", ["segundos", "milissegundos"], index=0)
    janela_ms = st.number_input("Janela inicial de referência (ms)", min_value=100, value=1000, step=100)
    usar_detrend = st.checkbox("Aplicar detrend após o alinhamento", value=False)

    pontos_ancora_texto = st.text_input(
        "Pontos usados no alinhamento (opcional)",
        value="",
        help=(
            "Deixe vazio para usar todos os pontos. Para reduzir a influência das expressões, "
            "informe índices de pontos relativamente estáveis separados por vírgula."
        )
    )

    tempo = df[tempo_col].astype(float).values
    janela_referencia = janela_ms / 1000.0 if unidade_tempo == "segundos" else float(janela_ms)
    tempo_inicial = tempo[0]
    idx_referencia = tempo <= tempo_inicial + janela_referencia

    if np.sum(idx_referencia) < 2:
        st.warning("A janela de referência possui menos de 2 amostras; serão usados os primeiros frames disponíveis.")
        idx_referencia = np.arange(min(5, len(tempo)))

    def interpolar_coluna(valores):
        return pd.Series(valores, dtype=float).interpolate(limit_direction="both").bfill().ffill().values

    def alinhar_similaridade_2d(configuracao, referencia, indices_ancora):
        """Alinha configuração à referência, removendo translação, rotação e escala isotrópica."""
        movel = configuracao[indices_ancora]
        fixa = referencia[indices_ancora]

        centro_movel = np.mean(movel, axis=0)
        centro_fixa = np.mean(fixa, axis=0)
        movel_c = movel - centro_movel
        fixa_c = fixa - centro_fixa

        norma_movel = np.linalg.norm(movel_c)
        norma_fixa = np.linalg.norm(fixa_c)

        if norma_movel <= np.finfo(float).eps or norma_fixa <= np.finfo(float).eps:
            return configuracao - centro_movel + centro_fixa, np.nan

        h = movel_c.T @ fixa_c
        u, _, vt = np.linalg.svd(h)
        rotacao = u @ vt

        # Impede reflexão da face durante o alinhamento.
        if np.linalg.det(rotacao) < 0:
            vt[-1, :] *= -1
            rotacao = u @ vt

        movel_rot = movel_c @ rotacao
        escala = np.sum(movel_rot * fixa_c) / np.sum(movel_c ** 2)
        alinhada = (configuracao - centro_movel) @ rotacao * escala + centro_fixa
        return alinhada, escala

    # Matriz: frames x pontos x coordenadas (X, Y, Z).
    coordenadas = np.empty((len(df), n_pontos, 3), dtype=float)
    for i in range(n_pontos):
        for eixo in range(3):
            valores = pd.to_numeric(df.iloc[:, 2 + 3*i + eixo], errors="coerce").values
            coordenadas[:, i, eixo] = interpolar_coluna(valores)

    try:
        if pontos_ancora_texto.strip():
            indices_ancora = sorted({int(v.strip()) for v in pontos_ancora_texto.split(",") if v.strip()})
            if len(indices_ancora) < 3:
                raise ValueError("Informe pelo menos três pontos de alinhamento.")
            if min(indices_ancora) < 0 or max(indices_ancora) >= n_pontos:
                raise ValueError(f"Os índices devem estar entre 0 e {n_pontos - 1}.")
        else:
            indices_ancora = list(range(n_pontos))
    except ValueError as e:
        st.error(f"Pontos de alinhamento inválidos: {e}")
        st.stop()

    # A referência é a forma média dos frames iniciais, centralizada e normalizada.
    referencia_xy = np.nanmean(coordenadas[idx_referencia, :, :2], axis=0)
    centro_ref = np.mean(referencia_xy[indices_ancora], axis=0)
    referencia_xy = referencia_xy - centro_ref
    tamanho_ref = np.linalg.norm(referencia_xy[indices_ancora])
    if tamanho_ref <= np.finfo(float).eps:
        st.error("Não foi possível construir uma referência facial válida.")
        st.stop()
    referencia_xy = referencia_xy / tamanho_ref

    coordenadas_alinhadas = coordenadas.copy()
    escalas = np.ones(len(df), dtype=float)

    if usar_procrustes:
        for frame in range(len(df)):
            alinhada, escala = alinhar_similaridade_2d(
                coordenadas[frame, :, :2], referencia_xy, indices_ancora
            )
            coordenadas_alinhadas[frame, :, :2] = alinhada
            escalas[frame] = escala
    else:
        coordenadas_alinhadas[:, :, :2] = coordenadas[:, :, :2]

    normas = pd.DataFrame({frame_col: df[frame_col].values, tempo_col: tempo})
    normas["fator_escala_procrustes"] = escalas

    rms_pontos = []
    x_medios = np.mean(referencia_xy[:, 0]) + referencia_xy[:, 0]
    y_medios = np.mean(referencia_xy[:, 1]) + referencia_xy[:, 1]
    z_medios = np.nanmean(coordenadas[:, :, 2], axis=0)

    for i in range(n_pontos):
        x_proc = coordenadas_alinhadas[:, i, 0]
        y_proc = coordenadas_alinhadas[:, i, 1]

        if usar_detrend:
            x_proc = detrend(x_proc)
            y_proc = detrend(y_proc)

        dx = np.diff(x_proc, prepend=x_proc[0])
        dy = np.diff(y_proc, prepend=y_proc[0])
        distancia_delta = np.sqrt(dx**2 + dy**2)

        normas[f"Pt{i}_norma"] = distancia_delta
        rms_pontos.append(np.sqrt(np.mean(distancia_delta**2)))

    rms_df = pd.DataFrame({
        "ponto": [f"Pt{i}" for i in range(n_pontos)],
        "x_medio": x_medios,
        "y_medio": y_medios,
        "z_medio": z_medios,
        "RMS_distancia_delta": rms_pontos
    })

    if usar_procrustes:
        st.info(
            "Os deslocamentos foram calculados após alinhamento de Procrustes 2D. "
            "Os limites do arquivo Parameters.csv precisam ter sido obtidos com o mesmo processamento."
        )

    st.subheader("Comparação normativa por amostras da distância euclidiana frame-a-frame")

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

        contagens_acima_basal = [
            n_2a3dp,
            n_3a4dp,
            n_4a5dp,
            n_maior_5dp
        ]

        if np.sum(contagens_acima_basal) == 0:
            classe = 0
        else:
            classe = int(np.argmax(contagens_acima_basal)) + 1

        descricoes = [
            "Basal: nenhum valor ≥ 2 DP",
            "2–3 DP dominante",
            "3–4 DP dominante",
            "4–5 DP dominante",
            "≥ 5 DP dominante"
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

    st.subheader("Máscara colorida pelo estrato dominante acima do basal")

    col_rotulo1, col_rotulo2, col_rotulo3 = st.columns(3)

    with col_rotulo1:
        mostrar_numeros = st.checkbox("Mostrar número dos pontos", value=False)

    with col_rotulo2:
        mostrar_valores = st.checkbox("Mostrar valor RMS ao lado dos pontos", value=False)

    with col_rotulo3:
        casas_decimais = st.number_input(
            "Casas decimais do valor",
            min_value=2,
            max_value=8,
            value=5,
            step=1,
            disabled=not mostrar_valores
        )

    texto_pontos = []
    for i, valor in enumerate(rms_df["RMS_distancia_delta"]):
        partes = []

        if mostrar_numeros:
            partes.append(f"Pt{i}")

        if mostrar_valores:
            partes.append(f"{valor:.{int(casas_decimais)}f}")

        texto_pontos.append(" | ".join(partes))

    exibir_rotulos = mostrar_numeros or mostrar_valores
    texto_pontos = texto_pontos if exibir_rotulos else None
    modo = "markers+text" if exibir_rotulos else "markers"

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
                    title="Classificação",
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=["Basal", "2–3DP", "3–4DP", "4–5DP", "≥5DP"]
                ),
                showscale=True
            ),
            text=texto_pontos,
            textposition="top center",
            customdata=np.stack(
                [
                    rms_df["ponto"],
                    rms_df["RMS_distancia_delta"],
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
                "RMS distância delta: %{customdata[1]:.6f}<br>"
                "Classificação: %{customdata[2]}<br>"
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

    st.subheader("Resumo das classificações")
    resumo_classes = rms_df["interpretação"].value_counts().reset_index()
    resumo_classes.columns = ["Classificação", "Número de pontos"]
    st.dataframe(resumo_classes)

    st.subheader("Visualização temporal")

    pontos = [f"Pt{i}_norma" for i in range(n_pontos)]

    pontos_selecionados = st.multiselect(
        "Escolha os pontos",
        pontos,
        default=["Pt0_norma"]
    )

    usar_tempo = st.radio("Eixo X", ["Tempo", "Frame"], horizontal=True)

    mostrar_linhas_niveis = st.checkbox(
        "Mostrar linhas horizontais dos níveis normativos",
        value=True
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

        if mostrar_linhas_niveis and len(pontos_selecionados) == 1:

            ponto_ref = pontos_selecionados[0]

            if ponto_ref in parametros.columns:

                try:
                    lim_2sd = float(
                        parametros.loc[
                            parametros[level_col].astype(str).str.strip() == "Mean + 2 SD",
                            ponto_ref
                        ].values[0]
                    )

                    lim_3sd = float(
                        parametros.loc[
                            parametros[level_col].astype(str).str.strip() == "Mean + 3 SD",
                            ponto_ref
                        ].values[0]
                    )

                    lim_4sd = float(
                        parametros.loc[
                            parametros[level_col].astype(str).str.strip() == "Mean + 4 SD",
                            ponto_ref
                        ].values[0]
                    )

                    lim_5sd = float(
                        parametros.loc[
                            parametros[level_col].astype(str).str.strip() == "Mean + 5 SD",
                            ponto_ref
                        ].values[0]
                    )

                    niveis = [
                        (lim_2sd, "Mean + 2 SD"),
                        (lim_3sd, "Mean + 3 SD"),
                        (lim_4sd, "Mean + 4 SD"),
                        (lim_5sd, "Mean + 5 SD")
                    ]

                    for valor, nome in niveis:
                        fig.add_hline(
                            y=valor,
                            line_dash="dash",
                            annotation_text=nome,
                            annotation_position="top right"
                        )

                except Exception as e:
                    st.warning(f"Não foi possível adicionar os níveis normativos: {e}")

            else:
                st.warning(f"O ponto {ponto_ref} não foi encontrado no arquivo normativo.")

        elif mostrar_linhas_niveis and len(pontos_selecionados) > 1:

            st.info(
                "As linhas normativas são mostradas apenas quando um único ponto é selecionado, "
                "pois cada ponto possui limites próprios."
            )

        y_label = "Distância euclidiana frame-a-frame após alinhamento de Procrustes" if usar_procrustes else "Distância euclidiana frame-a-frame"

        fig.update_layout(
            height=650,
            xaxis_title=eixo_x,
            yaxis_title=y_label,
            margin=dict(l=40, r=20, t=40, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

    rms_df_ordenado = rms_df.sort_values("RMS_distancia_delta", ascending=False)

    csv_normas = normas.to_csv(index=False).encode("utf-8")
    csv_rms = rms_df_ordenado.to_csv(index=False).encode("utf-8")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Baixar distâncias delta",
            data=csv_normas,
            file_name="distancias_delta.csv",
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
