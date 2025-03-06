import pandas as pd
import plotly.express as px

import plotly.io as pio
bg_color="plotly_dark" # https://plotly.com/python/templates/
pio.templates.default = bg_color # Establecer el tema oscuro por defecto

def mostrar_mc(mc,title=""):
    labels = None
    if mc.shape[0] == 3:
        labels = ["NEG","NEU","POS"]
    elif mc.shape[0] == 7:
        labels = ["others","joy","fear","anger","sadness","disgust","surprise"]
    mc_df = pd.DataFrame(
        mc,
        index=labels,
        columns=labels
    )
    fig = px.imshow(
        mc_df,
        text_auto=True,
        color_continuous_scale="ylorbr",
        title=f"Matriz de confusión {title}",
        labels=dict(color="Numero de muestras")
    )
    fig.update_layout(
        xaxis_title="Predicción",
        yaxis_title="Real",
        font=dict(size=14)
    )
    fig.show()