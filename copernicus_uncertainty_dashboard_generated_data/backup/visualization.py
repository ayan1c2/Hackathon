import plotly.graph_objects as go
import plotly.express as px


def fan_chart(summary, title, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary['date'], y=summary['q95'], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=summary['date'], y=summary['q05'], mode='lines', fill='tonexty', name='5-95% ensemble range', line=dict(width=0), fillcolor='rgba(231,76,60,0.20)'))
    fig.add_trace(go.Scatter(x=summary['date'], y=summary['q75'], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=summary['date'], y=summary['q25'], mode='lines', fill='tonexty', name='25-75% ensemble range', line=dict(width=0), fillcolor='rgba(241,196,15,0.30)'))
    fig.add_trace(go.Scatter(x=summary['date'], y=summary['mean'], mode='lines+markers', name='Ensemble mean', line=dict(width=3)))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=y_label, template='plotly_white')
    return fig


def interval_prediction_chart(summary, interval_df, title, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=interval_df['date'], y=interval_df['q95_pred'], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=interval_df['date'], y=interval_df['q05_pred'], mode='lines', fill='tonexty', name='ML 90% prediction interval', line=dict(width=0), fillcolor='rgba(52,152,219,0.22)'))
    fig.add_trace(go.Scatter(x=interval_df['date'], y=interval_df['q50_pred'], mode='lines', name='ML median prediction', line=dict(width=3)))
    fig.add_trace(go.Scatter(x=summary['date'], y=summary['mean'], mode='markers', name='Ensemble mean'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=y_label, template='plotly_white')
    return fig


def uncertainty_map(map_df, title, value_label):
    """Geospatial risk map: color = risk, marker size = uncertainty spread."""
    fig = px.scatter_mapbox(
        map_df,
        lat='latitude',
        lon='longitude',
        color='mean',
        size='std_scaled',
        hover_name='location_id',
        hover_data={
            'mean': ':.2f',
            'std': ':.2f',
            'q05': ':.2f',
            'q95': ':.2f',
            'prob_exceedance': ':.0%',
            'confidence': True,
            'latitude': ':.3f',
            'longitude': ':.3f',
            'std_scaled': False
        },
        color_continuous_scale='YlOrRd',
        zoom=6,
        height=540,
        title=title,
        labels={'mean': value_label}
    )
    fig.update_layout(mapbox_style='open-street-map', margin=dict(l=0, r=0, t=45, b=0))
    return fig


def traffic_light_card(label, value, risk, confidence, risk_color, confidence_color):
    return f"""
    <div style='border-radius:14px;padding:18px;background:#fff;box-shadow:0 2px 12px rgba(0,0,0,.12);margin-bottom:14px;'>
      <h3 style='margin-bottom:4px'>{label}</h3>
      <p style='font-size:28px;font-weight:700;margin:0'>{value}</p>
      <div style='margin-top:12px;padding:10px;border-radius:10px;background:{risk_color};color:white;font-weight:700'>Risk: {risk}</div>
      <div style='margin-top:8px;padding:10px;border-radius:10px;background:{confidence_color};color:white;font-weight:700'>Confidence: {confidence}</div>
    </div>
    """
