import plotly.graph_objects as go
import plotly.express as px


def fan_chart(summary, title, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary['date'], y=summary['q95'], mode='lines', line=dict(width=0), showlegend=False,
        hovertemplate='Upper 95% bound: %{y:.2f}<br>Date: %{x}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=summary['date'], y=summary['q05'], mode='lines', fill='tonexty', name='Wider uncertainty band: 5-95%',
        line=dict(width=0), fillcolor='rgba(231,76,60,0.20)',
        hovertemplate='Lower 5% bound: %{y:.2f}<br>This wide band means most possible forecast values fall here.<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=summary['date'], y=summary['q75'], mode='lines', line=dict(width=0), showlegend=False,
        hovertemplate='Upper 75% bound: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=summary['date'], y=summary['q25'], mode='lines', fill='tonexty', name='Core uncertainty band: 25-75%',
        line=dict(width=0), fillcolor='rgba(241,196,15,0.30)',
        hovertemplate='Lower 25% bound: %{y:.2f}<br>This inner band shows the most common forecast range.<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=summary['date'], y=summary['mean'], mode='lines+markers', name='Average forecast', line=dict(width=3),
        customdata=summary[['std', 'q05', 'q95', 'cv']].round(3),
        hovertemplate=(
            '<b>Average forecast</b>: %{y:.2f}<br>'
            'Date: %{x}<br>'
            'Spread: %{customdata[0]}<br>'
            'Likely range: %{customdata[1]} to %{customdata[2]}<br>'
            'Relative uncertainty: %{customdata[3]}<br>'
            '<extra></extra>'
        )
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='x unified',
        legend_title_text='What the lines mean'
    )
    return fig


def interval_prediction_chart(summary, interval_df, title, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=interval_df['date'], y=interval_df['q95_pred'], mode='lines', line=dict(width=0), showlegend=False,
        hovertemplate='Predicted upper bound: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=interval_df['date'], y=interval_df['q05_pred'], mode='lines', fill='tonexty',
        name='ML likely range: 5-95%', line=dict(width=0), fillcolor='rgba(52,152,219,0.22)',
        hovertemplate='Predicted lower bound: %{y:.2f}<br>The model expects most values inside this range.<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=interval_df['date'], y=interval_df['q50_pred'], mode='lines', name='ML middle estimate', line=dict(width=3),
        hovertemplate='ML middle estimate: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=summary['date'], y=summary['mean'], mode='markers', name='Ensemble average',
        hovertemplate='Observed/simulated ensemble average: %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=y_label, template='plotly_white', hovermode='x unified')
    return fig


def uncertainty_map(map_df, title, value_label):
    """Geospatial risk map: color = risk intensity, marker size = uncertainty spread."""
    fig = px.scatter_mapbox(
        map_df,
        lat='latitude',
        lon='longitude',
        color='mean',
        size='std_scaled',
        hover_name='location_id',
        hover_data={
            'country': True,
            'risk_label': True,
            'plain_language': True,
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
        zoom=4,
        height=560,
        title=title,
        labels={
            'mean': value_label,
            'std': 'uncertainty spread',
            'q05': 'low likely value',
            'q95': 'high likely value',
            'prob_exceedance': 'chance above threshold',
            'risk_label': 'risk level',
            'plain_language': 'simple explanation'
        }
    )
    fig.update_layout(mapbox_style='open-street-map', margin=dict(l=0, r=0, t=45, b=0))
    return fig


def traffic_light_card(label, value, risk, confidence, risk_color, confidence_color, explanation=''):
    return f"""
    <div title='{explanation}' style='border-radius:14px;padding:18px;background:#fff;box-shadow:0 2px 12px rgba(0,0,0,.12);margin-bottom:14px;'>
      <h3 style='margin-bottom:4px'>{label}</h3>
      <p style='font-size:28px;font-weight:700;margin:0'>{value}</p>
      <p style='font-size:13px;color:#555;margin-top:6px'>{explanation}</p>
      <div title='Risk tells how severe the potential impact is.' style='margin-top:12px;padding:10px;border-radius:10px;background:{risk_color};color:white;font-weight:700'>Risk: {risk}</div>
      <div title='Confidence tells how much the ensemble members agree.' style='margin-top:8px;padding:10px;border-radius:10px;background:{confidence_color};color:white;font-weight:700'>Confidence: {confidence}</div>
    </div>
    """
