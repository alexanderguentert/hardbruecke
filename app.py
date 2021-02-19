import pickle
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.tree import DecisionTreeRegressor

import dash
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output

# parameters
names = {
    'Ost-Süd total': 0,
    'Ost-Sd total': 0,   # alias, as seen in api query
    'Ost-Nord total': 1,
    'Ost-SBB total': 2,
    'West-SBB total': 3,
    'West-Süd total': 4,
    'West-Sd total': 4,  # alias, as seen in api query
    'Ost-VBZ Total': 5,
    'West-Nord total': 6,
    'West-VBZ total': 7,
}

XList = [
  'hour',
  'weekday',
  'minute',
  'month',
  'direction_cat',
  'name_cat',
]
y = 'count'

# functions


def data_preparation(df, names):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index(['Timestamp', 'Name']).stack().reset_index()
    df = df.rename(columns={'level_2': 'direction', 0: 'count'})

    df['hour'] = df['Timestamp'].dt.hour
    df['weekday'] = df['Timestamp'].dt.weekday
    df['minute'] = df['Timestamp'].dt.minute
    df['month'] = df['Timestamp'].dt.month

    df['day'] = pd.to_datetime(df['Timestamp'].dt.date)

    df['direction_cat'] = df['direction'].replace({'In': 0, 'Out': 1})

    df['name_cat'] = df['Name'].replace(names)

    return df


def plot_day(df, day, name, regressor, XList):
    df_filter = df[(df['day']==day) & (df['Name']==name)].copy()
    df_filter['prediction'] = regressor.predict(df_filter[XList])
    melted = df_filter.melt(id_vars=['Timestamp', 'direction', 'Name'], value_vars=['count', 'prediction'])
    fig = px.line(melted, x='Timestamp', y='value', color='variable', facet_row='direction', title=name)
    return fig


# load data
filepath = './data/frequenzen_hardbruecke_2020.zip'
hb = pd.read_csv(filepath, compression='zip')

hb2 = data_preparation(hb, names)

filename_model = './models/DecisionTreeRegressor.sav'

regressor = pickle.load(open(filename_model, 'rb'))

location_names = sorted(hb2['Name'].unique())

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Voraussagen', value='tab-1'),
        dcc.Tab(label='Historische Daten', value='tab-2'),
    ]),
    html.Div(id='tabs-example-content')
])


@app.callback(Output('tabs-example-content', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Tab content 1'),
            dcc.DatePickerSingle(
                id='date-picker',
                display_format='YYYY-MM-DD',
                min_date_allowed=hb2['Timestamp'].min(),
                max_date_allowed=hb2['Timestamp'].max(),
                initial_visible_month=hb2['Timestamp'].max(),
                date=hb2['Timestamp'].max(),
            ),
            dcc.RadioItems(
                id='category',
                options=[{'label': i, 'value': i} for i in location_names],
                value=location_names[0],
                labelStyle={'display': 'inline-block'}
            ),
            html.Div(dcc.Graph(id='plot_prediction_day', figure=plot_day(hb2, '2020-07-23', 'Ost-Nord total', regressor, XList))),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2')
        ])


if __name__ == '__main__':
    app.run_server(debug=True)
