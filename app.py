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
    'Ost-Sd total': 0,  # alias, as seen in api query
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
    df_filter = df[(df['day'] == day) & (df['Name'] == name)].copy()
    df_filter['prediction'] = regressor.predict(df_filter[XList])
    melted = df_filter.melt(id_vars=['Timestamp', 'direction', 'Name'], value_vars=['count', 'prediction'])
    fig = px.line(melted, x='Timestamp', y='value', color='variable', facet_row='direction', title=name)
    return fig


def create_date_range(date, freq='5min'):
    return pd.date_range('{} 00:00:00'.format(date), '{} 23:55:00'.format(date), freq=freq)

def create_future_df(name, date):
    future = pd.DataFrame({'Timestamp': create_date_range(date), })
    future['In'] = 0
    future['Out'] = 0
    #future = future.set_index('Timestamp').stack().reset_index().drop(columns=0)
    #future = future.rename(columns={'level_1':'direction'})
    future['Name'] = name
    return future

# load data
filepath = './data/frequenzen_hardbruecke_2020.zip'
hb = pd.read_csv(filepath, compression='zip')

hb2 = data_preparation(hb, names)
dates = hb2['day'].dt.strftime('%Y-%m-%d')
dates_min = dates.min()
dates_max = dates.max()

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
                # min_date_allowed=hb2['day'].min(),
                # max_date_allowed=hb2['day'].max(),
                initial_visible_month=dates_max,
                date=dates_max,
            ),
            dcc.RadioItems(
                id='location_names',
                options=[{'label': i, 'value': i} for i in location_names],
                value=location_names[0],
                labelStyle={'display': 'inline-block'}
            ),
            # html.Div(dcc.Graph(id='plot_prediction_day', figure=plot_day(hb2, '2020-07-23', 'Ost-Nord total', regressor, XList))),
            html.Div(dcc.Graph(id='plot_prediction_day', )),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2')
        ])


# callbacks for graphs in tabs
# tab-2
@app.callback(
    dash.dependencies.Output('plot_prediction_day', 'figure'),
    [dash.dependencies.Input('date-picker', 'date'),
     dash.dependencies.Input('location_names', 'value')]
)
def update_plots_tab2(date, location_name):
    # check if historical data is available
    if dates_min <= date <= dates_max:
        plot_df = hb2
    else:
        future = data_preparation(create_future_df(location_name, date), names)
        future['count'] = np.nan  # no real data available
        plot_df = future
    return plot_day(plot_df, date, location_name, regressor, XList)


if __name__ == '__main__':
    app.run_server(debug=True)
