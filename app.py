import pickle
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.tree import DecisionTreeRegressor

import dash
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output


# functions


def data_preparation(df, names):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index(['Timestamp', 'Name']).stack().reset_index()
    df = df.rename(columns={'level_2': 'direction', 0: 'count'})
    df['direction'] = df['direction'].astype('category')

    df['year'] = df['Timestamp'].dt.year

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
    melted['value'] = melted['value'].astype(float)
    fig = px.line(melted, x='Timestamp', y='value', color='variable', facet_row='direction', title=name)
    return fig


def create_date_range(date, freq='5min'):
    return pd.date_range('{} 00:00:00'.format(date), '{} 23:55:00'.format(date), freq=freq)


def create_future_df(name, date):
    future = pd.DataFrame({'Timestamp': create_date_range(date), })
    future['In'] = 0
    future['Out'] = 0
    # future = future.set_index('Timestamp').stack().reset_index().drop(columns=0)
    # future = future.rename(columns={'level_1':'direction'})
    future['Name'] = name
    return future


def download_from_api(date, resource):
    url_day = """https://data.stadt-zuerich.ch/api/3/action/datastore_search_sql?""" \
        """sql=SELECT%20%22Timestamp%22,%22Name%22,%22In%22,%22Out%22%20""" \
        """from%20%22{resource}%22""" \
        """where%20%22Timestamp%22::TIMESTAMP::DATE=%27{day}%27%20;"""
    df = pd.read_json(url_day.format(day=date, resource=resource)).loc['records', 'result']
    df = pd.DataFrame.from_dict(df)  # .drop(columns=['_full_text','_id'])
    if df.empty:
        data_available = False
    else:
        data_available = True
    return data_available, df


def get_from_api(url):
    """
    Download from api. For historical data
    :param url:
    :return:
    """
    df = pd.read_json(url).loc['records', 'result']
    df = pd.DataFrame.from_dict(df)  # .drop(columns=['_full_text','_id'])
    if df.empty:
        data_available = False
    else:
        data_available = True
    return data_available, df


def plot_time_group(resource, frequency, aggregation):
    url_time = """https://data.stadt-zuerich.ch/api/3/action/datastore_search_sql?""" \
               """sql=SELECT%20DATE_TRUNC(%27{frequency}%27,%22Timestamp%22::TIMESTAMP)%20AS%20timestamp,""" \
               """{aggregation}(%22In%22::INT)%20AS%20in,{aggregation}(%22Out%22::INT)%20as%20out%20""" \
               """from%20%22{resource}%22%20GROUP%20BY%201%20ORDER%20BY%201"""
    data_available, time_group_df = get_from_api(
        url_time.format(resource=resource, frequency=frequency, aggregation=aggregation))
    if data_available:
        time_group_df['timestamp'] = pd.to_datetime(time_group_df['timestamp'])
        time_group_df = time_group_df.set_index('timestamp').stack().reset_index()
        time_group_df = time_group_df.rename(columns={'level_1': 'direction', 0: aggregation})
        time_group_df[aggregation] = time_group_df[aggregation].astype(float)

        return px.line(time_group_df, x='timestamp', y=aggregation, color='direction', title='Zeitliche Verteilung',
                       color_discrete_sequence=px.colors.qualitative.Dark2)
    else:
        return px.line(title='Keine Daten verfügbar')


def plot_name_group(resource, aggregation):
    url_name = """https://data.stadt-zuerich.ch/api/3/action/datastore_search_sql?""" \
        """sql=SELECT%20%22Name%22,{aggregation}(%22In%22::INT)%20AS%20in,{aggregation}(%22Out%22::INT)%20as%20out%20""" \
        """from%20%22{resource}%22%20GROUP%20BY%201%20ORDER%20BY%201"""
    data_available, name_group_df = get_from_api(url_name.format(resource=resource,
                                                                 aggregation=aggregation))
    if data_available:
        name_group_df = name_group_df.set_index('Name').stack().reset_index()
        name_group_df = name_group_df.rename(columns={'level_1': 'direction', 0: aggregation})
        name_group_df[aggregation] = name_group_df[aggregation].astype(float)
        return px.bar(name_group_df, x='Name', y=aggregation,
                      color='direction', barmode='group', title='Verteilung je Ort',
                      color_discrete_sequence=px.colors.qualitative.Dark2)
    else:
        return px.bar(title='Keine Daten verfügbar')


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
    'year',
    'hour',
    'weekday',
    'minute',
    'month',
    'direction_cat',
    'name_cat',
]
y = 'count'

resource_api = {
    '2022': """a0c89c3e-72e7-4cbe-965a-efa16b3ecd5f""",
    '2021': """2f27e464-4910-46bf-817b-a9bac19f86f3""",
    '2020': """5baeaf58-9af2-4a39-a357-9063ca450893""",
}

# parameters
freq_dict = {
    'Woche': 'WEEK',
    'Monat': 'MONTH',
    'Quartal': 'QUARTER',
    'Tag': 'DAY',
}
agg_dict = {
    'Mittelwert': 'AVG',
    # 'Median': 'median',
    'Minimum': 'MIN',
    'Maximum': 'MAX',
    'Summe': 'SUM',
    'Anzahl': 'COUNT',
}

location_names = [x for x in names.keys() if 'ü' not in x]  # api has problems with ü

yesterday = (pd.to_datetime('today') - pd.Timedelta('1 days')).strftime('%Y-%m-%d')
dates_max = yesterday

# load model
filename_model = './models/DecisionTreeRegressor.sav'
regressor = pickle.load(open(filename_model, 'rb'))


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.title = 'Hardbrücke Fahrgastfrequenzen'

server = app.server

app.layout = html.Div([
    html.H1('Fahrgastfrequenzen an der VBZ-Haltestelle Hardbrücke'),
    dcc.Markdown(children='''Die Anzahl der Fahrgäste an einer Haltestelle unterliegt bestimmten Regelmässigkeiten, auch in Zeiten einer Pandemie.
            Die VBZ stellen die Fahrgastfrequenzen der Haltestelle Hardbrücke in Zürich offen zur Verfügung.
            Wie viele Personen die Haltestelle, aus welcher Richtung betreten oder verlassen haben, lässt sich so detailliert nachvollziehen.
            Mit Machine Learning können regelmässige Muster in den Daten erkannt und damit auch Prognosen für die Zukunft erstellt werden.
            Sehen Sie hier die Prognosen eines Modells, das mit den Daten des Jahres 2020 trainiert wurde. 
            Sofern die tatsächlichen Frequenzen zur Verfügung stehen, können sie direkt mit den Prognosen verglichen werden.
            '''),
    dcc.Markdown(children='''Daten- und Bildquelle: [https://data.stadt-zuerich.ch/dataset/vbz_frequenzen_hardbruecke]
                            (https://data.stadt-zuerich.ch/dataset/vbz_frequenzen_hardbruecke)
                            Quellcode: [https://github.com/alexanderguentert/hardbruecke]
                            (https://github.com/alexanderguentert/hardbruecke)'''),
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Voraussagen', value='tab-1'),
        dcc.Tab(label='Historische Daten', value='tab-2'),
        dcc.Tab(label='Situationsplan', value='tab-3'),
    ]),
    html.Div(id='tabs-example-content')
])


@app.callback(Output('tabs-example-content', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Prognosen (rot) und tatsächliche Fahrgastfrequenzen (blau)'),
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
            html.Div(dcc.Graph(id='plot_prediction_day', )),
        ])
    elif tab == 'tab-2':
        return html.Div([
            dcc.Markdown(children='Analysieren Sie hier die Daten, mit denen der Prognosealgorithmus trainiert wurde'),
            html.Div(dcc.Markdown(children='Jahr:'), style={'width': '15%', 'display': 'inline-block'}),
            html.Div(
                dcc.RadioItems(
                    id='resource',
                    options=[{'label': i, 'value': i} for i in sorted(resource_api.keys())],
                    value=list(sorted(resource_api.keys()))[0],
                    labelStyle={'display': 'inline-block'}
                ), style={'width': '85%', 'display': 'inline-block'}),
            html.Div(dcc.Markdown(children='Zeitliche Gruppierung:'), style={'width': '15%', 'display': 'inline-block'}),
            html.Div(
                dcc.RadioItems(
                    id='time_group',
                    options=[{'label': i, 'value': i} for i in freq_dict],
                    value=list(freq_dict.keys())[0],
                    labelStyle={'display': 'inline-block'}
                ), style={'width': '85%', 'display': 'inline-block'}),
            html.Div(dcc.Markdown(children='Aggregation:'), style={'width': '15%', 'display': 'inline-block'}),
            html.Div(
                dcc.RadioItems(
                    id='agg_value',
                    options=[{'label': i, 'value': i} for i in agg_dict],
                    value=list(agg_dict.keys())[0],
                    labelStyle={'display': 'inline-block'}
                ), style={'width': '85%', 'display': 'inline-block'}),
            html.Div(dcc.Graph(id='plot_time_group', )),
            html.Div(dcc.Graph(id='plot_name_group', )),
        ])
    elif tab == 'tab-3':
        return html.Div([
            dcc.Markdown(children="""
Hier einige Informationen zu den angegebenen Richtungen von: 
[https://data.stadt-zuerich.ch/dataset/vbz_frequenzen_hardbruecke]
            (https://data.stadt-zuerich.ch/dataset/vbz_frequenzen_hardbruecke)
> Die Daten werden richtungsgetrennt ausgewiesen.
> * "Ost" bezeichnet die Haltestelle Hardbrücke mit Fahrtrichtung Schiffbau.
> * "West" bezeichnet die Haltestelle Hardbrücke mit Fahrtrichtung Hardplatz.

> Zudem werden die Frequenzen an vier verschiedenen Zähllinien erfasst.
> * "Süd" bezeichnet die Zähllinie im Süden der Haltestelle. Es werden alle Personen erfasst, die vom Hardplatz kommend die VBZ-Haltestelle betreten bzw. die Haltestelle in diese Richtung verlassen.
> * "Nord" bezeichnet die Zähllinie im Norden der Haltestelle. Es werden alle Persoenen erfasst, die via Fussgängerrampe im Norden die VBZ-Haltestelle betreten/verlassen.
> * "SBB", bezeichnet die Zähllinie mit Zugang zur S-Bahnstation Hardbrücke. Es werden alle Personen erfasst, die von der SBB kommend die VBZ-Haltestelle betreten bzw. die Haltestelle in diese Richtung verlassen.
> * "VBZ", bezeichnet die Zähllinie mit den VBZ-Frequenzen. Es werden alle Persoenen erfasst, die von einem VBZ-Fahrzeug die VBZ-Haltestelle betreten bzw. die Haltestelle durch Einstieg in ein VBZ-Fahrzeug verlassen.

Situation an West und Ostkante:"""),
            html.Img(
                src='https://www.stadt-zuerich.ch/content/dam/stzh/portal/Deutsch/OGD/Bilder/ckan/zu-daten/vbz_Situation_Westkante.PNG',
                width=1000),
            html.Img(
                src='https://www.stadt-zuerich.ch/content/dam/stzh/portal/Deutsch/OGD/Bilder/ckan/zu-daten/vbz_Situation_Ostkante.PNG',
                width=1000),
        ])

# callbacks for graphs in tabs
# tab-1
@app.callback(
    dash.dependencies.Output('plot_prediction_day', 'figure'),
    [dash.dependencies.Input('date-picker', 'date'),
     dash.dependencies.Input('location_names', 'value')]
)
def update_plots_tab1(date, location_name):
    # check if historical data is available
    # if False:  # dates_min <= date <= dates_max:
    #     plot_df = hb2
    year = date[0:4]
    if year in resource_api:
        resource_year = resource_api[year]
        # data from api
        data_available, df_api = download_from_api(date, resource_year)
        if data_available:
            plot_df = data_preparation(df_api, names)
        else:
            future = data_preparation(create_future_df(location_name, date), names)
            future['count'] = np.nan  # no real data available
            plot_df = future
    else:
        future = data_preparation(create_future_df(location_name, date), names)
        future['count'] = np.nan  # no real data available
        plot_df = future
    return plot_day(plot_df, date, location_name, regressor, XList)


# callbacks for graphs in tabs
# tab-2
@app.callback(
    [dash.dependencies.Output('plot_time_group', 'figure'),
     dash.dependencies.Output('plot_name_group', 'figure'),
     ],
    [dash.dependencies.Input('resource', 'value'),
     dash.dependencies.Input('time_group', 'value'),
     dash.dependencies.Input('agg_value', 'value'),
     ]
)
def update_plots_tab2(resource, time_group, agg_value):
    return plot_time_group(resource_api[resource], freq_dict[time_group], agg_dict[agg_value]), plot_name_group(resource_api['2021'], agg_dict[agg_value])


if __name__ == '__main__':

    # run the app
    app.run_server(debug=True)
