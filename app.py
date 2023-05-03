from dash import Dash, html, dcc, ctx
from dash.dependencies import Input, Output, State
import plotly.express as px
import xarray as xr
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO
import json
import numpy as np
import bambi as bmb
import arviz as az
import pandas as pd

addresses = pd.read_csv('dataa/addresses.csv')
urban_index_county = pd.read_csv('data/urban_index_county.csv')
pvi_county = pd.read_csv('data/pres_county_pvi_2.csv')
county_state = pd.read_csv('data/fips-by-state.csv')
states = pd.read_csv('data/state_codes.csv')
unemployment_county = pd.read_csv('data/unemployment.csv')
hh_income_county = pd.read_csv('data/hh_income.csv')
personal_income_county = pd.read_csv('data/personal_income.csv')
pa_midterms_county = pd.read_csv('data/pa_midterms_county.csv')
education_county = pd.read_csv('data/education_county.csv')
weather_county = pd.read_csv('data/weather_county_final.csv')

states_codes = ['AK', 'NY', 'DC', 'CO', 'HI', 'SD', 'VA']
states = states[~states['code'].isin(states_codes)]

def vote_outcome(x):
    if x > 0:
        return 'democrat'
    elif x < 0:
        return 'republican'
    else:
        return 'even'

urban_index_county = urban_index_county[~(urban_index_county['year'] == '2012-2016')]
pvi_county['county_fips'] = pvi_county['county_fips'].astype(int).apply(lambda x: '{0:0>5}'.format(x))
pvi_county['vote_outcome'] = pvi_county['vote_margin'].apply(lambda x: vote_outcome(x))
urban_index_county['county_fips'] = urban_index_county['county_fips'].astype(int).apply(lambda x: '{0:0>5}'.format(x))
unemployment_county['county_fips'] = unemployment_county['county_fips'].astype(int).apply(lambda x: '{0:0>5}'.format(x))
hh_income_county['county_fips'] = hh_income_county['county_fips'].astype(int).apply(lambda x: '{0:0>5}'.format(x))
personal_income_county['county_fips'] = personal_income_county['county_fips'].astype(int).apply(lambda x: '{0:0>5}'.format(x))
pa_midterms_county['county_fips'] = pa_midterms_county['county_fips'].astype(int).apply(lambda x: '{0:0>5}'.format(x))
education_county['county_fips'] = education_county['county_fips'].astype(int).apply(lambda x: '{0:0>5}'.format(x))
weather_county['county_fips'] = weather_county['county_fips'].astype(int).apply(lambda x: '{0:0>5}'.format(x))
pa_midterms_county = pa_midterms_county.rename(columns={'vote_margin': 'midterm_vote_margin', 'year': 'midterm_year', 'county': 'name', 'dem_pct': 'midterm_dem_pct', 'rep_pct': 'midterm_rep_pct'})
pa_midterms_county = pa_midterms_county[pa_midterms_county['office'] == 'U.S. House']
pa_midterms_county[['midterm_dem_pct', 'midterm_rep_pct']] /= 100
pvi_county[['dem_pct', 'rep_pct']] /= 100

f1 = open('data/geojson_states.json')
geo_counties = json.load(f1)

state_names = dict(zip(states.code, states.state))
state_fips = dict(zip(states.code, states.st))
result_indicators = pvi_county.columns

urban_year = {
    2000: '2000',
    2004: '2005-2009',
    2008: '2006-2010',
    2012: '2010-2014',
    2016: '2014-2018',
    2020: '2016-2020'
}

urban_year_replace = {
    '2000': 2000,
    '2005-2009': 2004,
    '2006-2010': 2008,
    '2010-2014': 2012,
    '2014-2018': 2016,
    '2016-2020': 2020
}

prev_midterm = {
    2004: 2002,
    2008: 2006,
    2012: 2010,
    2016: 2014,
    2020: 2018
}

prev_midterm_replace = {
    2002: 2004,
    2006: 2008,
    2010: 2012,
    2014: 2016,
    2018: 2020
}

education_year = {
    2000: '2000',
    2004: '2000',
    2008: '2007-11',
    2012: '2007-11',
    2016: '2016-20',
    2020: '2016-20'
}

education_year_replace = {
    '2000': 2000,
    '2000': 2004,
    '2007-11': 2008,
    '2007-11': 2012,
    '2016-20': 2016,
    '2016-20': 2020
}

axis_titles = {
    'urbanindex': 'Urbanization Index',
    'educationindex': 'Education Index',
    'midterm_vote_margin' : 'Previous Midterm U.S House Election Results',
    'unemployment' : 'Unemployment Rate',
    'hh_income': 'Estimate of Median Household Income',
    'personal_income': 'Per Capita Personal Income',
    'feelslike': 'Temperature on Election Day',
    'precip': 'Precipitation on Election Day',
    'visibility': 'Visibility on Election Day',
    'humidity': 'Humidity on Election Day'
}

scale_vars = {
    'hh_income': 'scale(hh_income)',
    'personal_income': 'scale(personal_income)'
}

mesh_sizes = {
    'urbanindex': 0.1,
    'educationindex': 0.1,
    'midterm_vote_margin': 4,
    'personal_income': 1000,
    'hh_income': 1000,
    'unemployment': 0.1,
    'feelslike': 0.1,
    'precip': 0.01,
    'visibility': 0.1,
    'humidity': 2
}

predictor_options = [{'label': 'Urbanization Index', 'value': 'urbanindex'},
                     {'label': 'Education Index', 'value': 'educationindex'},
                     {'label': 'Previous Midterm U.S House Election Results', 'value': 'midterm_vote_margin'},
                     {'label': 'Unemployment Rate', 'value': 'unemployment'},
                     {'label': 'Estimate of Median Household Income', 'value': 'hh_income'},
                     {'label': 'Per Capita Personal Income', 'value': 'personal_income'},
                     {'label': 'Temperature on Election Day', 'value': 'feelslike'},
                     {'label': 'Precipitation on Election Day', 'value': 'precip'},
                     {'label': 'Visibility on Election Day', 'value': 'visibility'},
                     {'label': 'Humidity on Election Day', 'value': 'humidity'}]

logit_map_fig, logit_plot_fig, logit_calib_fig, linear_map_fig, linear_plot_fig, linear_accuracy_fig = None, None, None, None, None, None

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

dash_app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])
app = dash_app.server

theme_switch = ThemeSwitchAIO(
    aio_id="theme", themes=[dbc.themes.BOOTSTRAP, dbc.themes.DARKLY]
)

dash_app.layout = dbc.Container([
    dcc.Store(id='model-data'),
    dbc.Row([
        dbc.Col([
            html.H3(children='COUNTY-LEVEL VOTING ANALYSIS'),
            html.Div(['This application uses Bayesian machine learning models to analyze multiple predictors and provide insights into county-level election results.'])
        ], width=8),
        dbc.Col([
            theme_switch
        ], align="end", width=1, style={'display': 'flex', 'justify-content': 'flex-end'})
    ], style={'margin-top': '10px', 'margin-bottom': '15px'}, justify="between"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Select State'),
                        dcc.Dropdown(
                            id='state-dropdown',
                            options=[{'label': state, 'value': code} for code, state in state_names.items()],
                            value='PA')
                    ]),
                    dbc.Col([
                        dbc.Label('Select Election Years to Fit Model:'),
                        dcc.RangeSlider(id='year-range-slider', min=2000, max=2020, step=4, value=[2008, 2016], marks={
                            2000: {'label': '2000', 'style': {'color': '#77b0b1'}},
                            2004: {'label': '2004', 'style': {'color': '#77b0b1'}},
                            2008: {'label': '2008', 'style': {'color': '#77b0b1'}},
                            2012: {'label': '2012', 'style': {'color': '#77b0b1'}},
                            2016: {'label': '2016', 'style': {'color': '#77b0b1'}},
                            2020: {'label': '2020', 'style': {'color': '#77b0b1'}}        
                        })
                    ]),   
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Select Predictors'),
                        dcc.Dropdown(
                            options=predictor_options,
                            value='urbanindex',
                            id='preds-model',
                            multi=True)
                    ]),
                    dbc.Col([
                        dbc.Label('Select Election Year to Test:'),
                        dcc.Slider(min=2000, max=2020, step=4, id='year-slider', value=2020, marks={
                            2000: {'label': '2000', 'style': {'color': '#77b0b1'}},
                            2004: {'label': '2004', 'style': {'color': '#77b0b1'}},
                            2008: {'label': '2008', 'style': {'color': '#77b0b1'}},
                            2012: {'label': '2012', 'style': {'color': '#77b0b1'}},
                            2016: {'label': '2016', 'style': {'color': '#77b0b1'}},
                            2020: {'label': '2020', 'style': {'color': '#77b0b1'}}        
                        })
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Button('Analyze', id='fit-model', style={'float': 'right'})
                    ], style={'margin-top': '5px'})
                ])
            ], body=True, style={'margin-top': '0.75rem'}),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Bayesian Linear Regression Model'),
                dcc.Loading([
                     dbc.Row(id='linear-layout')
                ])
            ], body=True, style={'margin-top': '0.75rem'}),
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Bayesian Logistic Regression Model'),
                dcc.Loading([
                    dbc.Row(id='logit-layout')
                ])
            ], body=True, style={'margin-top': '0.75rem'}),
        ]),
    ])
], fluid=True, class_name='dbc p-5')


def mergeTrainData(state, train_years):
    join_type = "inner"

    train_data = pvi_county[(pvi_county['year'] >= train_years[0]) & (pvi_county['year'] <= train_years[1])]
    train_data['year'] = train_data['year'].astype(int)

    df = urban_index_county.copy()
    df['year'] = df['year'].replace(urban_year_replace)
    df['year'] = df['year'].astype(int)
    train_data = df[(df['year'] >= train_years[0]) & (df['year'] <= train_years[1])].merge(train_data, on=['year', 'county_fips'], how=join_type)

    df = education_county.copy()
    train_data = df[(df['year'] >= train_years[0]) & (df['year'] <= train_years[1])].merge(train_data, on=['year', 'county_fips', 'state'], how=join_type)

    train_data = unemployment_county[(unemployment_county['year'] >= train_years[0]) & (unemployment_county['year'] <= train_years[1])].merge(train_data, on=['year', 'county_fips', 'state'], how="outer")

    train_data = hh_income_county[(hh_income_county['year'] >= train_years[0]) & (hh_income_county['year'] <= train_years[1])].merge(train_data, on=['year', 'county_fips', 'state'], how="outer")

    train_data = personal_income_county[(personal_income_county['year'] >= train_years[0]) & (personal_income_county['year'] <= train_years[1])].merge(train_data, on=['year', 'county_fips', 'state'], how="outer")

    df = pa_midterms_county.copy()
    df['year'] = df['midterm_year'].copy()
    df['year'] = df['year'].replace(prev_midterm_replace)
    df['year'] = df['year'].astype(int)
    train_data = df[(df['year'] >= train_years[0]) & (df['year'] <= train_years[1])].merge(train_data, on=['year', 'county_fips'], how='outer')

    train_data = weather_county[(weather_county['year'] >= train_years[0]) & (weather_county['year'] <= train_years[1])].merge(train_data, on=['year', 'county_fips', 'state'], how=join_type)
    
    train_data = train_data.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y', 'name_x', 'Unnamed: 0_x', 'name_y', 'Unnamed: 0_y'])

    state_train = train_data[train_data['state'] == state]

    return state_train


def mergeTestData(state, test_year):
    join_type = "inner"

    test_data = pvi_county[pvi_county['year'] == test_year]
    test_data['year'] = test_data['year'].astype(int)

    df = urban_index_county.copy()
    df['year'] = df['year'].replace(urban_year_replace)
    df['year'] = df['year'].astype(int)
    test_data = df[df['year'] == test_year].merge(test_data, on=['year', 'county_fips'], how=join_type)

    df = education_county.copy()
    test_data = df[df['year'] == test_year].merge(test_data, on=['year', 'county_fips', 'state'], how=join_type)

    test_data = unemployment_county[unemployment_county['year'] == test_year].merge(test_data, on=['year', 'county_fips', 'state'], how="outer")

    test_data = hh_income_county[hh_income_county['year'] == test_year].merge(test_data, on=['year', 'county_fips', 'state'], how="outer")

    test_data = personal_income_county[personal_income_county['year'] == test_year].merge(test_data, on=['year', 'county_fips', 'state'], how="outer")

    df = pa_midterms_county.copy()
    df['year'] = df['midterm_year'].copy()
    df['year'] = df['year'].replace(prev_midterm_replace)
    df['year'] = df['year'].astype(int)
    test_data = df[df['year'] == test_year].merge(test_data, on=['year', 'county_fips'], how='outer')

    test_data = weather_county[weather_county['year'] == test_year].merge(test_data, on=['year', 'county_fips', 'state'], how=join_type)

    test_data = test_data.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y', 'name_x', 'Unnamed: 0_x', 'name_y', 'Unnamed: 0_y'])
    
    state_test = test_data[test_data['state'] == state]

    return state_test

def plot_accuracy(y_preds, ci_bound, test_data, dark_mode):
    if dark_mode:
        theme = 'white'
        line_color = 'rgba(0,150,255,0.7)'
    else:
        theme = 'black'
        line_color = 'rgba(0,0,255,0.5)'

    y_actual = test_data['vote_margin']
    y_pred = y_preds

    traces = []

    line_trace = go.Scatter(
        x=[min(y_actual), max(y_actual)],
        y=[min(y_actual), max(y_actual)],
        mode='lines',
        line=dict(color=theme, 
        width=2, 
        dash='dash'),
        hoverinfo='skip')

    traces.append(line_trace)
    print(test_data['name'])
    actual_trace = go.Scatter(x=y_actual, 
                          y=y_pred, 
                          mode='markers', 
                          line=dict(color='red'), 
                          name='Counties',
                          customdata=np.stack((test_data['name']), axis=-1), 
                          hovertemplate='<b>%{customdata}<b><br><b>Actual Vote Margin</b>: %{x:.2f}<br><b>Predicted Vote Margin</b>: %{y:.2f}')
    
    ci_bounds = ci_bound

    show_legend = True
    for i in range(len(y_actual)):
        custom_data = np.repeat(np.stack((ci_bounds[0][i], ci_bounds[1][i]), axis=-1).reshape(1, -1), 2, axis=0)
        trace = go.Scatter(
            x=[y_actual.iloc[i], y_actual.iloc[i]],
            y=[ci_bounds[0][i], ci_bounds[1][i]],
            mode='lines',
            customdata=custom_data,
            line=dict(color=line_color, width=1.5),
            legendgroup='ci', 
            name='94% Credible Interval',
            showlegend=show_legend,
            hovertemplate='<b>94% Credible Interval</b>: (%{customdata[0]:.2f}, %{customdata[1]:.2f})')
        show_legend = False
        traces.append(trace)

    traces.append(actual_trace)

    layout = go.Layout(
        title='Predicted vs. actual county-level vote shares',
        xaxis=dict(title='Actual Vote Share'),
        yaxis=dict(title='Predicted Vote Share'),
        showlegend=False
        # font=dict(size=18)
    )

    fig = go.Figure(data=traces, layout=layout)

    if dark_mode:
        fig = fig.update_layout(coloraxis_colorbar_x=0, template='plotly_dark')

    rmse = np.sqrt(np.mean((y_pred - y_actual)**2))
    mae = np.mean(np.abs((y_pred - y_actual)))

    fig.add_annotation(
        x=min(y_actual) + 20,
        y=max(y_actual),
        text=f"Average Error: {mae:.2f}",
        showarrow=False
        # font=dict(size=16),
    )

    return fig

def plot_calibration(pps_mean, test_data, dark_mode):
    print("Plotting calibration curve...")
    if dark_mode:
        print(dark_mode)
        theme = 'white'
        line_color = 'rgba(0,150,255,1.0)'
    else:
        print(dark_mode)
        theme = 'black'
        line_color = 'rgba(0,0,255,1.0)'
    print(test_data)
    y_actual = test_data['vote_outcome'].replace({'democrat': 1, 'republican': 0})
    y_pred = pps_mean

    print(y_pred)

    mask = test_data.isna()
    print(test_data.loc[mask.any(axis=1)])
    df = pd.DataFrame({'prob': y_pred, 'actual': y_actual})
    df['prob_bin'] = pd.cut(df.prob, bins=np.arange(0, 1.1, 0.1))

    print("Calculating calibration curve...")
    print(df)
    prob_means = df.groupby('prob_bin').prob.mean().reset_index()['prob'].dropna().to_numpy()
    print(prob_means)
    tp_rate = df.groupby('prob_bin').actual.mean().reset_index()['actual'].dropna().to_numpy()
    print(tp_rate)
    print("Finished calculating calibration curve...")
    traces = []

    print("Plotting perfect line...")
    line_trace = go.Scatter(
        x=np.arange(0,1.1, 0.1),
        y=np.arange(0,1.1, 0.1),
        mode='lines',
        line=dict(color=theme, width=2, dash='dash')
    )

    traces.append(line_trace)

    print("Plotting calibration curve 2...")
    calibration_trace = go.Scatter(
        x=prob_means,
        y=tp_rate,
        mode='markers+lines',
        line=dict(color=line_color, width=1.5),
        showlegend=True,
        hovertemplate='<b>Forecasted Probability</b>: %{x:.2f}<br><b>Actual Proportion</b>: %{y:.2f}')
    traces.append(calibration_trace)

    print("Plotting calibration curve 3...")
    layout = go.Layout(
        title='Calibration Plot',
        xaxis=dict(title='Forecasted Probability of County Voting Democrat'),
        yaxis=dict(title='Actual Proportion of Counties Voting Democratic'),
        showlegend=False
        # font=dict(size=18)
    )

    fig = go.Figure(data=traces, layout=layout)

    if dark_mode:
        fig = fig.update_layout(coloraxis_colorbar_x=0, template='plotly_dark')
    print("Finished plotting calibration curve.")
    return fig


def getModelResults(train_data, test_data, logit_formula, linear_formula, preds, dark_mode):
    # print("Preds: ")
    # print(preds)
    model_results = {}
    logit_results = {'map_data': {}}
    linear_results = {'map_data': {}}
    logit_model = bmb.Model(logit_formula, train_data, dropna=True, family="bernoulli")
    linear_model = bmb.Model(linear_formula, train_data, dropna=True)
    logit_fitted = logit_model.fit(draws=5000, chains=4)
    linear_fitted = linear_model.fit(draws=5000, chains=4)

    if type(preds) != list or len(preds) == 1:
        print("Single predictor")
        logit_preds = [preds] + ['vote_outcome', 'name']
        linear_preds = [preds] + ['vote_margin', 'name']
        logit_cap_data = test_data[logit_preds].reset_index().drop(columns=['index'])
        logit_cap_data = logit_cap_data.sort_values(preds)
        linear_cap_data = test_data[linear_preds].reset_index().drop(columns=['index'])
        linear_cap_data = linear_cap_data.sort_values(preds)
        model_results['main'] = preds
        logit_idata = logit_model.predict(idata=logit_fitted, data=logit_cap_data[preds].to_frame(), kind='pps', inplace=False)
        linear_idata = linear_model.predict(idata=linear_fitted, data=linear_cap_data[preds].to_frame(), kind='pps', inplace=False)
    else:
        print("Multiple predictors")
        logit_preds = preds + ['vote_outcome', 'name']
        linear_preds = preds + ['vote_margin', 'name']
        logit_cap_data = test_data[logit_preds].reset_index().drop(columns=['index'])
        logit_cap_data = logit_cap_data.sort_values(preds[0])
        linear_cap_data = test_data[linear_preds].reset_index().drop(columns=['index'])
        linear_cap_data = linear_cap_data.sort_values(preds[0])
        if (len(preds) > 1 and len(preds) < 3):
            X = test_data[preds]
            print(X)
            margin = 0
            x_min, x_max = X.iloc[:, 0].min() - margin, X.iloc[:, 0].max() + margin
            y_min, y_max = X.iloc[:, 1].min() - margin, X.iloc[:, 1].max() + margin
            xrange = np.arange(x_min, x_max, mesh_sizes[preds[0]])
            yrange = np.arange(y_min, y_max, mesh_sizes[preds[1]])
            xx, yy = np.meshgrid(xrange, yrange)
            cap_data = {preds[0]: xx.ravel(), preds[1]: yy.ravel()}
            cap_data = pd.DataFrame(cap_data)
            print("predicting 1..")
            logit_idata = logit_model.predict(idata=logit_fitted, data=cap_data, kind='pps', inplace=False)
            linear_idata = linear_model.predict(idata=linear_fitted, data=cap_data, kind='pps', inplace=False)

            print("calculating hdi values 1...")
            logit_z_hat = logit_idata.posterior['vote_outcome_mean']
            logit_z_hat_mean = logit_z_hat.mean(("chain", "draw")).to_numpy()
            logit_z_hat_bounds = logit_idata.posterior["vote_outcome_mean"].quantile((0.025, 0.975), ("chain", "draw")).to_numpy()
            logit_results['z_hat_mean'] = logit_z_hat_mean
            logit_results['z_hat_bounds'] = logit_z_hat_bounds

            print("calculating hdi values 2...")
            linear_z_hat = linear_idata.posterior['vote_margin_mean']
            linear_z_hat_mean = linear_z_hat.mean(("chain", "draw")).to_numpy()
            linear_z_hat_bounds = linear_idata.posterior["vote_margin_mean"].quantile((0.025, 0.975), ("chain", "draw")).to_numpy()
            linear_results['z_hat_mean'] = linear_z_hat_mean
            linear_results['z_hat_bounds'] = linear_z_hat_bounds

            logit_z_pp = az.extract(logit_idata.posterior_predictive, num_samples=50)["vote_outcome"].to_numpy().T
            linear_z_pp = az.extract(linear_idata.posterior_predictive, num_samples=50)["vote_margin"].to_numpy().T
            model_results['logit_z_pp'] = logit_z_pp
            model_results['linear_z_pp'] = linear_z_pp

            model_results['xrange'] = xrange
            model_results['yrange'] = yrange
            model_results['xx_shape'] = xx.shape

        cap_data = test_data[preds].reset_index().drop(columns=['index'])
        cap_data = cap_data.sort_values(preds[0])
        model_results['main'] = preds
        model_results['cap_data'] = cap_data
        print("predicting 2...")
        print(linear_formula)
        print(cap_data)
        logit_idata = logit_model.predict(idata=logit_fitted, data=cap_data, kind='pps', inplace=False)
        linear_idata = linear_model.predict(idata=linear_fitted, data=cap_data, kind='pps', inplace=False)
    
    print("predicting values for map 1...")
    logit_y_hat = logit_idata.posterior['vote_outcome_mean']
    logit_y_hat_mean = logit_y_hat.mean(("chain", "draw")).to_numpy()
    logit_y_hat_bounds = logit_idata.posterior["vote_outcome_mean"].quantile((0.025, 0.975), ("chain", "draw")).to_numpy()
    logit_results['y_hat_mean'] = logit_y_hat_mean
    logit_results['y_hat_bounds'] = logit_y_hat_bounds

    print("predicting values for map 2...")
    linear_y_hat = linear_idata.posterior['vote_margin_mean']
    linear_y_hat_mean = linear_y_hat.mean(("chain", "draw")).to_numpy()
    linear_y_hat_bounds = linear_idata.posterior["vote_margin_mean"].quantile((0.025, 0.975), ("chain", "draw")).to_numpy()
    linear_results['y_hat_mean'] = linear_y_hat_mean
    linear_results['y_hat_bounds'] = linear_y_hat_bounds

    logit_y_pp = az.extract(logit_idata.posterior_predictive, num_samples=50)["vote_outcome"].to_numpy().T
    linear_y_pp = az.extract(linear_idata.posterior_predictive, num_samples=50)["vote_margin"].to_numpy().T
    model_results['logit_y_pp'] = logit_y_pp
    model_results['linear_y_pp'] = linear_y_pp

    model_results['logit_cap_data'] = logit_cap_data
    model_results['linear_cap_data'] = linear_cap_data

    model_results['linear_idata'] = [linear_idata.posterior['vote_margin_mean'].mean(("chain", "draw")).to_numpy(), linear_idata.posterior_predictive['vote_margin'].quantile((0.025, 0.975), ("chain", "draw")).to_numpy()] 
    model_results['logit_idata'] = logit_idata.posterior['vote_outcome_mean'].mean(("chain", "draw")).to_numpy()

    model_results['logit_y'] = logit_cap_data['vote_outcome'].replace({'democrat': 1, 'republican': 0})
    model_results['linear_y'] = linear_cap_data['vote_margin']

    idx = 0
    for i, j in logit_cap_data.iterrows():
        county_fips = test_data.iloc[i]['county_fips']
        county_name = test_data.iloc[i]['name']
        logit_results['map_data'][county_fips] = {'predicted_mean': logit_y_hat_mean.item(idx), 'county_name': county_name}
        linear_results['map_data'][county_fips] = {'predicted_mean': linear_y_hat_mean.item(idx), 'county_name': county_name}
        idx += 1

    model_results['logit_model'] = logit_results
    model_results['linear_model'] = linear_results

    mask = logit_cap_data.isna()
    print(logit_cap_data.loc[mask.any(axis=1)])
    accuracy_results = plot_accuracy(model_results['linear_idata'][0], model_results['linear_idata'][1], linear_cap_data, dark_mode)
    calibration_results = plot_calibration(model_results['logit_idata'], logit_cap_data, dark_mode)

    return model_results, accuracy_results, calibration_results


def resultsDataframe(logit_results, linear_results):
    print("resultsDataframe()...")
    logit_df = pd.DataFrame(columns=['county_fips', 'county_name', 'dem_prob', 'rep_prob', 'dem_odds'])
    linear_df = pd.DataFrame(columns=['county_fips', 'county_name', 'vote_margin_mean'])
    print(logit_df)
    print(linear_df)
    for county_fips, obs in logit_results.items():
        dem_probability = obs['predicted_mean']
        rep_probability = 1 - dem_probability
        dem_odds = dem_probability / rep_probability
        logit_df = logit_df.append({'county_fips': county_fips, 'county_name': obs['county_name'], 'dem_prob': np.round(dem_probability, 2), 'rep_prob': np.round(rep_probability, 2), 'dem_odds': dem_odds}, ignore_index=True)
    for county_fips, obs in linear_results.items():
        vote_margin_mean = obs['predicted_mean']
        linear_df = linear_df.append({'county_fips': county_fips, 'county_name': obs['county_name'], 'vote_margin_mean': np.round(vote_margin_mean, 2)}, ignore_index=True)
    print("Finished resultsDataframe()...")
    return logit_df, linear_df


def plot_results(model_results, logit_df, linear_df, state, test_year, dark_mode):
    print("plotting results...")
    linear_plot_fig = None
    logit_plot_fig = None
    main = model_results['main']
    surface_plot = False
    scatter_plot = False
    model_plots = True
    if dark_mode:
        theme = '255,255,255'
        pps_color = 'rgba(0,150,255,0.2)'
    else:
        theme = '0,0,0'
        pps_color = 'rgba(0,0,255,0.2)'
    if type(main) == list:
        if len(main) == 1:
            main = main[0]
            scatter_plot = True
        elif len(main) > 1 and len(main) < 3:
            surface_plot = True
        elif len(main) > 2:
            model_plots = False
    else:
        scatter_plot = True

    print("plotting logit map...")
    print(logit_df)
    logit_map_fig = px.choropleth(logit_df, geojson=geo_counties[state], 
                            locations='county_fips', 
                            color='dem_prob', 
                            hover_data=['county_fips', 'county_name', 'dem_prob', 'rep_prob'],
                            color_continuous_scale="Blues",
                            range_color=(0, 1),
                            scope='usa',
                            title=f'Predicted Election Outcomes in {state_names[state]}, {test_year}',
                            labels={'dem_prob':'Democratic Probability',
                                    'county_fips': 'FIPS',
                                    'county_name': 'County Name',
                                    'rep_prob':'Republican Probability'}).update_geos(
                                        fitbounds="locations", visible=False)
    hovtemplate = '<b>%{customdata[1]}</b><br><b>Democratic Probability</b>: %{customdata[2]}<br><b>Republican Probability</b>: %{customdata[3]}<br>'
    logit_map_fig.update_traces(hovertemplate=hovtemplate)
    
    # logit_map_fig.update_layout(title_font=dict(size=24), font=dict(size=16))
    
    vm_min = linear_df['vote_margin_mean'].min()
    vm_max = linear_df['vote_margin_mean'].max()
    
    print("plotting linear map...")
    print(linear_df)
    linear_map_fig = px.choropleth(linear_df, geojson=geo_counties[state], 
                            locations='county_fips', 
                            color='vote_margin_mean', 
                            hover_data=['county_fips', 'county_name', 'vote_margin_mean'],
                            color_continuous_scale="RdBu",
                            range_color=(vm_min, vm_max),
                            scope='usa',
                            title=f'Predicted Election Outcomes in {state_names[state]}, {test_year}',
                            labels={'vote_margin_mean':'Mean Vote Margin',
                                    'county_fips': 'FIPS',
                                    'county_name': 'County Name'}).update_geos(
                                        fitbounds="locations", visible=False)
    hovtemplate = '<b>%{customdata[1]}</b><br><b>Predicted Vote Margin</b>: %{customdata[2]}<br>'
    linear_map_fig.update_traces(hovertemplate=hovtemplate)
    # linear_map_fig.update_layout(title_font=dict(size=26), font=dict(size=18))
    
    logit_cap_data = model_results['logit_cap_data']
    linear_cap_data = model_results['linear_cap_data']
    values_main = logit_cap_data[main]
    
    if surface_plot:
        logit_plot_fig = go.Figure(data=[
            go.Surface(x=model_results['xrange'], y=model_results['yrange'], z=model_results['logit_model']['z_hat_bounds'][0].reshape(model_results['xx_shape']), colorscale='Blues', opacity=0.9),
            go.Surface(x=model_results['xrange'], y=model_results['yrange'], z=model_results['logit_model']['z_hat_mean'].reshape(model_results['xx_shape']), colorscale='Blues'),
            go.Surface(x=model_results['xrange'], y=model_results['yrange'], z=model_results['logit_model']['z_hat_bounds'][1].reshape(model_results['xx_shape']), colorscale='Blues', opacity=0.9),
        ])
        logit_plot_fig.update_scenes(xaxis_title_text=axis_titles[main[0]],  
                  yaxis_title_text=axis_titles[main[1]],  
                  zaxis_title_text='Probability of Democrat')
        logit_plot_fig.update_layout(coloraxis=dict(colorbar=dict(x=-0.1)), legend=dict(x=1.1))
        
        linear_plot_fig = go.Figure(data=[
            go.Surface(x=model_results['xrange'], y=model_results['yrange'], z=model_results['linear_model']['z_hat_bounds'][0].reshape(model_results['xx_shape']), colorscale='RdBu', opacity=0.9),
            go.Surface(x=model_results['xrange'], y=model_results['yrange'], z=model_results['linear_model']['z_hat_mean'].reshape(model_results['xx_shape']), colorscale='RdBu'),
            go.Surface(x=model_results['xrange'], y=model_results['yrange'], z=model_results['linear_model']['z_hat_bounds'][1].reshape(model_results['xx_shape']), colorscale='RdBu', opacity=0.9),
        ])
        linear_plot_fig.update_scenes(xaxis_title_text=axis_titles[main[0]],  
                yaxis_title_text=axis_titles[main[1]],  
                zaxis_title_text='Vote Margin')
        linear_plot_fig.update_layout(coloraxis=dict(colorbar=dict(x=-0.1)), legend=dict(x=1.1))
        
        logit_z_pp = model_results['logit_z_pp']
        show_legend = True
        for z_values in logit_z_pp:
            pps_scatter = go.Scatter3d(x=logit_cap_data[model_results['main'][0]], y=logit_cap_data[model_results['main'][1]], z=z_values, mode='markers', legendgroup='logit_pps', name="Posterior Predictive Samples", showlegend=show_legend, marker=dict(color=pps_color, opacity=1))
            logit_plot_fig.add_trace(pps_scatter)
            show_legend = False
        test_data = go.Scatter3d(x=logit_cap_data[model_results['main'][0]], y=logit_cap_data[model_results['main'][1]], z=model_results['logit_y'], mode='markers', name="Observed Data", marker=dict(color='red'))
        logit_plot_fig.add_trace(test_data)

        linear_z_pp = model_results['linear_z_pp']
        show_legend = True
        for z_values in linear_z_pp:
            pps_scatter = go.Scatter3d(x=linear_cap_data[model_results['main'][0]], y=linear_cap_data[model_results['main'][1]], z=z_values, mode='markers', legendgroup='linear_pps', name="Posterior Predictive Samples", showlegend=show_legend, marker=dict(color=pps_color, opacity=1))
            linear_plot_fig.add_trace(pps_scatter)
            show_legend = False
        test_data = go.Scatter3d(x=linear_cap_data[model_results['main'][0]], y=linear_cap_data[model_results['main'][1]], z=model_results['linear_y'], mode='markers', name="Observed Data", marker=dict(color='red'))
        linear_plot_fig.add_trace(test_data)
    elif scatter_plot:
        logit_traces = []
        mean_line = go.Scatter(x=values_main, y=model_results['logit_model']['y_hat_mean'], mode="lines", line=dict(color='rgba('+ theme +',1)'), legendgroup='logit_cap', name="Mean")
        hdi_3 = go.Scatter(x=values_main, y=model_results['logit_model']['y_hat_bounds'][0], mode="lines", line=dict(color='rgba('+ theme +',0)'), legendgroup='logit_cap', showlegend=False, name="3% HDI")
        hdi_97 = go.Scatter(x=values_main, y=model_results['logit_model']['y_hat_bounds'][1], fill="tonexty", fillcolor='rgba(' + theme + ',0.4)', mode="lines", line=dict(color='rgba(0,0,0,0)'), legendgroup='logit_cap', name="Uncertainty in mean")
        logit_traces.append(hdi_3)
        logit_traces.append(hdi_97)
        logit_traces.append(mean_line)

        logit_y_pp = model_results['logit_y_pp']
        show_legend = True
        for y_values in logit_y_pp:
            pps_scatter = go.Scatter(x=values_main, y=y_values, mode='markers', legendgroup='logit_pps', showlegend=show_legend, name="Posterior Predictive Samples", marker=dict(color=pps_color))
            logit_traces.append(pps_scatter)
            show_legend = False
        test_data = go.Scatter(x=values_main, y=model_results['logit_y'], name='Observed Data', mode='markers', marker=dict(color='red'))
        logit_traces.append(test_data)

        logit_plot_fig = go.Figure(data=logit_traces)
        logit_plot_fig.update_xaxes(title_text=axis_titles[main])
        logit_plot_fig.update_yaxes(title_text="Probability of County Voting Democrat")

        linear_traces = []
        mean_line = go.Scatter(x=values_main, y=model_results['linear_model']['y_hat_mean'], mode="lines", line=dict(color='rgba('+ theme +',1)'), legendgroup='linear_cap', name="Mean")
        hdi_3 = go.Scatter(x=values_main, y=model_results['linear_model']['y_hat_bounds'][0], mode="lines", line=dict(color='rgba('+ theme +',0)'), legendgroup='linear_cap', showlegend=False, name="3% HDI")
        hdi_97 = go.Scatter(x=values_main, y=model_results['linear_model']['y_hat_bounds'][1], fill="tonexty", fillcolor='rgba(' + theme + ',0.4)', mode="lines", line=dict(color='rgba(0,0,0,0)'), legendgroup='linear_cap', name="Uncertainty in mean")
        linear_traces.append(hdi_3)
        linear_traces.append(hdi_97)
        linear_traces.append(mean_line)

        linear_y_pp = model_results['linear_y_pp']
        show_legend = True
        for y_values in linear_y_pp:
            pps_scatter = go.Scatter(x=values_main, y=y_values, mode='markers', legendgroup='linear_pps', showlegend=show_legend, name="Posterior Predictive Samples", marker=dict(color=pps_color))
            linear_traces.append(pps_scatter)
            show_legend = False
        test_data = go.Scatter(x=values_main, y=model_results['linear_y'], name='Observed Data', mode='markers', marker=dict(color='red'))
        linear_traces.append(test_data)

        linear_plot_fig = go.Figure(data=linear_traces)
        linear_plot_fig.update_xaxes(title_text=axis_titles[main])
        linear_plot_fig.update_yaxes(title_text="Vote Margin")
        # linear_plot_fig.update_layout(title='Posterior Regression Plot', title_font=dict(size=26), font=dict(size=18))

    
    # linear_plot_fig = linear_plot_fig.update_layout(autosize=False,
    #                                                 width=700, 
    #                                                 height=500,
    #                                                 margin=dict(l=65, r=50, b=65, t=90),
    #                                                 showlegend=False)
    # logit_plot_fig = logit_plot_fig.update_layout(autosize=False,
    #                                                 width=700, 
    #                                                 height=500,
    #                                                 margin=dict(l=65, r=50, b=65, t=90),
    #                                                 showlegend=False)
    if model_plots:
        linear_plot_fig.update_layout(height=900)
        logit_plot_fig.update_layout(height=900)

        if dark_mode:
            logit_map_fig.update_layout(template='plotly_dark')
            linear_map_fig.update_layout(template='plotly_dark')
            linear_plot_fig.update_layout(template='plotly_dark')
            logit_plot_fig.update_layout(template='plotly_dark')
    else:
        linear_plot_fig = None
        logit_plot_fig = None

        if dark_mode:
            logit_map_fig.update_layout(template='plotly_dark')
            linear_map_fig.update_layout(template='plotly_dark')

    return logit_map_fig, logit_plot_fig, linear_map_fig, linear_plot_fig

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, xr.DataArray):
        return obj.values.tolist()
    else:
        return obj
    
def convert_from_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_from_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return np.array(obj)
    # elif isinstance(obj, list):
    #     if len(obj) > 0 and isinstance(obj[0], list):
    #         return np.array(obj)
    #     else:
    #         return [convert_from_serializable(elem) for elem in obj]
    else:
        return obj
    

def create_layout(logit_map_fig, logit_plot_fig, logit_calib_fig, linear_map_fig, linear_plot_fig, linear_accuracy_fig):
    linear_layout = None
    logit_layout = None
    print("Creating layout...")
    # hoverData={'points': [{'customdata': ['42043']}]}
    if linear_plot_fig and linear_accuracy_fig and linear_map_fig:
        linear_layout1 = [
            dbc.Col([
                dcc.Loading([
                    dcc.Graph(id='linear-model-plot', figure=linear_plot_fig)
                ], id='loading_linear_plot')
            ], width=7),
            dbc.Col([
                dbc.Row([
                    dcc.Loading([
                        dcc.Graph(id='linear-map', figure=linear_map_fig)
                    ], id='loading_linear_map')
                ]),
                dbc.Row([
                    dcc.Loading([
                        dcc.Graph(id='linear-accuracy-plot', figure=linear_accuracy_fig)
                    ], id='loading_linear_accuracy')
                ])
            ])
        ]
        linear_layout = linear_layout1
    else:
        linear_layout2 = [
            dbc.Col([
                dbc.Row([
                    dcc.Loading([
                        dcc.Graph(id='linear-accuracy-plot', figure=linear_accuracy_fig)
                    ], id='loading_linear_accuracy')
                ])
            ]),
            dbc.Col([
                dcc.Loading([
                    dcc.Graph(id='linear-map', figure=linear_map_fig)
                ], id='loading_linear_map')
            ])
        ]
        linear_layout = linear_layout2

    if logit_plot_fig and logit_calib_fig and logit_map_fig:
        logit_layout1 = [
            dbc.Col([
                dcc.Loading([
                dcc.Graph(id='logit-model-plot', figure=logit_plot_fig)
                ], id='loading_logit_plot')
            ], width=7),
            dbc.Col([
                dbc.Row([
                    dcc.Loading([
                    dcc.Graph(id='logit-map', figure=logit_map_fig)
                    ], id='loading_logit_map')
                ]),
                dbc.Row([
                    dcc.Loading([
                        dcc.Graph(id='logit-calib-plot', figure=logit_calib_fig)
                    ], id='loading_logit_calib')
                ])
            ])
        ]
        logit_layout = logit_layout1
    else:
        logit_layout2 = [
            dbc.Col([
                dcc.Loading([
                    dcc.Graph(id='logit-calib-plot', figure=logit_calib_fig)
                ], id='loading_logit_calib')
            ]),
            dbc.Col([
                dcc.Loading([
                dcc.Graph(id='logit-map', figure=logit_map_fig)
                ], id='loading_logit_map')
            ])
        ]
        logit_layout = logit_layout2
    print("Finished layout")
    return linear_layout, logit_layout

@dash_app.callback(
    Output('linear-layout', 'children'),
    Output('logit-layout', 'children'),
    Output('model-data', 'data'),
    Input('fit-model', 'n_clicks'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"),
    State('state-dropdown', 'value'),
    State('preds-model', 'value'),
    State('year-range-slider', 'value'),
    State('year-slider', 'value'),
    State('model-data', 'data'),
    State(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def fit_model(btn, toggle, state, preds, train_years, test_year, model_data, theme):
    if "fit-model" == ctx.triggered_id or None == ctx.triggered_id:
        # print("theme variable: ")
        # print(theme)
        dark_mode = False if theme else True
        train_data = mergeTrainData(state, train_years)
        test_data = mergeTestData(state, test_year)
        logit_formula = 'vote_outcome[democrat] ~'
        linear_formula = 'vote_margin ~'
        if type(preds) == list and len(preds) == 1:
            preds = preds[0]
        
        if type(preds) == list:
            for i in range(len(preds)):
                if i == 0:
                    if preds[i] in scale_vars.keys():
                        logit_formula += ' ' + scale_vars[preds[i]]
                        linear_formula += ' ' + scale_vars[preds[i]]
                    else:
                        logit_formula += ' ' + preds[i]
                        linear_formula += ' ' + preds[i]
                else:
                    if preds[i] in scale_vars.keys():
                        logit_formula += ' + ' + scale_vars[preds[i]]   
                        linear_formula += ' + ' + scale_vars[preds[i]]  
                    else:
                        logit_formula += ' + ' + preds[i]   
                        linear_formula += ' + ' + preds[i]                 
        else:
            if preds in scale_vars.keys():
                logit_formula += ' ' + scale_vars[preds]
                linear_formula += ' ' + scale_vars[preds]
            else:
                logit_formula += ' ' + preds
                linear_formula += ' ' + preds
        
        model_results, linear_accuracy_fig, logit_calib_fig = getModelResults(train_data, test_data, logit_formula, linear_formula, preds, dark_mode)

        logit_df, linear_df = resultsDataframe(model_results['logit_model']['map_data'], model_results['linear_model']['map_data'])

        logit_map_fig, logit_plot_fig, linear_map_fig, linear_plot_fig = plot_results(model_results, logit_df, linear_df, state, test_year, dark_mode)

        linear_layout, logit_layout = create_layout(logit_map_fig, logit_plot_fig, logit_calib_fig, linear_map_fig, linear_plot_fig, linear_accuracy_fig)

        for key, value in model_results.items():
            if isinstance(value, pd.DataFrame):
                model_results[key] = value.to_dict()
            elif isinstance(value, pd.Series):
                model_results[key] = value.to_dict()
            elif isinstance(value, xr.DataArray):
                model_results[key] = value.values.tolist()
            elif isinstance(value, np.ndarray):
                model_results[key] = value.tolist()
            elif isinstance(value, az.InferenceData):
                model_results[key] = convert_to_serializable(value.to_dict())
            elif isinstance(value, dict):
                model_results[key] = convert_to_serializable(value)
            elif isinstance(value, list):
                model_results[key] = convert_to_serializable(value)
        
        model_data = {'model_results': model_results, 'logit_df': logit_df.to_dict(), 'linear_df': linear_df.to_dict(), 'state': state, 'test_year': test_year}

        return linear_layout, logit_layout, json.dumps(model_data)
    else:
        template = "bootstrap" if toggle else "darkly"
        model_data_loads = json.loads(model_data)
        model_results = model_data_loads['model_results']
        for key, value in model_results.items():
            if key == 'logit_y_pp' or key == 'linear_y_pp':
                model_results[key] = np.array(value)
            elif key == 'logit_cap_data' or key == 'linear_cap_data':
                model_results[key] = pd.DataFrame.from_dict(value)
            elif key == 'linear_idata' or key == 'logit_idata':
                model_results[key] = convert_from_serializable(value)
            elif key == 'logit_y' or key == 'linear_y':
                model_results[key] = pd.Series(value)
            elif key == 'logit_model' or key == 'linear_model':
                model_results[key] = convert_from_serializable(value)
        logit_df = pd.DataFrame.from_dict(model_data_loads['logit_df'])
        linear_df = pd.DataFrame.from_dict(model_data_loads['linear_df'])
        state = model_data_loads['state']
        test_year = model_data_loads['test_year']
        logit_idata = model_results['logit_idata']
        logit_cap_data = model_results['logit_cap_data']
        linear_idata = model_results['linear_idata']
        linear_cap_data = model_results['linear_cap_data']
        if template == "darkly":
            logit_calib_fig = plot_calibration(logit_idata, logit_cap_data, True)
            linear_accuracy_fig = plot_accuracy(linear_idata[0], linear_idata[1], linear_cap_data, True)
            logit_map_fig, logit_plot_fig, linear_map_fig, linear_plot_fig = plot_results(model_results, logit_df, linear_df, state, test_year, True)
        else:
            logit_calib_fig = plot_calibration(logit_idata, logit_cap_data, False)
            linear_accuracy_fig = plot_accuracy(linear_idata[0], linear_idata[1], linear_cap_data, False)
            logit_map_fig, logit_plot_fig, linear_map_fig, linear_plot_fig = plot_results(model_results, logit_df, linear_df, state, test_year, False)

        linear_layout, logit_layout = create_layout(logit_map_fig, logit_plot_fig, logit_calib_fig, linear_map_fig, linear_plot_fig, linear_accuracy_fig)

        return linear_layout, logit_layout, model_data

@dash_app.callback(
    Output('preds-model', 'options'),
    Input('state-dropdown', 'value'),
    prevent_initial_call=True
)
def update_preds(state):
    if state != 'PA':
        new_predictor_options = [{'label': 'Urbanization Index', 'value': 'urbanindex'},
                     {'label': 'Education Index', 'value': 'educationindex'},
                     {'label': 'Unemployment Rate', 'value': 'unemployment'},
                     {'label': 'Estimate of Median Household Income', 'value': 'hh_income'},
                     {'label': 'Per Capita Personal Income', 'value': 'personal_income'},
                     {'label': 'Temperature on Election Day', 'value': 'feelslike'},
                     {'label': 'Precipitation on Election Day', 'value': 'precip'},
                     {'label': 'Visibility on Election Day', 'value': 'visibility'},
                     {'label': 'Humidity on Election Day', 'value': 'humidity'}]
    else:
        new_predictor_options = predictor_options

    return new_predictor_options

if __name__ == '__main__':
    dash_app.run_server(debug=True)
