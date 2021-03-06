from h2o_wave import main, app, Q, ui
import pandas as pd
import numpy as np

def load_data():
    data = pd.read_csv('train.csv')
    return data

def test_data():
    data = pd.read_csv('test.csv')
    return data
data = load_data()
test_data = test_data()
# df_point_sized =  df.loc[:200,['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']]
df_point =  data.loc[:200,:]


def preprocessing(data):
    data = data.drop(['ID'],axis=1)
    data = pd.get_dummies(data, columns=['Accomodation_Type','Reco_Insurance_Type','Is_Spouse'],drop_first=True) 
    data['City_Code'] = data['City_Code'].replace(to_replace='C',value='',regex=True)
    data['City_Code'] = data['City_Code'].astype(np.int64)
    
    data['Holding_Policy_Duration'] = data['Holding_Policy_Duration'].replace(to_replace='14+',value=15,regex=True)
    data['Holding_Policy_Duration'] = pd.to_numeric(data['Holding_Policy_Duration'])
    data['Holding_Policy_Duration'] = data['Holding_Policy_Duration'].fillna(data['Holding_Policy_Duration'].median())
    
    data['Health Indicator'] = data['Health Indicator'].replace(to_replace='X',value='',regex=True)
    data['Health Indicator'] = data['Health Indicator'].astype(np.float64)
    
    data  = data.fillna(data.mean())
    data['Health Indicator']  = data['Health Indicator'].fillna(data['Health Indicator'].mean())
    data = data.drop(['Upper_Age'],axis=1)
    return data

train = preprocessing(data)
test = preprocessing(test_data)
def rf_training(train,test):
    y = train.iloc[:,10].values
    train = train.drop(['Response'],axis=1)
    train = train.iloc[:,:].values
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(train, y)
    y_pred = clf.predict(test)
    y_pred = pd.DataFrame(y_pred,columns=['RF Predictions'])
    return y_pred

def dt_training(train,test):
    y = train.iloc[:,10].values
    train = train.drop(['Response'],axis=1)
    train = train.iloc[:,:].values
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state = 0)
    dt.fit(train, y)  
    y_pred = dt.predict(test)
    y_pred = pd.DataFrame(y_pred,columns=['DT Predictions'])
    return y_pred


def make_markdown_row(values):
    return f"| {' | '.join([str(x) for x in values])} |"


def make_markdown_table(fields, rows):
    return '\n'.join([
        make_markdown_row(fields),
        make_markdown_row('---' * len(fields)),
        '\n'.join([make_markdown_row(row) for row in rows]),
    ])


@app('/demo')
async def serve(q: Q):
    hash = q.args['#']
    if hash == 'rf':
        q.page['random'] = ui.form_card(box='1 12 5 2', items=[ui.progress('Running...')])
        values = await q.run(rf_training,train,test)
        message = 'RF Model trained successfully'
        q.page['random'] = ui.form_card(box='1 12 5 5', items=[
            ui.message_bar('info', message),
            ui.text(make_markdown_table(fields=values.columns.tolist(),rows=values.values.tolist()))
            ])
    await q.page.save()

    if hash == 'dt':
        q.page['decision'] = ui.form_card(box='6 12 5 5', items=[ui.progress('Running...')])
        value = await q.run(dt_training,train,test)
        message = 'DT Model trained successfully'
        q.page['decision'] = ui.form_card(box='6 12 5 5', items=[
            ui.message_bar('info', message),
            ui.text(make_markdown_table(fields=value.columns.tolist(),rows=value.values.tolist()))
            ])
    await q.page.save()   

    
    if hash:
        if hash == 'show_data':
            q.page['show_data'].items=[
                ui.text(make_markdown_table(fields=df_point.columns.tolist(),rows=df_point.values.tolist()))
            ]
        elif hash == 'preprocess':
            q.page['preprocess'].items=[
                 ui.text(make_markdown_table(fields=train.columns.tolist(),rows=train.values.tolist()))
            ]
             
        elif hash == 'train':
            q.page['train'] = ui.form_card(
            box='4 10 4 2',
            items=[
                ui.text('Click on the button below to train your model'),
                ui.button(name='#rf', label='Random Forest', primary=True),
                ui.button(name='#dt', label='Decision Trees', primary=True),
            ])
         
    else:
        q.page['nav'] = ui.tab_card(
            box='1 1 7 1',
            items=[
                ui.tab(name='#show_data', label='Show Data'),
                ui.tab(name='#preprocess', label='Preprocess'),
                ui.tab(name='#train', label='Train & Predict'),
                # ui.tab(name='#predict', label='Predict'),
            ],
        )  
        q.page['show_data'] = ui.form_card(
        box='1 2 9 4',
        items=[
            ui.text('Display your data here !')
            ])    

        q.page['preprocess'] = ui.form_card(
        box='1 6 9 4',
        items=[
            ui.text('Preprocessing of data')
        ]
            
        )
         
        q.page['btn'] = ui.form_card(
            box='8 1 2 1',
            items=[
            ui.link(label='GitHub', path ='https://github.com/aniketwattamwar/',target=''),
             
            ]) 

    await q.page.save()

 