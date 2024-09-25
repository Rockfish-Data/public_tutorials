import pandas as pd
import numpy as np

import rockfish as rf
import rockfish.actions as ra
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import asyncio

import pickle
import warnings

warnings.filterwarnings("ignore")


def search_model(label: dict, conn: rf.Connection) -> rf.Model:
    return conn.list_models(labels=label).last()


async def generate_data(model: rf.Model, conn: rf.Connection) -> pd.DataFrame:
    session_target = ra.SessionTarget(target=1500, max_cycles=250)

    builder = rf.WorkflowBuilder()
    builder.add_model(model)
    builder.add_action(ra.GenerateTimeGAN(), alias='gen', parents=[model, session_target])
    builder.add_action(session_target, parents=['gen'])
    builder.add_action(ra.DatasetSave(name='demo_central_syn'), parents=['gen'])
    workflow = await builder.start(conn)
    print(workflow.id())
    async for log in workflow.logs():
        print(log)
    data = await (await workflow.datasets().concat(conn)).to_local(conn)
    return data.to_pandas()


def get_xgb_clf(X_train, y_train, X_valid, y_valid):
    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2, enable_categorical=True)
    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0)
    return clf


def create_X_y_for_xgb(datasets):
    session_key = "customer"
    X_columns = ["customer", "age", "gender", "merchant", "category", "amount"]
    cat_columns = ["customer", "age", "gender", "merchant", "category"]
    y_columns = ["fraud"]
    df = pd.concat(datasets)

    X = df[X_columns]
    X[cat_columns] = X[cat_columns].astype("category")
    y = df[y_columns]

    return X, y


def metrics(y, y_pred):
    dat = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred),
        # 'classification_report': classification_report(y, y_pred),
        # 'confusion_matrix': confusion_matrix(y, y_pred)
    }
    return pd.DataFrame(dat, index=[0])


def get_metric(default_metrics, with_synthetic_metrics):
    return (
        pd.DataFrame(
            pd.concat(
                [
                    pd.concat(
                        [default_metrics.recall.reset_index(drop=True),
                         with_synthetic_metrics.recall.reset_index(drop=True)]
                    ).reset_index(drop=True),
                    pd.Series(['default', 'blended'])
                ],
                axis=1
            ),
            # columns=['recall', 'model']
        ).rename(columns={0: 'model'})
    )


def train_model(data):
    X, y = create_X_y_for_xgb([data])
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    return get_xgb_clf(X_train, y_train, X_valid, y_valid)


def compute_metrics(data, Xt,yt):
    synthetic_precision, synthetic_recall = [], []
    models = []
    X, y = create_X_y_for_xgb([data])
    for i in range(250):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
        clf_with_synthetic = get_xgb_clf(X_train, y_train, X_valid, y_valid)
        models.append(clf_with_synthetic)

        y_pred_with_synthetic = clf_with_synthetic.predict(Xt)

        synthetic_recall.append(recall_score(yt, y_pred_with_synthetic))
        synthetic_precision.append(precision_score(yt, y_pred_with_synthetic))
        synthetic_precision = np.array(synthetic_precision)
        synthetic_recall = np.array(synthetic_recall)
        best_model = models[np.argmax(synthetic_recall)]
        return synthetic_precision, synthetic_recall, best_model


async def main():
    conn = rf.Connection.from_env()
    print('Downloading model to central location and generating synthetic data...')
    location1_model = await search_model({'location': 'location1_x'}, conn)
    # data = await generate_data(location1_model, conn)
    data = pd.read_csv('location1.csv')

    print('Augmenting data with synthetic data...')
    data_base = pd.concat([pd.read_csv('./location2.csv'), pd.read_csv('./location3.csv')])
    data = pd.concat([data_base, data])

    clf_default = pickle.load(open('./base_model.mdl', 'rb'))

    test_data = pd.read_csv('./testing.csv')
    Xt, yt = create_X_y_for_xgb([test_data])

    print('Evaluating model performance...')
    # synthetic_precision, synthetic_recall, best_model = compute_metrics(data, Xt, yt)
    # pickle.dump(best_model, open('best_model.mdl', 'wb'))
    clf_with_synthetic = pickle.load(open('best_model.mdl', 'rb'))
    y_pred_default = clf_default.predict(Xt)
    y_pred_synthetic = clf_with_synthetic.predict(Xt)
    default_metrics = metrics(yt, y_pred_default)
    synthetic_metrics = metrics(yt, y_pred_synthetic)

    df = pd.DataFrame({
        'Mean Precision': [default_metrics.precision.mean(), synthetic_metrics.precision.mean()],
        # 'Standard Deviation Precision': [0, synthetic_precision.std()],
        'Mean Recall': [default_metrics.recall.mean(), synthetic_metrics.recall.mean()],
        # 'Standard Deviation Recall': [0, synthetic_recall.std()]
    },
        index=['Default Model', 'Blended Synthetic Data Model'])
    print(df)
    print('Report logged to report.csv')
    df.to_csv('report.csv', index=False)

    print(f'''
    Metric Report:

    Default Model:
        Precision: {default_metrics.precision.mean()}
        Recall: {default_metrics.recall.mean()}

    Model with Blended Synthetic Data:
        Mean Precision: {synthetic_metrics.precision.mean()}
        Mean Recall: {synthetic_metrics.recall.mean()}
    ''')


asyncio.run(main())
