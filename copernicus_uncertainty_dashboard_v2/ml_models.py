import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

FEATURES = ['temperature_c', 'relative_humidity', 'wind_speed', 'pm25', 'pm10', 'no2', 'o3', 'so2', 'co', 'aod', 'dust_deposition']


def train_random_forest_regressor(df, target='health_burden_score'):
    X = df[FEATURES]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, preds),
        'feature_importance': dict(zip(FEATURES, model.feature_importances_))
    }
    return model, metrics


def train_interval_models(df, target='health_burden_score'):
    X = df[FEATURES]
    y = df[target]
    models = {}
    for alpha, name in [(0.05, 'q05'), (0.50, 'q50'), (0.95, 'q95')]:
        model = GradientBoostingRegressor(loss='quantile', alpha=alpha, n_estimators=220, max_depth=3, random_state=42)
        model.fit(X, y)
        models[name] = model
    pred = pd.DataFrame({'date': df['date']})
    pred['q05_pred'] = models['q05'].predict(X)
    pred['q50_pred'] = models['q50'].predict(X)
    pred['q95_pred'] = models['q95'].predict(X)
    return models, pred.groupby('date')[['q05_pred', 'q50_pred', 'q95_pred']].mean().reset_index()


def train_random_forest_classifier(df, target='risk_code'):
    X = df[FEATURES]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, classification_report(y_test, preds, output_dict=True)


def train_keras_regressor(df, target='health_burden_score'):
    X = df[FEATURES].values
    y = df[target].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, validation_split=0.2, epochs=60, batch_size=16,
              callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)], verbose=0)
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    return model, scaler, {'loss': loss, 'mae': mae}
