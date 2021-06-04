# Target name
TARGET_NAME = 'tip'

# The features to be used for training.
# If FEATURE_NAMES is None, then all the available columns will be
# used as features, except for the target column.
FEATURE_NAMES = [
    'trip_miles',
    'trip_seconds',
    'fare',
    'trip_start_month',
    'trip_start_hour',
    'trip_start_day',
]

MODEL_FILE_NAME = 'model.joblib'
