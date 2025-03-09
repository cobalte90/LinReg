def one_hot(X):
    season_col = X['season']
    season_dict = {
        'season_1' : [1 if x == 1.0 else 0 for x in season_col],
        'season_2' : [1 if x == 2.0 else 0 for x in season_col],
        'season_3' : [1 if x == 3.0 else 0 for x in season_col],
        'season_4' : [1 if x == 4.0 else 0 for x in season_col]
    }
    X = X.drop('season', axis=1)
    for col in season_dict:
        X[col] = season_dict[col]

    yr_col = X['yr']
    yr_dict = {
        'yr_1' : [1 if x == 1 else 0 for x in season_col],
        'yr_0' : [1 if x == 0 else 0 for x in season_col]
    }
    X = X.drop('yr', axis=1)
    for col in yr_dict:
        X[col] = yr_dict[col]

    mnth_col = X['mnth']
    mnth_dict = {
        'january' : [1 if x == 1 else 0 for x in mnth_col],
        'february' : [1 if x == 2 else 0 for x in mnth_col],
        'march' : [1 if x == 3 else 0 for x in mnth_col],
        'april' : [1 if x == 4 else 0 for x in mnth_col],
        'may' : [1 if x == 5 else 0 for x in mnth_col],
        'june' : [1 if x == 6 else 0 for x in mnth_col],
        'july' : [1 if x == 7 else 0 for x in mnth_col],
        'august' : [1 if x == 8 else 0 for x in mnth_col],
        'september' : [1 if x == 9 else 0 for x in mnth_col],
        'october' : [1 if x == 10 else 0 for x in mnth_col],
        'november' : [1 if x == 11 else 0 for x in mnth_col],
        'december' : [1 if x == 12 else 0 for x in mnth_col]
    }
    X = X.drop('mnth', axis=1)
    for col in mnth_dict:
        X[col] = mnth_dict[col]

    holiday_col = X['holiday']
    holiday_dict = {
        'holiday_yes' : [1 if x == 1 else 0 for x in holiday_col],
        'holiday_no' : [1 if x == 0 else 0 for x in holiday_col]
    }
    X = X.drop('holiday', axis=1)
    for col in holiday_dict:
        X[col] = holiday_dict[col]

    weekday_col = X['weekday']
    weekday_dict = {
        'monday' : [1 if x == 0 else 0 for x in weekday_col],
        'tuesday' : [1 if x == 1 else 0 for x in weekday_col],
        'wednesday' : [1 if x == 2 else 0 for x in weekday_col],
        'thursday' : [1 if x == 3 else 0 for x in weekday_col],
        'friday' : [1 if x == 4 else 0 for x in weekday_col],
        'saturday' : [1 if x == 5 else 0 for x in weekday_col],
        'sunday' : [1 if x == 6 else 0 for x in weekday_col]
    }
    X = X.drop('weekday', axis=1)
    for col in weekday_dict:
        X[col] = weekday_dict[col]

    workingday_col = X['workingday']
    workingday_dict = {
        'workingday_yes' : [1 if x == 1 else 0 for x in workingday_col],
        'workingday_no' : [1 if x == 0 else 0 for x in workingday_col]
    }
    X = X.drop('workingday', axis=1)
    for col in workingday_dict:
        X[col] = workingday_dict[col]
    return X