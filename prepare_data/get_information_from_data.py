from pandas_profiling import ProfileReport


def get_information_from_data(dataframe, save_report_path):
    print("SHAPE:\n", dataframe.shape)
    print("DESCRIBE:\n", dataframe.describe())
    print("INFO:\n", dataframe.info())

    profile = ProfileReport(dataframe, title="Pandas Profiling Report from Heart Failure")
    profile.to_file(f"{save_report_path}pp_report_heart_failure.html")

