from pandas_profiling import ProfileReport

class DataProfiler:
    """
    Class to profile data and save report to file.
    """

    def __init__(self, data=None, report_title=None, out_path=None):
        assert data is not None, "Must provide data to DataProfiler class"
        assert report_title is not None, "Must provide report title to DataProfiler class"
        self.data = data
        self.report_title = report_title
        self.out_path = out_path

    def profile(self):
        """
        Profile the data and save a report in html
        """
        self.profile = ProfileReport(self.data, title=self.report_title, explorative=True)
        self.profile.to_file(f"{self.out_path}/{self.report_title}.html")
