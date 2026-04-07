from src.data_fetching.fetch_reports import FinancialReportsDownloader

if __name__ == "__main__":
    downloader = FinancialReportsDownloader("Classifier/config/config.yaml")
    downloader.run()
