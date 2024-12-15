import csv
import os

class CSVHandler:
    def __init__(self, output_file):
        self.output_file = output_file

    def write_header(self):
        """CSVヘッダーを書き込む"""
        with open(self.output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Timestamp", "PERCLOS", "State"])

    def append_row(self, timestamp, perclos_score, state):
        """CSVにデータ行を追記する"""
        with open(self.output_file, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([timestamp, perclos_score, state])

    @staticmethod
    def get_unique_filename(base_name):
        """ファイルが存在する場合、一意のファイル名を生成する"""
        if not os.path.exists(base_name):
            return base_name
        counter = 1
        while True:
            new_name = f"{os.path.splitext(base_name)[0]}({counter}){os.path.splitext(base_name)[1]}"
            if not os.path.exists(new_name):
                return new_name
            counter += 1

    @staticmethod
    def get_user_filename(default_name="perclos_output.csv"):
        """ユーザーからCSVファイル名を入力させる"""
        while True:
            base_filename = input(f"Enter the base name for the CSV file (e.g., '{default_name}', default: '{default_name}'): ").strip()
            if not base_filename:
                base_filename = default_name
            # 無効な文字が含まれていないかをチェック
            if any(char in base_filename for char in r"<>:\"/\\|?*"):
                print("Invalid characters in file name. Please try again.")
            else:
                return CSVHandler.get_unique_filename(base_filename)

