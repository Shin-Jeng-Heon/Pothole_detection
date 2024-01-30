import pandas as pd
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook
from openpyxl.styles import Alignment
import atexit
import os

class DataSaver:
    def __init__(self):
        self.data = []

    def add_data(self, value1:str, value2:float, value3:float): # 데이터 타입은 맞추지 않아도 됨 (모두 문자열로 저장해도 괜찮음)
        self.data.append([value1, value2, value3])

    def save_to_excel(self):
        folder_name = "Detection Result" # Excel이 저장될 파일명
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        filename = os.path.join(folder_name, "Detection_Result.xlsx")
        file_base = "Detection_Result"
        file_extension = ".xlsx"
        counter = 1

        while os.path.exists(filename):
            filename = os.path.join(folder_name, f"{file_base}_{counter}{file_extension}")
            counter += 1

        df = pd.DataFrame(self.data, columns=["Image Name", "Latitude", "Longitude"])

        wb = Workbook()
        ws = wb.active

        for r_idx, r in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
            ws.append(r)
            for cell in ws[r_idx]:
                cell.alignment = Alignment(horizontal='center', vertical='center')

        ws.column_dimensions['A'].width = 30
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 20

        wb.save(filename)

Saver = DataSaver()
# ExcelExport모듈을 불러온 프로그램이 종료된 경우에 Excel파일로 저장하고 싶으면 아래 주석 지우기
# atexit.register(Saver.save_to_excel)
