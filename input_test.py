from ExcelExport import Saver

while True :
    a, b, c = input().split()
    if a == '0' and b == '0' and c == '0' : # 프로그램 종료 조건
        break
    if a == '1' and b == '1' and c == '1' : # 입력받은 데이터 들을 엑셀 파일로 저장
        Saver.save_to_excel()
    else :
        Saver.add_data(img_name,latitude,longitude) # 입력 저장
    
