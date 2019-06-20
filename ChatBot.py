from Predict import Predict
from termcolor import colored


def runChatBot():
    # Laptop ChatBot
    laptopTrainingDataFileName = "laptop-training-data"
    laptopRawDataFileName = "laptop-raw-data.json"
    laptopTFlearnLog = "tflearn_laptop_logs"
    laptopTFlearnModel = "laptop-model.tflearn"
    laptopPredict = Predict(laptopTrainingDataFileName, laptopRawDataFileName, laptopTFlearnLog, laptopTFlearnModel)

    # Tablet ChatBot
    tabletTrainingDataFileName = "tablet-training-data"
    tabletRawDataFileName = "tablet-raw-data.json"
    tabletTFlearnLog = "tflearn_tablet_logs"
    tabletTFlearnModel = "tablet-model.tflearn"
    tabletPredict = Predict(tabletTrainingDataFileName, tabletRawDataFileName, tabletTFlearnLog, tabletTFlearnModel)

    # Mobile ChatBot
    mobileTrainingDataFileName = "mobile-training-data"
    mobileRawDataFileName = "mobile-raw-data.json"
    mobileTFlearnLog = "tflearn_mobile_logs"
    mobileTFlearnModel = "mobile-model.tflearn"
    mobilePredict = Predict(mobileTrainingDataFileName, mobileRawDataFileName, mobileTFlearnLog, mobileTFlearnModel)

    while True:
        predict = None
        second_chat = None

        first_chat = "Xin kính chào quý khách!\nQuý khách vui lòng chọn lựa các mục sau:"
        first_chat += "\n1. Chọn số 1 nếu quý khách mua LAPTOP (Máy tính xách tay)"
        first_chat += "\n2. Chọn số 2 nếu quý khách mua TABLET (Máy Tính Bảng, Ipad)"
        first_chat += "\n3. Chọn số 3 nếu quý khách mua Mobile (Điện Thoại Di Động)"
        first_chat += "\n4. Chọn số 4 nếu quý khách cần hỗ trợ khác"
        first_chat = colored(first_chat, 'blue')
        print(first_chat)
        choose = input('Chọn mục cần hỗ trợ: ')

        if choose is "1":
            predict = laptopPredict

        elif choose is "2":
            predict = tabletPredict

        elif choose is "3":
            predict = mobilePredict

        else:
            print('Nhân Viên Bán Hàng: ', colored('Chức năng này hiện tại vẫn chưa hoàn thiện\n', 'blue'))
            continue


        second_chat = colored(predict.response('Lời Chào Từ Khách Hàng'), 'blue')
        print('Nhân Viên Bán Hàng: ', second_chat)
        while True:
            inp = input('Bạn: ')
            response = predict.response(inp)
            if response is None:
                response = "Tôi không hiểu lắm, quý khách vui lòng nêu rõ yêu cầu hơn nữa?"
            print('Nhân Viên Bán Hàng: ', colored(response, 'blue'), '\n')