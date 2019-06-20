from Training import Training


def trainingChatBot():

    # Laptop Training
    print('START LAPTOP TRAINING')
    laptopInputFileName = "laptop-raw-data.json"
    laptopOutputFileName = "laptop-training-data"
    laptopTFlearnLog = "tflearn_laptop_logs"
    laptopTFlearnModel = "laptop-model.tflearn"
    laptopTraining = Training(laptopInputFileName,laptopOutputFileName, laptopTFlearnLog, laptopTFlearnModel)
    laptopTraining.execute()
    print('END LAPTOP TRAINING')

    # Tablet Training
    print('START TABLET TRAINING')
    tabletInputFileName = "tablet-raw-data.json"
    tabletOutputFileName = "tablet-training-data"
    tabletTFlearnLog = "tflearn_tablet_logs"
    tabletTFlearnModel = "tablet-model.tflearn"
    tabletTraining = Training(tabletInputFileName,tabletOutputFileName, tabletTFlearnLog, tabletTFlearnModel)
    tabletTraining.execute()
    print('END TABLET TRAINING')

    # Mobile Training
    print('START MOBILE TRAINING')
    mobileInputFileName = "mobile-raw-data.json"
    mobileOutputFileName = "mobile-training-data"
    mobileTFlearnLog = "tflearn_mobile_logs"
    mobileTFlearnModel = "mobile-model.tflearn"
    mobileTraining = Training(mobileInputFileName,mobileOutputFileName, mobileTFlearnLog, mobileTFlearnModel)
    mobileTraining.execute()
    print('END MOBILE TRAINING')