from shutil import copy2
import csv, shutil

numberOfDatasets = {'dog_bark' : 0,
                    'children_playing' : 0,
                    'car_horn' : 0,
                    'air_conditioner' : 0,
                    'engine_idling' : 0,
                    'siren':0,
                    'jackhammer':0,
                    'drilling':0,
                    'street_music' : 0,
                    'gun_shot' : 0
                    }
with open('UrbanSound8K.csv', newline = '') as csvfile:
    allAudioFiles = csv.DictReader(csvfile)
    for audio in allAudioFiles:
        if audio['class'] == 'dog_bark':
            numberOfDatasets['dog_bark'] += 1
        elif audio['class'] == 'children_playing':
            numberOfDatasets['children_playing'] += 1
        elif audio['class'] == 'car_horn':
            numberOfDatasets['car_horn'] += 1
        elif audio['class'] == 'air_conditioner':
            numberOfDatasets['air_conditioner'] += 1
        elif audio['class'] == 'engine_idling':
            numberOfDatasets['engine_idling'] += 1
        elif audio['class'] == 'siren':
            numberOfDatasets['siren'] += 1
        elif audio['class'] == 'jackhammer':
            numberOfDatasets['jackhammer'] += 1
        elif audio['class'] == 'drilling':
            numberOfDatasets['drilling'] += 1
        elif audio['class'] == 'street_music':
            numberOfDatasets['street_music'] += 1
        elif audio['class'] == 'gun_shot':
            numberOfDatasets['gun_shot'] += 1
print(numberOfDatasets.keys)
print(numberOfDatasets)

amount = {  'dog_bark' : 0,
            'children_playing' : 0,
            'car_horn' : 0,
            'air_conditioner' : 0,
            'engine_idling' : 0,
            'siren':0,
            'jackhammer':0,
            'drilling':0,
            'street_music' : 0,
            'gun_shot' : 0
        }

with open('UrbanSound8K.csv', newline = '') as csvfile:
    allAudioFiles = csv.DictReader(csvfile)
    for audio in allAudioFiles:
        folder          = audio['fold']
        sliceFileName   = audio['slice_file_name'][:-4] + '.png'
        testTrain       = ''
        imageClass      = audio['class']
        
        amount[imageClass] += 1
        destFolder = 'all_set'
        # if amount[imageClass] < numberOfDatasets[imageClass] * 0.8 :
        #         testTrain = 'training_set'
        # else :
        #         testTrain = 'test_set'
        src_path  = "Images_Backups/{fold}/{slice_file_name}".format( fold = folder,
                                                              slice_file_name = sliceFileName)
        if not os.path.exists("Images/{mode}/{image_class}".format(     mode = destFolder,
                                                                        image_class = imageClass)):
                os.makedirs("Images/{mode}/{image_class}".format(     mode = destFolder,
                                                                        image_class = imageClass))
        dest_path = "Images/{mode}/{image_class}/{slice_file_name}".format( mode = destFolder,
                                                                            image_class = imageClass,
                                                                            slice_file_name = sliceFileName)
        # print(src_path)
        # print(dest_path)

        shutil.move(src = src_path,
                    dst = dest_path,
                    copy_function = copy2)