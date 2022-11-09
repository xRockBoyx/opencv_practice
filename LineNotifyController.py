import time
import requests
from requests               import Session, Request
from GoogleBucketController import GCStorage
from google.cloud           import storage


class LineNotifier:

    def __init__(self, notifyToken, privateApiKeyJsonFilePath):
        self.gcs          = GCStorage(storage.Client.from_service_account_json(privateApiKeyJsonFilePath))
        self.notifyToken  = notifyToken
        self.IMAGE_BASE_URL = "https://storage.googleapis.com/ai-training-notifier-bucket/"
        self.POST_URL     = "https://notify-api.line.me/api/notify"
        self.POST_HEADERS = { "Authorization": "Bearer " + notifyToken }

    def upload_to_google_bucket(self, bucketName, bucketFileName, filePath):
        bucket_gcs = self.gcs.get_bucket(bucketName)
        self.gcs.upload_file(bucket_gcs, 
                             bucketFileName, 
                             filePath)
        return

    def send_message(self, text):
        data = { 'message' : text }
        s = Session()
        req = Request('POST',
                      url = self.POST_URL,
                      headers = self.POST_HEADERS,
                      data = data)
        prepared = s.prepare_request(req)
        
        resp = s.send(  prepared )
        return
            

    def send_image(self, text, bucketFileName):

        payload = {
            'message' : text,
            'imageThumbnail' : str(self.IMAGE_BASE_URL + 'thumbnail.png'),
            'imageFullsize' : str(self.IMAGE_BASE_URL + bucketFileName)
        }

        requests.post(  self.POST_URL,
                        headers = self.POST_HEADERS,
                        data = payload)

        return

# LINE Notify 權杖
# token = '8sINtMZ1MjV2mOnnbIe0j6KTbiWtlfv6ilzgALwfUai'

# LineNotifyTool = LineNotifier(notifyToken = '8sINtMZ1MjV2mOnnbIe0j6KTbiWtlfv6ilzgALwfUai',
#                               privateApiKeyJsonFilePath = 'line-notifier-image-storage-65936edbb18a.json')

# # LineNotifyTool.send_message(text = 'test')
# LineNotifyTool.send_image(text = 'image',
#                           imageURL = "./test1.png")
# localtime = time.asctime( time.localtime(time.time()) )
# print (str(localtime))

# #將結果用Line Notify傳給我
