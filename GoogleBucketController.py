from google.cloud import storage

class GCStorage:
    def __init__(self, storageClient):
        self.client = storageClient

    def create_bucket(self, bucket_name, storage_class, bucket_location='US'):
        bucket = self.client.bucket(bucket_name)
        bucket.storage_class = storage_class
        return self.client.create_bucket(bucket, bucket_location)        

    def get_bucket(self, bucket_name):
        return self.client.get_bucket(bucket_name)

    def list_buckets(self):
        buckets = self.client.list_buckets()
        return [bucket.name for bucket in buckets]

    def upload_file(self, bucket, blob_destination, file_path):
        file_type = file_path.split('.')[-1]
        if file_type == 'png':
            content_type = 'image/png'
        elif file_type == 'jpg':
            content_type = 'image/jpeg'

        blob = bucket.blob(blob_destination)
        blob.upload_from_filename(file_path, content_type=content_type)
        return blob

    def list_blobs(self, bucket_name):
        return self.client.list_blobs(bucket_name)

# # Step 1. prepare the variables
# working_dir = pathlib.Path.cwd()
# files_folder = working_dir.joinpath('My Files')
# downloads_folder = working_dir.joinpath('Downloaded')
# bucket_name = 'gcs_api_demo'

# # Construct GCStorage instance


# # Connect to gcp_api_demo Cloud Storage bucket
# if not bucket_name in gcs.list_buckets():
#     bucket_gcs = gcs.create_bucket('gcs_api_demo', STORAGE_CLASSES[0])
# else:
#     bucket_gcs = gcs.get_bucket(bucket_name)

# # Step 4. Upload Files
# for file_path in files_folder.glob('*.*'):
#     # use file name without the extension
#     gcs.upload_file(bucket_gcs, 'without extension/' + file_path.stem, str(file_path))

#     # use full file name

# # Step 5. Download & Delete Files
# for blob in notifierBucketBlobs:
#     path_download = downloads_folder.joinpath(blob.name)
#     if not path_download.parent.exists():
#         path_download.parent.mkdir(parents=True)
#     blob.download_to_filename(str(path_download))
#     blob.delete()
# https://storage.googleapis.com/ai-training-notifier-bucket/test.png