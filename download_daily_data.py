import boto3
import os
import gzip
import shutil
from botocore.config import Config
from botocore.exceptions import ClientError
from datetime import date, timedelta
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv(dotenv_path='polygon.env')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

if not AWS_ACCESS_KEY_ID or AWS_ACCESS_KEY_ID == "YOUR_POLYGON_ACCESS_KEY":
    print("ERROR: Please set your Polygon.io credentials in the polygon.env file.")
    exit()

# --- S3 Client Initialization ---
session = boto3.Session(
  aws_access_key_id=AWS_ACCESS_KEY_ID,
  aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)
s3 = session.client(
  's3',
  endpoint_url='https://files.polygon.io',
  config=Config(signature_version='s3v4'),
)

bucket_name = 'flatfiles'

# --- Date Range and Download Directory ---
start_date = date(2020, 8, 3)
end_date = date.today()
output_dir = 'daily-historical-DATA'

os.makedirs(output_dir, exist_ok=True)

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

# --- Main Download Loop ---
print(f"Starting download of daily aggregation data from {start_date} to {end_date}...")

for single_date in daterange(start_date, end_date):
    year = single_date.strftime("%Y")
    month = single_date.strftime("%m")
    date_str = single_date.strftime("%Y-%m-%d")

    object_key = f'us_stocks_sip/day_aggs_v1/{year}/{month}/{date_str}.csv.gz'
    
    local_gz_path = os.path.join(output_dir, f'{date_str}.csv.gz')
    local_csv_path = os.path.join(output_dir, f'{date_str}.csv')

    if os.path.exists(local_csv_path):
        print(f"File '{local_csv_path}' already exists. Skipping.")
        continue

    if not os.path.exists(local_gz_path):
        print(f"Downloading file '{object_key}'...")
        try:
            s3.download_file(bucket_name, object_key, local_gz_path)
            print(f"Successfully downloaded to '{local_gz_path}'")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"File '{object_key}' not found. Skipping.")
                continue
            else:
                print(f"An error occurred downloading '{object_key}': {e}")
                continue
    
    print(f"Decompressing '{local_gz_path}'...")
    try:
        with gzip.open(local_gz_path, 'rb') as f_in:
            with open(local_csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(local_gz_path)
        print(f"Successfully decompressed to '{local_csv_path}'.")
    except Exception as e:
        print(f"An error occurred during decompression: {e}")

print("\nDownload process finished.") 