import boto3
import os
import gzip
import shutil
from botocore.config import Config
from botocore.exceptions import ClientError
from datetime import date, timedelta
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from polygon.env file
load_dotenv(dotenv_path='polygon.env')

# It is highly recommended to use environment variables or a shared credentials file.
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Check if credentials are set
if not AWS_ACCESS_KEY_ID or AWS_ACCESS_KEY_ID == "YOUR_POLYGON_ACCESS_KEY":
    print("ERROR: Please set your Polygon.io credentials in the polygon.env file.")
    exit()


# --- S3 Client Initialization ---
# Initialize a session using your credentials
session = boto3.Session(
  aws_access_key_id=AWS_ACCESS_KEY_ID,
  aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Create a client with your session and specify the endpoint
s3 = session.client(
  's3',
  endpoint_url='https://files.polygon.io',
  config=Config(signature_version='s3v4'),
)

# Specify the bucket name
bucket_name = 'flatfiles'

# --- Date Range and Download Directory ---
start_date = date(2020, 8, 3)
end_date = date.today()
output_dir = 'historical-DATA'

# Create download directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# --- Helper function for date iteration ---
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

# --- Main Download Loop ---
print(f"Starting download of minute aggregation data from {start_date} to {end_date}...")

for single_date in daterange(start_date, end_date):
    year = single_date.strftime("%Y")
    month = single_date.strftime("%m")
    date_str = single_date.strftime("%Y-%m-%d")

    # Specify the S3 object key name
    # Format: us_stocks_sip/minute_aggs_v1/{year}/{month}/{YYYY-MM-DD}.csv.gz
    object_key = f'us_stocks_sip/minute_aggs_v1/{year}/{month}/{date_str}.csv.gz'

    # Specify the local file name and path to save the downloaded file
    local_gz_path = os.path.join(output_dir, f'{date_str}.csv.gz')
    local_csv_path = os.path.join(output_dir, f'{date_str}.csv')

    # Check if the decompressed file already exists to avoid re-doing work
    if os.path.exists(local_csv_path):
        print(f"File '{local_csv_path}' already exists. Skipping.")
        continue

    # If the gzipped file doesn't exist locally, download it
    if not os.path.exists(local_gz_path):
        print(f"Downloading file '{object_key}' from bucket '{bucket_name}'...")
        try:
            # Download the file
            s3.download_file(bucket_name, object_key, local_gz_path)
            print(f"Successfully downloaded to '{local_gz_path}'")
        except ClientError as e:
            # It's common for data not to exist for weekends or holidays
            if e.response['Error']['Code'] == '404':
                print(f"File '{object_key}' not found on the server. Skipping (likely a weekend or holiday).")
                continue
            else:
                # Handle other potential S3 errors
                print(f"An error occurred while downloading '{object_key}': {e}")
                continue
    
    # At this point, the .gz file exists, so we can decompress it.
    print(f"Decompressing '{local_gz_path}'...")
    try:
        with gzip.open(local_gz_path, 'rb') as f_in:
            with open(local_csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove the original .gz file to save space
        os.remove(local_gz_path)
        print(f"Successfully decompressed to '{local_csv_path}' and removed original.")
    except Exception as e:
        print(f"An error occurred during decompression of '{local_gz_path}': {e}")

print("\nDownload process finished.") 