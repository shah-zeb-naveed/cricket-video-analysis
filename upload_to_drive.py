from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def upload_to_drive(file_path):
    # Authenticate and create the PyDrive client
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Opens a browser for authentication
    drive = GoogleDrive(gauth)

    # Create a file on Google Drive
    file = drive.CreateFile({'title': file_path.split('/')[-1]})  
    file.SetContentFile(file_path)  # Set the file path
    file.Upload()  # Upload the file

    print(f"Uploaded {file_path} to Google Drive with ID: {file['id']}")

# Replace with the path to your MP4 file
file_path = "single.mp4"
upload_to_drive(file_path)
