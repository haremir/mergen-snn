from google.colab import drive
import sys

print("Attempting drive.mount...")
try:
    drive.mount('/content/drive', force_remount=False)
    print("Drive mounted successfully!")
except Exception as e:
    print("Drive mount exception:", e)
