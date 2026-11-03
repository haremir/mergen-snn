try:
    from google.colab import drive
    print("google.colab imported")
    # check if mount without force_remount works or asks for code
    # drive.mount('/content/drive')
except Exception as e:
    print("Failed:", e)
