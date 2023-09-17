import requests
import time

#insert the colab link to keep it alive 

while True:
    try:
        requests.get('https://colab.research.google.com/drive/1ZB0SWgUGEz2dRP7oj2G0HQdDOq3fRgPA?hl=it#scrollTo=VC3L3yWdnPKQ')
        print("Kept alive.")
    except:
        print("Failed to keep alive.")
    
    #one request every 10 minutes (600 sec)
    time.sleep(600)