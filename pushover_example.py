# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 21:47:43 2018

@author: Chingchongs
"""



import requests
import json



data=json.loads(open('keys.json').read())
data['priority'] = 2
data['message'] = 'Object Detected!'
data['retry'] = 30 
data['expire'] = 300

r = requests.post("https://api.pushover.net/1/messages.json", data = data,
files = {
  "attachment": ("test.jpg", open("test.jpg", "rb"), "image/jpeg")
})
print(r.text)

