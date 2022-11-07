from linebot        import LineBotApi
from linebot.models import TextSendMessage

line_bot_api = LineBotApi('aLqLtKUxvc+r6k/Tyh1wlCIfWNknlHx9V+apG+ubM9Acf1JjTqJPL6dYUj9GOW5mO8fD+LmQcovyz+yPvYQ/Kz86vVGWfK9bS4AzntufCNJfuIXZDmQ4PE4IziLAsGW/JF1zBh8Bd10hxizTJQuJvQdB04t89/1O/w1cDnyilFU=')

line_bot_api.push_message('KA9JofpzhxUGVb8zGZ26Sq12aH64CC', TextSendMessage(text='Hello World!!!'))
