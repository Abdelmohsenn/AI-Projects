import json, os
import random
from systemPrompt import systemPrompt

ActionMap = {
    # Application management
    'open_app': "open -a ",
    'close_app': "pkill ",
    'relaunch_app': "killall ",
    'hide_app': "osascript -e 'tell application \"System Events\" to set visible of process ",

    # System control
    'shutdown': "osascript -e 'tell app \"System Events\" to shut down'",
    'restart': "osascript -e 'tell app \"System Events\" to restart'",
    'lock_screen': "/System/Library/CoreServices/Menu\\ Extras/User.menu/Contents/Resources/CGSession -suspend",
    'sleep': "pmset sleepnow",
    'logout': "osascript -e 'tell application \"System Events\" to log out'",
    'empty_trash': "osascript -e 'tell application \"Finder\" to empty the trash'",

    # Volume control
    'adjust_volume': "osascript -e 'set volume output volume ",
    'mute_volume': "osascript -e 'set volume output muted true'",
    'unmute_volume': "osascript -e 'set volume output muted false'",

    # Display
    'adjust_brightness': "brightness ",

    # Web & URLs
    'search_web': "open 'https://www.google.com/search?q=",
    'open_url': "open '",

    # Music control
    'play_song': "osascript -e 'tell application \"Music\" to play'",
    'pause_music': "osascript -e 'tell application \"Music\" to pause'",
    'next_track': "osascript -e 'tell application \"Music\" to next track'",
    'previous_track': "osascript -e 'tell application \"Music\" to previous track'"
}


targetMap = {
    # Browsers
    'Safari': "'Safari'", 'Chrome': "'Google Chrome'", 'Google Chrome': "'Google Chrome'", 'Edge': "'Microsoft Edge'", 'Microsoft Edge': "'Microsoft Edge'",

    # Code Editors
    'Visual Studio': "'Code'", 'VSCode': "'Code'", 'Code': "'Code'",

    # Office Apps
    'Microsoft Word': "'Microsoft Word'", 'Word': "'Microsoft Word'",
    'Microsoft Excel': "'Microsoft Excel'", 'Excel': "'Microsoft Excel'",
    'Microsoft PowerPoint': "'Microsoft PowerPoint'", 'PowerPoint': "'Microsoft PowerPoint'",

    # System Apps
    'Finder': "'Finder'", 'System Preferences': "'System Preferences'", 'Settings': "'System Preferences'", 'Mail': "'Mail'", 'FaceTime': "'FaceTime'",

    # Music / Photos
    'Music': "'Music'", 'iTunes': "'Music'", 'Photos': "'Photos'", 'Gallery': "'Photos'",

    # Messaging / Social
    'WhatsApp': "'WhatsApp'", 'Messages': "'Messages'", 'Messenger': "'Messages'",

    # Utilities
    'Reminders': "'Reminders'", 'Calendar': "'Calendar'", 'Weather': "'Weather'",

    # Optional common shortcuts / nicknames
    'Chrome Browser': "'Google Chrome'", 'MS Word': "'Microsoft Word'", 'MS Excel': "'Microsoft Excel'", 'MS PowerPoint': "'Microsoft PowerPoint'"
}

ValueRelatedList = ['adjust_brightness', 'unmute_volume','mute_volume', 'adjust_volume']
errorFlag = False
def execute(client):
    errorFlag = False
    audio = client.recordAudio()
    query = client.STT(audio)
    result = client.response(query, systemPrompt)
    print(result)
    res = json.loads(result)
    print(res['action'])

    if res['action'] not in ValueRelatedList:
        finalCommand = ActionMap[res['action']] + " " + (targetMap[res['target']] if res['target'] in targetMap else f"'{res['target']}'")
    else :
        finalCommand = ActionMap[res['action']] + " " + str(res['value'] + "'")


    print(f"Executing command: {finalCommand}")  # Debug print

    if 'clarify' in finalCommand:
        errorFlag = True
    else:
        print(res)
        os.system(finalCommand)

    return errorFlag