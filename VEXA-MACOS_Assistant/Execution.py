import json, os
import random
from systemPrompt import systemPrompt

seconds = random.choice([3, 4, 5])  # Default recording time in seconds
errorFlag = False
def execute(client):
    audio = client.recordAudio(seconds=seconds)
    query = client.STT(audio)
    result = client.response(query, systemPrompt)
    print(result)
    res = json.loads(result)
    print(res['action'])

    finalCommand = ''
    try:
        match res['action']:
            # APP CONTROLS
            case 'open_app':
                finalCommand = "open -a "
            case 'close_app':
                finalCommand = "pkill "
            case 'relaunch_app':
                finalCommand = "killall "
            case 'hide_app':
                finalCommand = "osascript -e 'tell application \"System Events\" to set visible of process "
            # SYSTEM STUFF
            case 'shutdown':
                finalCommand = "osascript -e 'tell app \"System Events\" to shut down'"
            case 'restart':
                finalCommand = "osascript -e 'tell app \"System Events\" to restart'"
            case 'lock_screen':
                finalCommand = "/System/Library/CoreServices/Menu\\ Extras/User.menu/Contents/Resources/CGSession -suspend"
            case 'sleep':
                finalCommand = "pmset sleepnow"
            case 'logout':
                finalCommand = "osascript -e 'tell application \"System Events\" to log out'"
            case 'empty_trash':
                finalCommand = "osascript -e 'tell application \"Finder\" to empty the trash'"
            # VOLUME/BRIGHTNESS
            case 'adjust_volume':
                value = res.get('value', 50)
                if value == 'max':
                    value = '100'
                elif value == 'min':
                    value = '0'
                finalCommand = f"osascript -e 'set volume output volume {value}'"
            case 'mute_volume':
                finalCommand = "osascript -e 'set volume output muted true'"
            case 'unmute_volume':
                finalCommand = "osascript -e 'set volume output muted false'"
            case 'adjust_brightness':
                value = res.get('value', 0.5)
                if value == 'max' or value == '100%':
                    value = '1.0'
                elif value == 'min' or value == '0%':
                    value = '0.0'
                else:
                    value = str(value)
                finalCommand = f"brightness {value}"  # requires https://github.com/nriley/brightness
            # WEB/SEARCH
            case 'search_web':
                query = res['target'].replace(' ', '+')
                finalCommand = f"open 'https://www.google.com/search?q={query}'"
            case 'open_url':
                url = res['target']
                finalCommand = f"open '{url}'"
            # MUSIC/MEDIA
            case 'play_song':
                finalCommand = f'''osascript -e 'tell application "Music" to play' '''
            case 'pause_music':
                finalCommand = '''osascript -e 'tell application "Music" to pause' '''
            case 'next_track':
                finalCommand = '''osascript -e 'tell application "Music" to next track' '''
            case 'previous_track':
                finalCommand = '''osascript -e 'tell application "Music" to previous track' '''
            # REMINDERS/CALENDAR/ALARMS
            case 'set_alarm':
                finalCommand = f"open -a 'Clock'"
            case 'add_reminder':
                reminder = res['target']
                finalCommand = f'''osascript -e 'tell application "Reminders" to make new reminder with properties {{name: "{reminder}"}}' '''
            case 'add_calendar_event':
                event = res['target']
                finalCommand = f'''osascript -e 'display dialog "Calendar event added: {event}"' '''
            # WEATHER & SYSTEM INFO
            case 'get_weather':
                location = res['target'].replace(' ', '+')
                finalCommand = f"open 'https://www.google.com/search?q=weather+{location}'"
            case 'get_system_info':
                finalCommand = "system_profiler SPHardwareDataType"
            case 'check_disk_space':
                finalCommand = "df -h"
            # FILES & FOLDERS
            case 'open_folder':
                folder = res['target']
                finalCommand = f"open '{folder}'"
            case 'list_files':
                folder = res['target']
                finalCommand = f"ls -l '{folder}'"
            case 'delete_file':
                file = res['target']
                finalCommand = f"rm -i '{file}'"
            # FUN/UTILITY
            case 'say':
                finalCommand = "say "
            case _:
                finalCommand = res['action'] 
    except BaseException as e:
        print(f"echo 'Error Occurred: Invalid action {res['action']}'")
        errorFlag = True

    try:
        match res['target']:

            case "Safari":
                finalCommand += "'Safari'"
            case "Edge" | "Microsoft Edge":
                finalCommand += "'Microsoft Edge'"
            case "Google Chrome"| "Chrome":
                finalCommand += "'Google Chrome'"
            case "System Preferences" | "System Settings" | "Settings":
                finalCommand += "'System Preferences'"
            case "Activity Monitor":
                finalCommand += "'Activity Monitor'"
            case "Terminal":
                finalCommand += "'Terminal'"
            case "Calculator":
                finalCommand += "'Calculator'"
            case "Contacts":
                finalCommand += "'Contacts'"
            case "Reminders":
                finalCommand += "'Reminders'"
            case "Notes":
                finalCommand += "'Notes'"
            case "FaceTime":
                finalCommand += "'FaceTime'"
            case "Music" | "Apple Music"| "iTunes":
                finalCommand += "'Music'"
            case "Calendar":
                finalCommand += "'Calendar'"
            case "Photos":
                finalCommand += "'Photos'"
            case "Mail":
                finalCommand += "'Mail'"
            case "Messages":
                finalCommand += "'Messages'"
            case "WhatsApp":
                finalCommand += "'WhatsApp'"
            case "FaceTime":
                finalCommand += "'FaceTime'"
            case "Finder":
                finalCommand += "'Finder'"
            # --- URLs/Queries ---
            case "OpenAI":
                finalCommand += "'OpenAI'"
            case "news":
                finalCommand += "'news'"
            case "weather today":
                finalCommand += "'weather today'"
            case "github" | "GitHub":
                finalCommand += "'GitHub'"
            # --- System/File utility ---
            case "disk space":
                finalCommand += "'disk space'"
            case "system info":
                finalCommand += "'system info'"
    except BaseException as e:
        print(f"echo 'Error Occurred: Invalid action {res['action']}'")
        errorFlag = True

    if 'clarify' in finalCommand:
        errorFlag = True
    else:
        print(res)
        print(f"Executing command: {finalCommand}")  # Debug print
        os.system(finalCommand)
        
    return errorFlag