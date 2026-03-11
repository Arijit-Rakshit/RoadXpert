import os


# Change directory to platform-tools
platform_tools_directory = r"ADB_CALL\platform-tools"
os.chdir(platform_tools_directory)

number = "8972276411"

os.system(f"adb.exe shell am start -a android.intent.action.CALL -d tel:+91{number}")



#     •    shell: This command is used to open a remote shell on the Android device. It allows you to execute commands on the device as if you were directly inputting them on the device itself.

#     •    am start: am stands for Activity Manager, and start is the command to start an activity. In this case, the activity is initiating a call.

#     •    -a <ACTION>: This flag is used to specify the intent action, which is a string that identifies the type of operation to be performed. In the case of android.intent.action.CALL, it indicates that the action is to initiate a call.

#     •    android.intent.action.CALL: This is specifying the action to be performed, which in this case is to make a call.

#     •    -d <DATA_URI>: This flag is used to specify the data URI for the intent. In the case of tel:+91, it indicates that the data is a phone number, and +91 is the specific number to call.

#     •    tel:+91: This is specifying the data for the action, which in this case is the phone number to call. The +91 is the country code for India.