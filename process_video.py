import os 
import subprocess

files = os.listdir("videos")
for file in files:
    file_number = file.split(". ")[0]
    file_name = file.split(". ")[1].split("_Sigma")[0]
    print(file_number, file_name)
    print()

    subprocess.run(["ffmpeg","-i",f"videos/{file}",f"audios/{file_number}_{file_name}.mp3"])  # its just the same command that I ran in terminal which was...
                                                                                              # ffmpeg -i ".\filename.ext" 3.mp3    [for example].