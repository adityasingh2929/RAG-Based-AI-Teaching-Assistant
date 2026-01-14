import whisper
import torch
import json
import os

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model = whisper.load_model("small").to("cuda")

audios = os.listdir("audios")
for audio in audios:
    if("_" in audio):
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        print(number,title)
        print()
    result = model.transcribe(
        audio= f"audios/{audio}",
        language="hi",
        task="translate"
    )

    chunks=[]

    for segment in result["segments"]:
        chunks.append({"number" : number, "title" : title, "start": segment["start"], "end" : segment["end"], "text" : segment["text"]})

    chunks_with_metadata = {"chunks" : chunks, "text" : result["text"]}

    with open(f"jsons/{title}", "w") as f:
        json.dump(chunks_with_metadata,f)


    print("\n\ndone")

 