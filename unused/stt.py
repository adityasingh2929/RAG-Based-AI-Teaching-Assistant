import whisper
import torch
import json

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model = whisper.load_model("small").to("cuda")

result = model.transcribe(
    audio="audios/sample.mp3",
    language="hi",
    task="translate"
)

chunks=[]

for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end" : segment["end"], "text" : segment["text"]})


with open("output.json", "w") as f:
    json.dump(chunks,f)


print("\n\ndone")

