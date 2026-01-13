import whisper
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model = whisper.load_model("small").to("cuda")

result = model.transcribe(
    audio="audios/1_Installing_VS_Code_How_Websites_Work.mp3",
    language="hi",
    task="translate"
)

print(result["text"])
print("\n\ndone")
