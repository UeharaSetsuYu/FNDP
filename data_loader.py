from datasets import load_dataset
import os

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"



dataset = load_dataset(
    "anson-huang/mirage-news",
    split="train",
    streaming=True
)

'''
    0: fake
    1: real
'''

for i, sample in enumerate(dataset):
    print("Sample:", i)
    print("Text:", sample["text"])
    print("Label:", sample["label"])
    print("Image:", sample["image"])
    print("-" * 50)


    print(i + 1)






