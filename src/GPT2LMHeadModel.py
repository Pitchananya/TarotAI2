import pandas as pd
import tensorflow as tf
import sentencepiece as spm
from transformers import TFGPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import Dataset
import shutil
import os
import psutil  # สำหรับตรวจสอบพื้นที่ดิสก์

# ตรวจสอบว่า TensorFlow เห็น GPU หรือไม่
print("TensorFlow Version:", tf.__version__)
print("GPU Availabpip uninstall tensorflowle:", tf.config.list_physical_devices("GPU"))

# กำหนดพาธของไฟล์ข้อมูลและผลลัพธ์
DATA_PATH = r"D:\TarotAI\TarotAI\data\tarot_predictions.csv"
OUTPUT_DIR = r"D:\TarotAI\TarotAI\src\llama_tarot_model"
ALTERNATE_DIR = r"C:\TarotAI\llama_tarot_model"  # พาธสำรองถ้า D: เต็ม

# ตรวจสอบพื้นที่ดิสก์
def check_disk_space(path):
    disk = psutil.disk_usage(path)
    free_space_gb = disk.free / (1024 ** 3)
    print(f"Free space on {path}: {free_space_gb:.2f} GB")
    return free_space_gb > 2  # ต้องการอย่างน้อย 2 GB

if not check_disk_space(os.path.splitdrive(OUTPUT_DIR)[0] + "\\"):
    print(f"Warning: Insufficient space on {os.path.splitdrive(OUTPUT_DIR)[0]}. Trying alternate directory {ALTERNATE_DIR}")
    if check_disk_space(os.path.splitdrive(ALTERNATE_DIR)[0] + "\\"):
        OUTPUT_DIR = ALTERNATE_DIR
    else:
        raise Exception("No disk with sufficient space found!")

# 1️⃣ โหลดและเตรียม Dataset
def load_tarot_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    if "prediction" in df.columns:
        texts = df["prediction"].tolist()
    elif "text" in df.columns:
        texts = df["text"].tolist()
    else:
        raise KeyError("ไม่พบคอลัมน์ 'prediction' หรือ 'text' ในไฟล์ CSV")
    texts = [t for t in texts if pd.notna(t) and t != ""]
    print(f"Loaded {len(texts)} valid text samples")
    return Dataset.from_dict({"text": texts})

dataset = load_tarot_data()

# 2️⃣ สร้างและเทรน Tokenizer
tokenizer_model_prefix = "tarot_tokenizer"
tokenizer_train_data = "tarot_texts.txt"

with open(tokenizer_train_data, "w", encoding="utf-8") as f:
    for text in dataset["text"]:
        f.write(text + "\n")

spm.SentencePieceTrainer.train(
    input=tokenizer_train_data,
    model_prefix=tokenizer_model_prefix,
    vocab_size=3000,
    model_type="bpe",
    character_coverage=1.0
)
sp = spm.SentencePieceProcessor(model_file=f"{tokenizer_model_prefix}.model")
print("Tokenizer training completed")

# 3️⃣ สร้าง Config สำหรับโมเดล GPT-2
config = GPT2Config(
    vocab_size=3000,
    n_embd=512,  # เทียบเท่า hidden_size
    n_layer=6,   # เทียบเท่า num_hidden_layers
    n_head=8,    # เทียบเท่า num_attention_heads
    n_positions=2048,  # เทียบเท่า max_position_embeddings
    use_cache=False,   # ปิดการใช้ cache เพื่อให้เข้ากันได้กับ gradient checkpointing
)

# 4️⃣ สร้างโมเดลใหม่ (ใช้ TFGPT2LMHeadModel)
model = TFGPT2LMHeadModel(config)

# 5️⃣ เตรียมข้อมูลสำหรับการเทรน
def tokenize_function(examples):
    input_ids_list = []
    attention_mask_list = []
    for text in examples["text"]:
        if not text or pd.isna(text):
            input_ids = [0] * 2048
            attention_mask = [0] * 2048
        else:
            input_ids = sp.encode(text, out_type=int)
            if len(input_ids) < 2048:
                padding = [0] * (2048 - len(input_ids))
                input_ids += padding
                attention_mask = [1] * len(input_ids) + padding
            elif len(input_ids) > 2048:
                input_ids = input_ids[:2048]
                attention_mask = [1] * 2048
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
    return {"input_ids": input_ids_list, "attention_mask": attention_mask_list}

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# แยก Train/Test
train_size = int(0.9 * len(tokenized_dataset))
train_dataset = tokenized_dataset.select(range(train_size))
eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
print(f"Training dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")

# แปลง Dataset เป็น tf.data.Dataset
def convert_to_tf_dataset(dataset):
    def gen():
        for item in dataset:
            yield {
                "input_ids": tf.constant(item["input_ids"], dtype=tf.int32),
                "attention_mask": tf.constant(item["attention_mask"], dtype=tf.int32),
                "labels": tf.constant(item["input_ids"], dtype=tf.int32),  # GPT-2 ใช้ input_ids เป็น labels
            }
    return tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "input_ids": tf.TensorSpec(shape=(2048,), dtype=tf.int32),
            "attention_mask": tf.TensorSpec(shape=(2048,), dtype=tf.int32),
            "labels": tf.TensorSpec(shape=(2048,), dtype=tf.int32),
        }
    ).batch(1).prefetch(tf.data.AUTOTUNE)

tf_train_dataset = convert_to_tf_dataset(train_dataset)
tf_eval_dataset = convert_to_tf_dataset(eval_dataset)

# 6️⃣ กำหนด optimizer และ compile โมเดล
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)

# 7️⃣ เริ่มเทรน
EPOCHS = 3
history = model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    epochs=EPOCHS,
    steps_per_epoch=len(train_dataset) // 1,  # ปรับตาม batch size
    validation_steps=len(eval_dataset) // 1,
)

# 8️⃣ บันทึกโมเดลและ tokenizer
try:
    print(f"Attempting to save model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    print(f"Attempting to save config to {OUTPUT_DIR}")
    config.save_pretrained(OUTPUT_DIR)
    print(f"Attempting to copy tokenizer to {OUTPUT_DIR}")
    shutil.copy(f"{tokenizer_model_prefix}.model", os.path.join(OUTPUT_DIR, "tarot_tokenizer.model"))

    # ตรวจสอบว่าไฟล์ถูกบันทึกจริง
    required_files = ["tf_model.h5", "config.json", "tarot_tokenizer.model"]
    for file in required_files:
        file_path = os.path.join(OUTPUT_DIR, file)
        if os.path.exists(file_path):
            print(f"Successfully saved {file} in {OUTPUT_DIR}")
        else:
            raise FileNotFoundError(f"Failed to save {file} in {OUTPUT_DIR} after saving attempt")
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")
except Exception as e:
    print(f"Error during saving: {str(e)}")
    if "No space left on device" in str(e) or "Permission denied" in str(e):
        print(f"Attempting to save to alternate directory: {ALTERNATE_DIR}")
        OUTPUT_DIR = ALTERNATE_DIR
        try:
            model.save_pretrained(OUTPUT_DIR)
            config.save_pretrained(OUTPUT_DIR)
            shutil.copy(f"{tokenizer_model_prefix}.model", os.path.join(OUTPUT_DIR, "tarot_tokenizer.model"))
            for file in required_files:
                file_path = os.path.join(OUTPUT_DIR, file)
                if os.path.exists(file_path):
                    print(f"Successfully saved {file} in {OUTPUT_DIR}")
                else:
                    raise FileNotFoundError(f"Failed to save {file} in {OUTPUT_DIR}")
            print(f"Model and tokenizer saved to {OUTPUT_DIR}")
        except Exception as e2:
            print(f"Failed to save to alternate directory: {str(e2)}")
            raise
    raise