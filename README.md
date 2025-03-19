ต่อไปนี้คือไฟล์ Markdown (`.md`) ที่เหมาะสำหรับใช้ใน GitHub เพื่ออธิบายโปรเจกต์ "Tarot AI" ตามโค้ด Flask ที่คุณให้มา เนื้อหาจะครอบคลุมภาพรวม, การติดตั้ง, การใช้งาน, และโครงสร้างของโปรเจกต์ โดยออกแบบให้อ่านง่ายและเป็นมืออาชีพ:

```markdown
# Tarot AI

![Tarot AI Banner](https://via.placeholder.com/800x200.png?text=Tarot+AI+-+Mystical+Fortune-Telling+with+AI)  
*ผสานความลึกลับของไพ่ทาโรต์เข้ากับพลังของ AI เพื่อการทำนายที่แม่นยำและเป็นส่วนตัว*

---

## เกี่ยวกับโปรเจกต์

Tarot AI เป็นเว็บแอปพลิเคชันที่ให้ผู้ใช้สัมผัสประสบการณ์การดูดวงด้วยไพ่ทาโรต์ในรูปแบบทันสมัย โดยใช้ Flask เป็นเซิร์ฟเวอร์หลัก และผสานการวิเคราะห์อารมณ์ด้วย **mBERT** และการสร้างคำทำนายด้วย **GPT-2** เพื่อให้คำทำนายมีความฉลาดและเหมาะสมกับบริบทของผู้ใช้

### คุณสมบัติเด่น
- **เลือกไพ่แบบโต้ตอบ**: ผู้ใช้สามารถสับและเลือกไพ่ Major Arcana ได้เอง
- **วิเคราะห์อารมณ์**: ใช้ mBERT วิเคราะห์คำถาม เพื่อปรับคำทำนายให้เหมาะกับอารมณ์ (บวก/ลบ/เป็นกลาง)
- **คำทำนายส่วนตัว**: ดึงคำทำนายจากข้อมูล CSV หรือสร้างด้วย GPT-2 พร้อมคำอวยพรจากดวงดาว
- **หน้าเว็บและ API**: รองรับทั้งการใช้งานผ่านหน้าเว็บและการเชื่อมต่อผ่าน Dialogflow

---

## การติดตั้ง

### ความต้องการ
- Python 3.8+
- Flask, Pandas, Transformers, PyTorch
- GPU (แนะนำ แต่ไม่บังคับ)

### ขั้นตอนการติดตั้ง
1. **โคลนโปรเจกต์**
   ```bash
   git clone https://github.com/yourusername/tarot-ai.git
   cd tarot-ai
   ```

2. **สร้าง Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **ติดตั้ง Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **เตรียมข้อมูล**
   - วางไฟล์ `tarot_data_th.csv` ในโฟลเดอร์ `data/`  
   - รูปแบบคอลัมน์: `card_name`, `category`, `prediction`

5. **รันเซิร์ฟเวอร์**
   ```bash
   python app.py
   ```
   - เข้าใช้งานได้ที่ `http://localhost:5000`

### requirements.txt
```plaintext
flask
flask-cors
pandas
transformers
torch
```

---

## การใช้งาน

### หน้าเว็บ
- **หน้าแรก (`/`)**: แนะนำ Tarot AI
- **เลือกไพ่ (`/select`)**: เลือกไพ่ทาโรต์
- **แชท (`/chat`)**: ถามคำถามและรับคำทำนาย
- **คำทำนาย (`/prediction`)**: ดูผลการทำนาย
- **ประวัติ (`/history`)**: ดูประวัติการทำนาย (ผ่าน API)

### API
- **GET `/get_prediction`**  
  - **Request**: `POST` JSON เช่น `{"message": "งานดีไหม", "category": "career"}`
  - **Response**: 
    ```json
    {
      "prediction": "งานจะดีขึ้นในเร็วๆ นี้",
      "card": "The Sun",
      "category": "career",
      "sentiment": "บวก",
      "status": "success"
    }
    ```

- **POST `/rate_prediction`**  
  - บันทึกคะแนนคำทำนาย (0-5)

- **POST `/submit_review`**  
  - บันทึกรีวิวและคะแนน

- **GET `/history`**  
  - ดึงประวัติการทำนายตาม `session_id`

### ตัวอย่างคำถาม
- "วันนี้โชคดีไหม" → เลือกไพ่และรับคำทำนายพร้อมคำอวยพร
- "รักจะดีไหม (คนโสด)" → คำทำนายสำหรับหมวด `love_single`

---

## โครงสร้างโค้ด

```
tarot-ai/
├── data/
│   └── tarot_data_th.csv  # ข้อมูลไพ่ทาโรต์
├── static/               # ไฟล์ CSS, JS, รูปภาพ
├── templates/            # ไฟล์ HTML
│   └── index.html
├── app.py                # โค้ดหลัก
├── app.log               # ไฟล์ log
├── ratings.txt           # คะแนนคำทำนาย
├── reviews.txt           # รีวิว
└── requirements.txt      # Dependencies
```

### ฟังก์ชันหลัก
- **`load_tarot_data()`**: โหลดข้อมูลไพ่จาก CSV ด้วย caching
- **`analyze_sentiment()`**: วิเคราะห์อารมณ์ด้วย mBERT
- **`generate_gpt2_prediction()`**: สร้างคำทำนายด้วย GPT-2
- **`select_card()`**: เลือกไพ่ Major Arcana แบบสุ่ม
- **`get_prediction_or_response()`**: จัดการคำทำนายทั้งหมด

---

## การทำงานของระบบ

1. **โหลดข้อมูลและโมเดล**:
   - โหลด `tarot_data_th.csv` และโมเดล mBERT/GPT-2 แบบ asynchronous
2. **รับคำถาม**:
   - ผู้ใช้พิมพ์คำถามหรือเลือกหมวดหมู่ผ่านหน้าเว็บ/API
3. **วิเคราะห์และทำนาย**:
   - วิเคราะห์อารมณ์ → เลือกไพ่ → ดึงหรือสร้างคำทำนาย → เพิ่มคำอวยพร
4. **แสดงผล**:
   - ส่งคำทำนายกลับไปที่หน้าเว็บหรือ Dialogflow

---

## การพัฒนาเพิ่มเติม

- **ภาษา**: เพิ่มการแปลคำทำนายเป็นภาษาอื่น
- **UI**: ปรับปรุงหน้าเว็บให้สวยงามด้วย CSS/JS
- **ฐานข้อมูล**: เปลี่ยนจาก CSV เป็น MySQL หรือ MongoDB
- **โมเดล**: ใช้โมเดลภาษาไทย (เช่น WangchanBERT)

---

## การมีส่วนร่วม

1. Fork โปรเจกต์นี้
2. สร้าง branch (`git checkout -b feature/new-feature`)
3. Commit การเปลี่ยนแปลง (`git commit -m "Add new feature"`)
4. Push ไปที่ branch (`git push origin feature/new-feature`)
5. สร้าง Pull Request

---

## เครดิต

- **ทีมพัฒนา**: [ใส่ชื่อคุณหรือทีม]
- **โมเดล AI**: Hugging Face Transformers (mBERT, GPT-2)
- **เฟรมเวิร์ก**: Flask

---

## ใบอนุญาต

โปรเจกต์นี้อยู่ภายใต้ [MIT License](LICENSE)  
© 2025 [ใส่ชื่อคุณ]
```

---

### **คำอธิบายเพิ่มเติม**
- **ภาพ Banner**: ใช้ `via.placeholder.com` เป็นตัวอย่าง คุณสามารถแทนที่ด้วยลิงก์รูปจริง (เช่น จาก GitHub หรือที่อื่น)
- **โครงสร้าง**: จัดเรียงให้อ่านง่าย เริ่มจากภาพรวม → การติดตั้ง → การใช้งาน → รายละเอียดเทคนิค
- **การปรับแต่ง**: 
  - เปลี่ยน `yourusername` ใน URL GitHub เป็นชื่อผู้ใช้ของคุณ
  - เพิ่มเครดิตหรือลิงก์เพิ่มเติมตามต้องการ
- **ไฟล์อื่น**: อย่าลืมสร้าง `requirements.txt` และโฟลเดอร์ `data/`, `static/`, `templates/` ใน GitHub repository

ถ้าต้องการปรับแก้หรือเพิ่มส่วนใด เช่น ตัวอย่างคำทำนายจริง หรือใส่ลิงก์ demo แจ้งมาได้เลยครับ!
 
 
