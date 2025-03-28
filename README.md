
# Tarot AI

![Tarot AI Banner](https://via.placeholder.com/800x200.png?text=Tarot+AI+-+Mystical+Insights+Powered+by+AI)  
*สำรวจอนาคตของคุณด้วยไพ่ทาโรต์และปัญญาประดิษฐ์*

---

## ภาพรวม

Tarot AI เป็นเว็บแอปพลิเคชันที่นำศาสตร์แห่งไพ่ทาโรต์มาผสานกับเทคโนโลยี AI สมัยใหม่ โดยใช้ **Flask** เป็นเซิร์ฟเวอร์หลัก **mBERT** สำหรับวิเคราะห์อารมณ์ และ **GPT-2** สำหรับสร้างคำทำนายที่ไม่ซ้ำใคร ผู้ใช้สามารถเลือกไพ่ทาโรต์ ถามคำถาม และรับคำทำนายที่ปรับแต่งตามบริบทและอารมณ์ได้ในทันที

### คุณสมบัติเด่น
- **การเลือกไพ่แบบโต้ตอบ**: สับและเลือกไพ่ Major Arcana ได้ด้วยตัวเอง
- **การวิเคราะห์อารมณ์**: ใช้ mBERT ตรวจจับอารมณ์จากคำถาม (บวก/ลบ/เป็นกลาง)
- **คำทำนายส่วนตัว**: ดึงคำทำนายจากไฟล์ CSV หรือสร้างด้วย GPT-2 พร้อมคำอวยพรจากดวงดาว
- **การเชื่อมต่อหลากหลาย**: รองรับการใช้งานผ่านหน้าเว็บ, API, และการเชื่อมต่อกับ Dialogflow

---

## การติดตั้ง

### ความต้องการของระบบ
- **Python**: 3.8 หรือสูงกว่า
- **ไลบรารี**: Flask, Flask-CORS, Pandas, Transformers, PyTorch
- **ฮาร์ดแวร์**: GPU (แนะนำสำหรับประสิทธิภาพสูงสุด แต่ CPU ก็ใช้งานได้)
- **ไฟล์ข้อมูล**: `tarot_data_th.csv` (ต้องมีคอลัมน์ `card_name`, `category`, `prediction`)

### ขั้นตอนการติดตั้ง
1. **โคลน Repository**
   ```bash
   git clone https://github.com/yourusername/tarot-ai.git
   cd tarot-ai
   ```

2. **ตั้งค่า Virtual Environment**
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
   - สร้างโฟลเดอร์ `data/` ใน root directory
   - วางไฟล์ `tarot_data_th.csv` ในโฟลเดอร์ `data/`
   - **ตัวอย่างข้อมูล**:
     ```
     card_name,category,prediction
     The Fool,career,โอกาสใหม่ในงานกำลังรอคุณอยู่...
     The Star,love_single,ความหวังใหม่ในความรักกำลังก่อตัว...
     ```

5. **รันแอปพลิเคชัน**
   ```bash
   python app.py
   ```
   - เข้าใช้งานได้ที่: `http://localhost:5000`
   - Log การทำงานจะบันทึกใน `app.log`

### ไฟล์ `requirements.txt`
```plaintext
flask==2.3.3
flask-cors==4.0.0
pandas==2.2.1
transformers==4.38.2
torch==2.2.1
```

---

## การใช้งาน

### ผ่านหน้าเว็บ
| เส้นทาง         | คำอธิบาย                              |
|-----------------|---------------------------------------|
| `/`            | หน้าแรกของ Tarot AI                  |
| `/select`      | หน้าเลือกไพ่ทาโรต์                  |
| `/chat`        | แชทกับ Tarot AI เพื่อถามคำถาม         |
| `/prediction`  | แสดงผลคำทำนายและคำอวยพร             |
| `/article`     | ข้อมูลเกี่ยวกับไพ่ Major Arcana       |

### ผ่าน API
- **POST `/get_prediction`**  
  - **Request**:  
    ```json
    {
      "message": "วันนี้โชคดีไหม",
      "category": "luck",
      "relationship_status": "single",
      "session_id": "user123"
    }
    ```
  - **Response**:  
    ```json
    {
      "prediction": "โชคดีแน่นอน! ดวงดาวนำพาความสำเร็จมาให้",
      "card": "The Star",
      "category": "luck",
      "sentiment": "บวก",
      "date_mention": "วันที่ 18 มีนาคม 2568",
      "status": "success"
    }
    ```

- **POST `/rate_prediction`**  
  - **Request**:  
    ```json
    {
      "chatId": "user123",
      "messageId": "msg001",
      "rating": 4.5
    }
    ```
  - บันทึกคะแนน (0-5) ลง `ratings.txt`

- **POST `/submit_review`**  
  - **Request**:  
    ```json
    {
      "review": "คำทำนายแม่นมาก!",
      "rating": 5
    }
    ```
  - บันทึกรีวิวลง `reviews.txt`

- **GET `/history?session_id=user123`**  
  - ดึงประวัติการทำนายของผู้ใช้

### ตัวอย่างการใช้งาน
- **คำถาม**: "งานจะดีไหมในสัปดาห์นี้"  
  - **ผลลัพธ์**:  
    ```
    ไพ่: The Sun
    หมวด: การงาน
    อารมณ์: บวก
    ทำนาย: งานของคุณจะรุ่งเรืองในสัปดาห์นี้ มีโอกาสใหม่เข้ามา!
    คำอวยพร: ขอให้ดวงดาวนำความสำเร็จมาสู่คุณในวันที่ 18 มีนาคม 2568
    ```

---

## โครงสร้างโปรเจกต์

```
tarot-ai/
├── data/
│   └── tarot_data_th.csv   # ข้อมูลคำทำนายไพ่ทาโรต์
├── static/
│   ├── css/                # ไฟล์ CSS
│   └── images/             # รูปภาพไพ่และกราฟิก
├── templates/
│   └── index.html          # เทมเพลต HTML
├── app.py                  # โค้ดหลักของแอปพลิเคชัน
├── app.log                 # ไฟล์ log การทำงาน
├── ratings.txt             # บันทึกคะแนนคำทำนาย
├── reviews.txt             # บันทึกรีวิวจากผู้ใช้
└── requirements.txt        # รายการ Dependencies
```

### ฟังก์ชันหลัก
- **`load_tarot_data()`**: โหลดข้อมูลไพ่จาก CSV พร้อม caching และการตรวจสอบไฟล์
- **`analyze_sentiment()`**: วิเคราะห์อารมณ์จากข้อความด้วย mBERT
- **`generate_gpt2_prediction()`**: สร้างคำทำนายหรือคำอวยพรด้วย GPT-2
- **`select_card()`**: สุ่มเลือกไพ่ Major Arcana
- **`generate_star_blessing()`**: สร้างคำอวยพรจากดวงดาวตามไพ่และหมวดหมู่

---

## การทำงานของระบบ

1. **เริ่มต้นระบบ**:
   - โหลดข้อมูลจาก `tarot_data_th.csv` และโมเดล AI (mBERT, GPT-2) แบบ asynchronous ด้วย `ThreadPoolExecutor`
2. **รับคำถาม**:
   - ผู้ใช้ป้อนคำถามผ่านหน้า `/chat` หรือ API
3. **ประมวลผล**:
   - วิเคราะห์อารมณ์ด้วย mBERT
   - เลือกไพ่ด้วย `select_card()`
   - กำหนดหมวดหมู่จากคำถามหรือผู้ใช้ระบุ
   - ดึงคำทำนายจาก CSV หรือสร้างด้วย GPT-2
   - เพิ่มคำอวยพรด้วย `generate_star_blessing()`
4. **ส่งผลลัพธ์**:
   - แสดงผลผ่านหน้าเว็บหรือคืนค่าในรูปแบบ JSON

### ตัวอย่างการประมวลผล
```
คำถาม: "วันนี้โชคดีไหม"
1. วิเคราะห์อารมณ์: บวก (mBERT)
2. เลือกไพ่: The Star
3. หมวด: โชคลาภ
4. ทำนาย: ความหวังและโชคลาภกำลังส่องสว่าง โอกาสดีๆ รอคุณอยู่!
5. คำอวยพร: ขอให้ดวงดาวนำโชคมาให้คุณในวันที่ 18 มีนาคม 2568
```

---

## การพัฒนาเพิ่มเติม

- **การออกแบบ**: อัปเกรด UI ด้วยเฟรมเวิร์กเช่น Bootstrap หรือ Tailwind CSS
- **โมเดล AI**: เปลี่ยนไปใช้โมเดลภาษาไทย เช่น WangchanBERT เพื่อความแม่นยำในภาษาไทย
- **ฐานข้อมูล**: แทนที่ CSV ด้วย SQLite หรือ MongoDB เพื่อการจัดการข้อมูลที่ดีขึ้น
- **ฟีเจอร์ใหม่**: เพิ่มการแปลหลายภาษา หรือการแจ้งเตือนคำทำนายประจำวัน

---

## การมีส่วนร่วม

เรายินดีต้อนรับทุกไอเดียและการช่วยเหลือ!
1. Fork โปรเจกต์นี้
2. สร้าง branch:  
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit การเปลี่ยนแปลง:  
   ```bash
   git commit -m "Add your feature description"
   ```
4. Push ไปที่ branch:  
   ```bash
   git push origin feature/your-feature-name
   ```
5. สร้าง Pull Request บน GitHub

### รายงานปัญหา
หากพบข้อบกพร่อง โปรดสร้าง [Issue](https://github.com/yourusername/tarot-ai/issues) พร้อมระบุ:
- ขั้นตอนที่ทำให้เกิดปัญหา
- ข้อความ error (ถ้ามี)
- สภาพแวดล้อม (เช่น Python version, OS)

---

## เครดิต

- **ผู้พัฒนา**: [ใส่ชื่อคุณหรือทีม]
- **เทคโนโลยี**:
  - [Flask](https://flask.palletsprojects.com/): Web Framework
  - [Transformers](https://huggingface.co/): mBERT และ GPT-2 จาก Hugging Face
  - [PyTorch](https://pytorch.org/): Machine Learning Framework

---

## ใบอนุญาต

โปรเจกต์นี้อยู่ภายใต้ [MIT License](LICENSE)  
© 2025 [ใส่ชื่อคุณ]
```

---

### **การปรับปรุงในเวอร์ชันนี้**
1. **ภาพรวมที่ชัดเจนขึ้น**:
   - ปรับคำอธิบายให้กระชับและเน้นจุดเด่นของโปรเจกต์
   - เปลี่ยนคำขวัญเป็น "Mystical Insights Powered by AI" เพื่อความลึกลับและทันสมัย
2. **การติดตั้งที่ละเอียด**:
   - เพิ่มคำแนะนำเกี่ยวกับโฟลเดอร์ `data/` และตัวอย่าง CSV
   - อธิบายการรันและการดู log
3. **การใช้งานที่ครบถ้วน**:
   - เพิ่มตัวอย่าง JSON Request/Response ใน API
   - ปรับตารางหน้าเว็บให้อ่านง่าย
4. **โครงสร้างและการทำงาน**:
   - อธิบายขั้นตอนการทำงานแบบ step-by-step
   - เพิ่มตัวอย่างคำทำนายที่สมบูรณ์
5. **การมีส่วนร่วม**:
   - เพิ่มคำแนะนำการรายงานปัญหา พร้อมตัวอย่างข้อมูลที่ต้องการ
6. **เครดิตและลิงก์**:
   - ใส่ลิงก์ไปยัง Flask, Transformers, PyTorch เพื่อให้ผู้อ่านสำรวจได้

### **คำแนะนำเพิ่มเติม**
- **รูปภาพ**: อัปโหลดภาพจริงของหน้าเว็บหรือไพ่ทาโรต์ไปที่ `static/images/` แล้วเปลี่ยน URL ใน `![Tarot AI Banner]`
- **เดโม**: ถ้ามีลิงก์เดโม (เช่น Heroku, Replit) เพิ่มในส่วน "การใช้งาน" เช่น:  
  ```markdown
  ### ลองใช้งานเดโม
  เข้าไปทดลองได้ที่: [https://tarot-ai-demo.herokuapp.com](https://tarot-ai-demo.herokuapp.com)
  ```


