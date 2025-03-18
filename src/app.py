from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import pandas as pd
import random
from datetime import datetime, timedelta
import logging
import logging.handlers
import re
import os
from werkzeug.exceptions import BadRequest, InternalServerError
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
import torch
from functools import lru_cache
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import atexit
from collections import defaultdict

# ตั้งค่าแอปพลิเคชัน Flask
app = Flask(__name__)
CORS(app)

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler('app.log', maxBytes=10000, backupCount=5, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Executor สำหรับการโหลดโมเดลแบบ async
executor = ThreadPoolExecutor(max_workers=2)

# เก็บประวัติการทำนายชั่วคราว
prediction_history = defaultdict(list)

# ฟังก์ชันโหลดข้อมูล CSV ด้วยการรีโหลดอัตโนมัติ
@lru_cache(maxsize=1)
def load_tarot_data():
    def _load():
        file_path = os.path.join('data', 'tarot_data_th.csv')
        last_modified = 0
        while True:
            try:
                if os.path.exists(file_path):
                    current_modified = os.path.getmtime(file_path)
                    if current_modified > last_modified:
                        df = pd.read_csv(file_path, encoding='utf-8-sig')
                        if df.empty or not isinstance(df, pd.DataFrame):
                            logger.warning("ไฟล์ tarot_data_th.csv ว่างเปล่า")
                            return pd.DataFrame(columns=['card_name', 'category', 'prediction'])
                        required_columns = ['card_name', 'category', 'prediction']
                        if not all(col in df.columns for col in required_columns):
                            missing_cols = [col for col in required_columns if col not in df.columns]
                            logger.error(f"ขาดคอลัมน์: {missing_cols}")
                            return pd.DataFrame(columns=required_columns)
                        last_modified = current_modified
                        logger.info(f"โหลด tarot_data_th.csv สำเร็จ จำนวนแถว: {len(df)}")
                        return df
                    else:
                        logger.debug("ไฟล์ไม่มีการเปลี่ยนแปลง")
                else:
                    logger.error(f"ไม่พบ {file_path} ใน {os.getcwd()}")
                    return pd.DataFrame(columns=['card_name', 'category', 'prediction'])
            except pd.errors.ParserError as pe:
                logger.error(f"ParserError: {str(pe)}")
                return pd.DataFrame(columns=['card_name', 'category', 'prediction'])
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาด: {str(e)}", exc_info=True)
                time.sleep(5)  # รีทรีทุก 5 วินาที
    return _load()

# โหลดข้อมูลทันที
df = load_tarot_data()

# หมวดหมู่
category_translations = {
    "love_single": "ความรัก (คนโสด)",
    "love_in_relationship": "ความรัก (คนมีคู่)",
    "career": "การงาน",
    "finance": "การเงิน",
    "health": "สุขภาพ",
    "personality": "บุคลิกภาพ",
    "education": "การเรียน",
    "luck": "โชคลาภ",
    "general": "ทั่วไป"
}

# ฟังก์ชันโหลดโมเดลแบบ async
def load_model(model_func, name):
    for attempt in range(3):  # ลอง 3 ครั้ง
        try:
            tokenizer, model = model_func()
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
                logger.info(f"ใช้ GPU สำหรับ {name}")
            else:
                logger.info(f"ใช้ CPU สำหรับ {name}")
            logger.info(f"โหลด {name} สำเร็จ")
            return tokenizer, model
        except Exception as e:
            logger.error(f"ลองที่ {attempt + 1} สำหรับ {name} ล้มเหลว: {str(e)}", exc_info=True)
            time.sleep(2 ** attempt)  # รอเพิ่มขึ้นตามเลขชี้กำลัง
    logger.critical(f"ไม่สามารถโหลด {name} ได้หลัง 3 ครั้ง")
    return None, None

@lru_cache(maxsize=1)
def load_mbert_model():
    return load_model(lambda: (BertTokenizer.from_pretrained('bert-base-multilingual-cased'),
                              BertModel.from_pretrained('bert-base-multilingual-cased')), "mBERT")

@lru_cache(maxsize=1)
def load_gpt2_model():
    return load_model(lambda: (GPT2Tokenizer.from_pretrained('gpt2'),
                              GPT2LMHeadModel.from_pretrained('gpt2')), "GPT-2")

# โหลดโมเดลใน background
def init_models():
    global mbert_tokenizer, mbert_model, gpt2_tokenizer, gpt2_model
    mbert_tokenizer, mbert_model = load_mbert_model()
    gpt2_tokenizer, gpt2_model = load_gpt2_model()

executor.submit(init_models)

# ฟังก์ชันวิเคราะห์อารมณ์
@lru_cache(maxsize=128)
def analyze_sentiment(text):
    if not text or not isinstance(text, str) or not mbert_tokenizer or not mbert_model:
        return "ไม่สามารถวิเคราะห์ได้", 0.0
    try:
        inputs = mbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = mbert_model(**inputs)
            sentiment_score = torch.mean(outputs.last_hidden_state[:, 0, :]).item()
        sentiment = "บวก" if sentiment_score > 0.2 else "ลบ" if sentiment_score < -0.2 else "เป็นกลาง"
        return sentiment, sentiment_score
    except Exception as e:
        logger.error(f"วิเคราะห์อารมณ์ล้มเหลว: {str(e)}")
        return "ไม่สามารถวิเคราะห์ได้", 0.0

# ฟังก์ชันสร้างคำทำนายด้วย GPT-2
@lru_cache(maxsize=128)
def generate_gpt2_prediction(prompt, max_length=100):
    if not gpt2_tokenizer or not gpt2_model:
        logger.error("โมเดล GPT-2 ไม่พร้อมใช้งาน")
        return "ขออภัย เกิดข้อผิดพลาดในการสร้างคำทำนาย กรุณาลองใหม่ภายหลัง"
    try:
        inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=50)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs = gpt2_model.generate(
            inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2,
            do_sample=True, top_k=50, top_p=0.95, temperature=0.7
        )
        return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        logger.error(f"GPT-2 ล้มเหลว: {str(e)}")
        return "ขออภัย เกิดข้อผิดพลาดในการสร้างคำทำนาย กรุณาลองใหม่ภายหลัง"

# ฟังก์ชันเลือกไพ่
@lru_cache(maxsize=1)
def select_card():
    major_arcana = [
        "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
        "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
        "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
        "The Devil", "The Tower", "The Star", "The Moon", "The Sun",
        "Judgement", "The World"
    ]
    available_cards = [card for card in major_arcana if card in df['card_name'].values]
    return random.choice(available_cards) if available_cards else random.choice(major_arcana)

# ฟังก์ชันกำหนดหมวดหมู่ (ปรับปรุง)
@lru_cache(maxsize=128)
def determine_category(message, specified_category=None):
    if specified_category and specified_category in category_translations:
        return specified_category
    if not message or not isinstance(message, str):
        return "general"
    message_lower = message.lower()
    categories = {
        "career": ["งาน", "การงาน", "อาชีพ", "ธุรกิจ", "เจ้านาย"],
        "love": ["รัก", "ความรัก", "แฟน", "คนรัก", "คู่", "โสด", "ความสัมพันธ์"],
        "education": ["เรียน", "การเรียน", "สอบ", "โรงเรียน", "มหาวิทยาลัย"],
        "finance": ["เงิน", "การเงิน", "รายได้", "หนี้", "ลงทุน"],
        "health": ["สุขภาพ", "ป่วย", "เจ็บ", "โรค"],
        "luck": ["โชค", "ดวง", "โชคลาภ", "ลาภลอย"]
    }
    if any(keyword in message_lower for keyword in ["ดูดวง", "ไพ่", "ยิปซี", "ทาโรต์"]):
        for category, keywords in categories.items():
            if any(keyword in message_lower for keyword in keywords):
                return category
    return "general"  # ค่าเริ่มต้นถ้าไม่เจอหมวดหมู่

# ฟังก์ชันอธิบาย
@lru_cache(maxsize=128)
def explain_response(message):
    return "ฉันวิเคราะห์คำถามของคุณโดยใช้ AI เพื่อหาความหมายและบริบท จากนั้นเลือกไพ่ทาโรต์และหมวดหมู่เพื่อทำนายที่แม่นยำค่ะ"

# ฟังก์ชันสร้างคำอวยพรตามดวงดาว (ปรับปรุง)
@lru_cache(maxsize=128)
def generate_star_blessing(card, category):
    if not gpt2_tokenizer or not gpt2_model:
        logger.warning("โมเดล GPT-2 ไม่พร้อมใช้งาน ใช้ข้อความสำรอง")
        # ข้อความสำรองที่สร้างขึ้นเองตามหมวดหมู่
        fallback_messages = {
            "education": f"ดวงดาวหนุนนำให้คุณมุ่งมั่นกับการเรียนในวันที่ {datetime.now().strftime('%d %B %Y')} ค่ะ!",
            "career": f"ขอให้ดวงดาวนำพาความสำเร็จในงานมาสู่คุณในวันที่ {datetime.now().strftime('%d %B %Y')} ค่ะ!",
            "love": f"ดวงดาวส่งพลังบวกให้ความรักของคุณในวันที่ {datetime.now().strftime('%d %B %Y')} ค่ะ!",
            "finance": f"ขอให้ดวงดาวช่วยให้การเงินคล่องตัวในวันที่ {datetime.now().strftime('%d %B %Y')} ค่ะ!",
            "health": f"ดวงดาวปกป้องสุขภาพของคุณในวันที่ {datetime.now().strftime('%d %B %Y')} ค่ะ!",
            "luck": f"ขอให้ดวงดาวนำโชคลาภมาให้คุณในวันที่ {datetime.now().strftime('%d %B %Y')} ค่ะ!",
            "general": f"ขอให้ดวงดาวนำความสุขมาให้คุณในวันที่ {datetime.now().strftime('%d %B %Y')} ค่ะ!"
        }
        return fallback_messages.get(category, "ขอให้โชคดีจากดวงดาว!")
    
    today = datetime.now().strftime("%d %B %Y")
    prompt = f"คำอวยพรสั้นๆ จากดวงดาวในวันที่ {today} สำหรับไพ่ {card} ในหมวด {category_translations.get(category, category)}"
    blessing = generate_gpt2_prediction(prompt, max_length=40)
    if "เกิดข้อผิดพลาด" in blessing:
        logger.warning("GPT-2 สร้างคำอวยพรล้มเหลว ใช้ข้อความสำรอง")
        fallback_messages = {
            "education": f"ดวงดาวหนุนนำให้คุณมุ่งมั่นกับการเรียนในวันที่ {today} ค่ะ!",
            "career": f"ขอให้ดวงดาวนำพาความสำเร็จในงานมาสู่คุณในวันที่ {today} ค่ะ!",
            "love": f"ดวงดาวส่งพลังบวกให้ความรักของคุณในวันที่ {today} ค่ะ!",
            "finance": f"ขอให้ดวงดาวช่วยให้การเงินคล่องตัวในวันที่ {today} ค่ะ!",
            "health": f"ดวงดาวปกป้องสุขภาพของคุณในวันที่ {today} ค่ะ!",
            "luck": f"ขอให้ดวงดาวนำโชคลาภมาให้คุณในวันที่ {today} ค่ะ!",
            "general": f"ขอให้ดวงดาวนำความสุขมาให้คุณในวันที่ {today} ค่ะ!"
        }
        return fallback_messages.get(category, "ขอให้โชคดีจากดวงดาว!")
    return blessing

# ฟังก์ชันดึงคำทำนายหรือคำตอบ (ปรับปรุง)
@lru_cache(maxsize=128)
def get_prediction_or_response(card=None, category=None, message="", relationship_status=None, is_explanation=False, use_gpt2=False):
    if is_explanation:
        return explain_response(message), True, category or "general", ""
    
    # ขั้นตอน 1: กำหนดหมวดหมู่และไพ่
    final_category = determine_category(message, category)
    if not card:
        card = select_card()

    # ขั้นตอน 2: ถ้าใช้ GPT-2 โดยตรง
    if use_gpt2 and gpt2_tokenizer and gpt2_model:
        prompt = f"ทำนายดวงด้วยไพ่ทาโรต์ '{card}' ในหมวด {category_translations.get(final_category, final_category)}: {message}"
        prediction = generate_gpt2_prediction(prompt)
        sentiment, sentiment_score = analyze_sentiment(message)
        logger.info(f"GPT-2 อารมณ์: {sentiment} (คะแนน: {sentiment_score})")
        return prediction, False, final_category, ""

    # ขั้นตอน 3: ดึงคำทำนายจากข้อมูล CSV
    card_data = df[df['card_name'] == card]
    if card_data.empty:
        logger.error(f"ไม่พบข้อมูลไพ่ {card}")
        return f"ขออภัย ไม่พบข้อมูลไพ่ {card}", False, final_category, ""

    # ปรับหมวดหมู่ความรักตามสถานะ
    if final_category == "love" and relationship_status in ["single", "in_relationship"]:
        final_category = f"love_{relationship_status}"

    # ตรวจสอบว่ามีหมวดหมู่ในข้อมูลหรือไม่
    if final_category not in card_data['category'].values:
        logger.warning(f"ไม่พบคำทำนายสำหรับ {card} ใน {final_category}")
        return f"ขออภัย ไม่พบคำทำนายสำหรับ {category_translations.get(final_category, final_category)} ในไพ่ {card}", False, final_category, ""

    prediction_row = card_data[card_data['category'] == final_category]
    if prediction_row.empty:
        logger.warning(f"ไม่พบคำทำนายสำหรับ {card} ใน {final_category}")
        return f"ขออภัย ไม่พบคำทำนายสำหรับ {category_translations.get(final_category, final_category)} ในไพ่ {card}", False, final_category, ""

    prediction = prediction_row['prediction'].iloc[0]

    # ขั้นตอน 4: วิเคราะห์อารมณ์และปรับแต่งคำทำนาย
    sentiment, sentiment_score = analyze_sentiment(message)
    if sentiment == "ลบ" and sentiment_score < -0.3:
        prediction += " อย่ากังวลไปค่ะ ทุกอย่างจะดีขึ้น!"
    elif sentiment == "บวก" and sentiment_score > 0.3:
        prediction += " อารมณ์ดีมาก! เพิ่มพลังบวกให้คุณ!"

    # ขั้นตอน 5: เพิ่มคำอวยพรตามดวงดาว
    star_blessing = generate_star_blessing(card, final_category)
    prediction += f"\n\n**คำอวยพรจากดวงดาว**: {star_blessing}"

    # ขั้นตอน 6: จัดการวันที่ (ถ้ามี)
    date_mention = ""
    date_match = re.search(r"วันที่\s*(\d+)", message) or re.search(r"(วันนี้|พรุ่งนี้)", message)
    if date_match:
        day = date_match.group(1) if date_match.group(1) else date_match.group(0)
        current_date = datetime.now()
        target_date = current_date if day == "วันนี้" else current_date + timedelta(days=1) if day == "พรุ่งนี้" else current_date
        date_mention = f"วันที่ {target_date.strftime('%d %B %Y')} "
        date_mention += {
            "love_single": "ลองเปิดใจรับโอกาสใหม่ๆ ค่ะ",
            "love_in_relationship": "สานสัมพันธ์กับคู่ค่ะ",
            "education": "เตรียมตัวเรียนหรือสอบค่ะ",
            "career": "โฟกัสงานและทักษะใหม่ค่ะ",
            "finance": "วางแผนการเงินค่ะ",
            "health": "ดูแลสุขภาพค่ะ",
            "luck": "ลองเสี่ยงโชคเล็กน้อยค่ะ",
            "general": "เตรียมตัววันใหม่ค่ะ"
        }.get(final_category, "") + "\n"

    return prediction, False, final_category, date_mention

# Webhook สำหรับ Dialogflow
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        req = request.get_json(silent=True, force=True)
        if not req or 'queryResult' not in req:
            raise BadRequest("ข้อมูลจาก Dialogflow ไม่ถูกต้อง")
        intent = req['queryResult'].get('intent', {}).get('displayName', '')
        query_text = req['queryResult'].get('queryText', '').strip()
        parameters = req['queryResult'].get('parameters', {})
        session_id = req.get('session', '').split('/')[-1]
        category = parameters.get('category', '').lower()
        relationship_status = parameters.get('relationship_status', '').lower()
        use_gpt2 = parameters.get('use_gpt2', False)
        logger.info(f"Session: {session_id}, Intent: {intent}, Query: {query_text}")

        responses = {
            "Default Welcome Intent": "สวัสดี! ฉันคือ Tarot AI พิมพ์ 'เลือกไพ่' หรือ 'ทำนาย' ค่ะ",
            "ExplainPrediction": explain_response(query_text),
            "RefreshCard": f"ไพ่ใหม่: {select_card()} ถามใหม่เพื่อทำนายค่ะ"
        }
        prediction, is_explanation, final_category, date_mention = get_prediction_or_response(
            category=category, message=query_text, relationship_status=relationship_status, use_gpt2=use_gpt2
        )
        if intent in ["PredictFuture", "SelectCard"]:
            if not is_explanation and final_category:
                sentiment, _ = analyze_sentiment(query_text)
                category_display = category_translations.get(final_category, final_category)
                response_text = f"ไพ่: {select_card()}\nหมวด: {category_display}\nอารมณ์: {sentiment}\n{date_mention}ทำนาย: {prediction}"
            else:
                response_text = prediction
        else:
            response_text = responses.get(intent, prediction)
        return jsonify({"fulfillmentText": response_text, "source": "webhook", "session": session_id})
    except BadRequest as e:
        logger.error(f"BadRequest: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}", exc_info=True)
        raise InternalServerError("เซิร์ฟเวอร์มีปัญหา")

# เส้นทางหน้าเว็บ
@app.route('/')
def home():
    return render_template('index.html', page='home')

@app.route('/chat')
def chat():
    return render_template('index.html', page='chat', auto_refresh=True)

@app.route('/select')
def select():
    return render_template('index.html', page='select')

@app.route('/loading')
def loading():
    return render_template('index.html', page='loading')

@app.route('/prediction')
def prediction():
    try:
        message = request.args.get('message', 'ดูดวงทั่วไป').strip()
        relationship_status = request.args.get('relationship_status', '').lower()
        card = request.args.get('card', '')
        category = request.args.get('category', '')
        use_gpt2 = request.args.get('use_gpt2', 'false').lower() == 'true'

        if not message:
            raise BadRequest("กรุณาระบุคำถาม")
        if relationship_status not in ["single", "in_relationship"]:
            relationship_status = None
        prediction, is_explanation, final_category, date_mention = get_prediction_or_response(
            card=card, category=category, message=message, relationship_status=relationship_status, use_gpt2=use_gpt2
        )
        sentiment, _ = analyze_sentiment(message)

        # บันทึกประวัติ
        session_id = request.args.get('session_id', 'default')
        prediction_history[session_id].append({
            'message': message,
            'prediction': prediction,
            'card': card or select_card(),
            'category': final_category,
            'timestamp': datetime.now().isoformat()
        })

        return render_template(
            'index.html',
            page='prediction',
            card=card or select_card(),
            category=final_category,
            prediction=prediction,
            date_mention=date_mention,
            category_translations=category_translations,
            sentiment=sentiment,
            auto_refresh=True
        )
    except BadRequest as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', page='error', error=str(e)), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise InternalServerError("เกิดข้อผิดพลาด")

@app.route('/get_prediction', methods=['POST'])
def get_prediction_route():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            raise BadRequest("กรุณาระบุคำถาม")
        message = data.get('message', '').strip()
        relationship_status = data.get('relationship_status', '').lower()
        is_explanation = data.get('is_explanation', False)
        card = data.get('card', '')
        category = data.get('category', '')
        use_gpt2 = data.get('use_gpt2', False)
        session_id = data.get('session_id', 'default')

        if not message or not isinstance(message, str):
            raise BadRequest("คำถามไม่ถูกต้อง")
        if relationship_status and relationship_status not in ["single", "in_relationship"]:
            relationship_status = None

        prediction, is_explanation, final_category, date_mention = get_prediction_or_response(
            card=card, category=category, message=message, relationship_status=relationship_status, is_explanation=is_explanation, use_gpt2=use_gpt2
        )
        sentiment, _ = analyze_sentiment(message)

        # บันทึกประวัติ
        prediction_history[session_id].append({
            'message': message,
            'prediction': prediction,
            'card': card or select_card(),
            'category': final_category,
            'timestamp': datetime.now().isoformat()
        })

        return jsonify({
            'prediction': prediction,
            'card': card or select_card(),
            'category': final_category,
            'is_explanation': is_explanation,
            'date_mention': date_mention,
            'sentiment': sentiment,
            'status': 'success'
        })
    except BadRequest as e:
        logger.error(f"Get prediction error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 400
    except Exception as e:
        logger.error(f"Get prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/rate_prediction', methods=['POST'])
def rate_prediction():
    try:
        data = request.get_json()
        if not all(k in data for k in ['chatId', 'messageId', 'rating']):
            raise BadRequest("ต้องมี chatId, messageId, rating")
        chat_id = data['chatId']
        message_id = data['messageId']
        rating = float(data['rating'])
        if not 0 <= rating <= 5:
            raise BadRequest("คะแนน 0-5 เท่านั้น")
        logger.info(f"คะแนน {rating} สำหรับ {chat_id}/{message_id}")
        with open('ratings.txt', 'a', encoding='utf-8') as f:
            f.write(f"{chat_id},{message_id},{rating},{datetime.now().isoformat()}\n")
        return jsonify({'status': 'success', 'message': 'บันทึกคะแนน'})
    except BadRequest as e:
        logger.warning(f"Rating error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Rating error: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/submit_review', methods=['POST'])
def submit_review():
    try:
        data = request.get_json()
        if not all(k in data for k in ['review', 'rating']):
            raise BadRequest("ต้องมี review, rating")
        review = data.get('review', '').strip()
        rating = float(data.get('rating', 0))
        if (not review and rating == 0) or not 0 <= rating <= 5:
            raise BadRequest("รีวิวหรือคะแนน 0-5 ต้องระบุ")
        logger.info(f"รีวิว: {review}, คะแนน: {rating}")
        with open('reviews.txt', 'a', encoding='utf-8') as f:
            f.write(f"{review},{rating},{datetime.now().isoformat()}\n")
        return jsonify({'status': 'success', 'message': 'บันทึกรีวิว'})
    except BadRequest as e:
        logger.warning(f"Review error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Review error: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/article')
def article():
    major_arcana = [
        "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
        "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
        "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
        "The Devil", "The Tower", "The Star", "The Moon", "The Sun",
        "Judgement", "The World"
    ]
    return render_template('index.html', page='article', major_arcana=major_arcana)

@app.route('/history', methods=['GET'])
def history():
    session_id = request.args.get('session_id', 'default')
    return jsonify({'history': prediction_history.get(session_id, [])})

# จัดการข้อผิดพลาด
@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({'error': str(e), 'status': 'error'}), 400

@app.errorhandler(InternalServerError)
def handle_internal_server_error(e):
    logger.critical(f"Internal Error: {str(e)}", exc_info=True)
    return jsonify({'error': 'เซิร์ฟเวอร์ล่ม กรุณาลองใหม่', 'status': 'error'}), 500

@app.errorhandler(Exception)
def handle_general_exception(e):
    logger.critical(f"Unexpected Error: {str(e)}", exc_info=True)
    return jsonify({'error': 'ข้อผิดพลาดไม่คาดคิด ติดต่อผู้ดูแล', 'status': 'error'}), 500

# ปิด Executor เมื่อแอปปิด
@atexit.register
def shutdown():
    executor.shutdown(wait=False)
    logger.info("Executor ปิดการทำงาน")

if __name__ == '__main__':
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        logger.critical(f"เซิร์ฟเวอร์เริ่มต้นล้มเหลว: {str(e)}", exc_info=True)
        sys.exit(1)