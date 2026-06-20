from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import json
import re
import time
from groq import Groq

app = Flask(__name__)
CORS(app)

# 1. Khởi tạo Groq API qua Environment Variable (Biến môi trường) để bảo mật
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# 2. Thay đổi đường dẫn CSV thành đường dẫn tương đối trong cùng thư mục project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "JLPT_N2_Grammar_v2.csv")
grammar_data = []

if os.path.exists(CSV_PATH):
    try:
        df = pd.read_csv(CSV_PATH)
        grammar_data = df.to_dict(orient='records')
        print(f"[RAG SYSTEM] Đã nạp thành công {len(grammar_data)} mục ngữ pháp N2 từ file CSV!")
    except Exception as e:
        print(f"[RAG ERROR] Không thể đọc file CSV: {str(e)}")
else:
    print(f"[RAG WARNING] Không tìm thấy file {CSV_PATH}. Vui lòng kiểm tra lại.")

def retrieve_n2_context(user_query):
    if not grammar_data: return ""
    matched_results = []
    query_clean = user_query.lower()
    for item in grammar_data:
        pattern_name = str(item.get('pattern', '')).lower()
        if pattern_name in query_clean or query_clean in pattern_name:
            context_str = (
                f"- Cấu trúc: {item.get('pattern', 'N/A')}\n"
                f"  Ý nghĩa: {item.get('meaning', 'N/A')}\n"
                f"  Cách dùng (Explanation): {item.get('explanation', 'N/A')}\n"
                f"  Ví dụ mẫu: {item.get('example_sentence', 'N/A')}\n"
                f"  Dịch ví dụ: {item.get('example_translation', 'N/A')}\n"
            )
            matched_results.append(context_str)
        if len(matched_results) >= 2: break
    if matched_results:
        return "\n[DỮ LIỆU SÁCH GIÁO KHOA CHUẨN ĐƯỢC TÌM THẤY]:\n" + "\n".join(matched_results)
    return ""

JSON_FORMAT_INSTRUCTION = """
BẮT BUỘC phản hồi theo cấu trúc JSON sạch sau đây, không kèm theo bất kỳ văn bản giải thích nào ở ngoài khối JSON:
{
  "reply": "Câu thoại tiếp theo của bạn bằng tiếng Nhật để tiếp tục cuộc trò chuyện",
  "errors": [
    {
      "original": "Cụm từ hoặc đoạn văn cụ thể mà người dùng viết sai ngữ pháp/từ vựng/kính ngữ nâng cao",
      "correction": "Cụm từ đúng sau khi bạn đã sửa lại chính xác",
      "explanation": "Giải thích chi tiết bằng tiếng Việt lý do tại sao sai và cách diễn đạt đúng chuẩn N2 công sở"
    }
  ]
}
"""

SYSTEM_PROMPTS = {
    "study": """Bạn là một chuyên gia đào tạo tiếng Nhật. Nhiệm vụ của bạn là tập trung hoàn toàn vào việc giải thích ý nghĩa từ vựng và cấu trúc ngữ pháp.
QUY TẮC PHẢN HỒI:
1. Nếu có dữ liệu trong [DỮ LIỆU SÁCH GIÁO KHOA CHUẨN], hãy ưu thiện dùng thông tin đó để giải thích.
2. Giải thích bằng tiếng Việt chi tiết: Ý nghĩa, Cách dùng.
3. Đưa ra các câu ví dụ thực tế bằng tiếng Nhật kèm theo Furigana/Kanji và bản dịch tiếng Việt tương ứng.
4. Tuyệt đối không trả lời ngắn gọn theo kiểu giao tiếp.""",

    "kaiwa": """You are a friendly Japanese conversation partner. Engage in a natural, casual, and authentic Japanese daily conversation (Kaiwa).
MỤC TIÊU: Giúp người dùng luyện phản xạ giao tiếp tự nhiên. Lỗi nhỏ trong văn nói hoàn toàn được chấp nhận.
QUY TẮC PHẢN HỒI:
1. Giao tiếp bằng tiếng Nhật đời thường, tự nhiên, gần gũi (Trình độ N3 - N4).
2. Trả lời trực tiếp vào nội dung người dùng nói, sau đó đặt một câu hỏi gợi mở ngắn gọn ở cuối câu để duy trì cuộc hội thoại. Không dịch hay giải thích tiếng Việt.
3. TUYỆT ĐỐI KHÔNG bắt lỗi ngữ pháp. Mảng 'errors' bắt buộc luôn luôn để trống là [].

BẮT BUỘC trả về định dạng JSON sau:
{
  "reply": "Câu thoại tiếng Nhật đời thường của bạn ở đây",
  "errors": []
}""",

    "n2": f"""Bạn là một đối tác giao tiếp tiếng Nhật trình độ cao (JLPT N2). Bạn đang trò chuyện hoặc phỏng vấn người dùng trong một bối cảnh trang trọng hoặc chuyên nghiệp (công sở, IT Comtor, phỏng vấn xin việc).
MỤC TIÊU: Chuẩn hóa học thuật, rèn luyện văn phong công sở chuyên nghiệp nâng cao.
QUY TẮC PHẢN HỒI:
1. Trong phần 'reply', sử dụng mẫu câu ngữ pháp nâng cao, cấu trúc đặc thù chuẩn N2 (đặc biệt là từ ghép Kanji - 熟語) hoặc kính ngữ trang trọng (Keigo).
2. Bạn có nhiệm vụ khắt khe là phân tích tỉ mỉ câu nói của người dùng. Nếu phát hiện lỗi sai về ngữ pháp N2, từ vựng hoặc sắc thái diễn đạt chưa chuẩn doanh nghiệp, hãy nhặt lỗi ra và điền vào mảng 'errors'.
3. Nếu câu nói của người dùng hoàn toàn chính xác và tự nhiên, mảng 'errors' sẽ để trống là [].
{JSON_FORMAT_INSTRUCTION}"""
}

# Thêm một endpoint GET để kiểm tra trạng thái sống của server (Health Check)
@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "healthy", "message": "N2 AI Tutor Backend is running!"})

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    try:
        data = request.json
        user_message = data.get("message", "")
        selected_mode = data.get("mode", "kaiwa").lower()

        if selected_mode not in SYSTEM_PROMPTS:
            selected_mode = "kaiwa"

        active_system_prompt = SYSTEM_PROMPTS[selected_mode]

        if selected_mode == "study":
            rag_context = retrieve_n2_context(user_message)
            if rag_context:
                active_system_prompt += "\n" + rag_context

        if selected_mode in ["kaiwa", "n2"]:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": active_system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
        else:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": active_system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.5,
                max_tokens=1024
            )

        response_text = chat_completion.choices[0].message.content.strip()
        final_reply = response_text
        detected_errors = []

        if selected_mode in ["kaiwa", "n2"]:
            try:
                json_data = json.loads(response_text)
                final_reply = json_data.get("reply", "")
                detected_errors = json_data.get("errors", [])
            except Exception as json_err:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group(0))
                        final_reply = json_data.get("reply", "")
                        detected_errors = json_data.get("errors", [])
                    except:
                        final_reply = response_text
                else:
                    final_reply = response_text

        elapsed_time = time.time() - start_time
        print(f"[{selected_mode.upper()} MODE] Processed in {elapsed_time:.3f}s")

        return jsonify({
            "reply": final_reply,
            "mode": selected_mode,
            "errors": detected_errors,
            "has_context": True if (selected_mode == "study" and rag_context) else False
        })
    except Exception as e:
        print(f"[CRITICAL SERVER ERROR]: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Khởi chạy bằng Gunicorn / Uvicorn ở production, đoạn này chỉ chạy khi test local
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)