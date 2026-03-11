import os
import base64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from dotenv import load_dotenv
import datetime
import time
import cv2
import numpy as np
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import threading
import requests
import ast
from yolo_vehicle import YOLO, SimpleTracker, AccidentDetector, PERSON_CLASS_ID, VEHICLE_CLASS_IDS
import threading
import re

# ---  ---
PREV_CONGESTION_STATE = "low"
LAST_SAVE_TIME = 0
SAVE_COOLDOWN = 2  # seconds (prevents double save spam)

def save_last_frame_if_needed(frame, current_congestion):
    global PREV_CONGESTION_STATE, LAST_SAVE_TIME

    now = time.time()

    # Save when LOW → HIGH
    if PREV_CONGESTION_STATE != "high" and current_congestion == "high":
        if now - LAST_SAVE_TIME > SAVE_COOLDOWN:
            cv2.imwrite("last_frame.jpg", frame)
            print("✅ Saved last_frame.jpg (LOW → HIGH)")
            send( image_path="last_frame.jpg")
            LAST_SAVE_TIME = now

    # Save when HIGH → LOW
    elif PREV_CONGESTION_STATE == "high" and current_congestion == "low":
        if now - LAST_SAVE_TIME > SAVE_COOLDOWN:
            cv2.imwrite("last_frame.jpg", frame)
            print("✅ Saved last_frame.jpg (HIGH → LOW)")
            send( image_path="last_frame.jpg")
            LAST_SAVE_TIME = now

    PREV_CONGESTION_STATE = current_congestion

URL = "https://roadxpert-public-page-v1.onrender.com/update_ui"

def send(text=None, image_path=None):
    payload = {}

    if text:
        payload["text"] = text

    if image_path:
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            payload["image"] = "data:image/jpeg;base64," + b64
        else:
            print("❌ Image not found:", image_path)
            return

    r = requests.post(URL, json=payload)
    print("Server:", r.status_code, r.text)

# --- YOLO Vehicle Detection Setup ---
yolo_model = YOLO("yolo11n.pt")   # lightweight YOLO model
tracker = SimpleTracker(iou_threshold=0.3, max_missed=8)
accident_detector = AccidentDetector()

def safe_load_json(text):
    """
    Try multiple strategies to safely parse a string into JSON/dict.
    Returns dict on success, otherwise None.
    """
    if not isinstance(text, str):
        return None

    txt = text.strip()

    # direct attempt
    try:
        return json.loads(txt)
    except Exception:
        pass

    # try to extract the first balanced {...}
    start = txt.find('{')
    if start != -1:
        stack = []
        for i in range(start, len(txt)):
            ch = txt[i]
            if ch == '{':
                stack.append(i)
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        candidate = txt[start:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break

    # fallback: literal_eval (handles single quotes etc.)
    try:
        obj = ast.literal_eval(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return None

load_dotenv()

pos_line = 300   # adjust depending on resized frame height
offset = 6
total_count = 0
# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=True)

API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

# Full path to your adb executable
#adb_path = "RoadXpert-Server-1/platform-tools-latest-linux/platform-tools/adb" #for linux
adb_path = "Roadxpert\ADB_CALL" #for windows

number = "101"

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def extract_ai_content(response: dict):
    """
    Extract only the AI's descriptive content from the response dict.
    """
    if isinstance(response, dict) and "content" in response:
        return response["content"]
    return ""

def emergency_detect(text):

    # Convert non-string to string (dict, list, etc.)
    if not isinstance(text, str):
        text = str(text)

    cleaned = text.lower()

    pattern = r"e.*?m.*?e.*?r.*?g.*?e.*?n.*?c.*?y[\s\S]{0,200}t.*?r.*?u.*?e"

    # Perform search
    if re.search(pattern, cleaned):
        return True

    return False


def json_to_text(content):
    """
    Convert JSON or dict into human-readable text using AI.
    """
    command = f"""
You are RoadXpert AI. Convert the following information into a concise,
professional, human-readable summary paragraph.
Do not output JSON, only text. Avoid emojis, URLs, or unnecessary formatting.

INPUT:
{content}

TASK:
Provide a detailed paragraph summary with no special symbols or characters.
"""
    response = query({
    "messages": [
        {
            "role": "user",
            "content": command
        }
    ],
    "model": "meta-llama/Llama-3.1-8B-Instruct"
})

    #print(extract_ai_content(response["choices"][0]["message"]))
    return extract_ai_content(response["choices"][0]["message"])

def generate_daily_report():
    today = datetime.date.today().strftime("%Y-%m-%d")

    # --- Vehicle Summary ---
    total_vehicles = 0
    avg_speeds = []

    try:
        with open("vehicle_log.json", "r") as f:
            for line in f:
                entry = json.loads(line)
                if entry["timestamp"].startswith(today):
                    total_vehicles += entry.get("vehicles", 0)
                    for obj in entry.get("objects", []):
                        if "speed_kmph" in obj:
                            avg_speeds.append(obj["speed_kmph"])
    except FileNotFoundError:
        pass

    mean_speed = sum(avg_speeds) / len(avg_speeds) if avg_speeds else 0

    # --- AI Analysis Summary ---
    damage_reports = []
    accidents = 0
    recommendations = []

    try:
        with open("analysis_log.json", "r") as f:
            for line in f:
                entry = json.loads(line)
                if entry["timestamp"].startswith(today):
                    analysis = entry.get("analysis", {})
                    damage_reports.append(analysis.get("road_damage", {}))
                    if analysis.get("accident", {}).get("detected"):
                        accidents += 1
                    rec = analysis.get("recommendations", {}).get("safety_alert")
                    if rec:
                        recommendations.append(rec)
    except FileNotFoundError:
        pass

    # --- Weather Summary ---
    weather_reports = []
    try:
        with open("analysis_log.json", "r") as f:
            for line in f:
                entry = json.loads(line)
                if entry["timestamp"].startswith(today):
                    analysis = entry.get("analysis", {})
                    weather_reports.append(analysis.get("weather", {}).get("raw_data", ""))
    except FileNotFoundError:
        pass

    # --- Build Report Text ---
    report = f"""
🚦 RoadXpert Daily Report – {today}

📊 Traffic Summary:
- Total vehicles: {total_vehicles}
- Average speed: {mean_speed:.2f} km/h

🌦️ Weather Conditions:
- {len(weather_reports)} observations recorded
- Summary: {", ".join(weather_reports[-3:]) if weather_reports else "No data"}

🛠️ Road Condition Summary:
- Total analyses: {len(damage_reports)}
- Accidents detected: {accidents}
- Recommendations:
  {"; ".join(recommendations) if recommendations else "No urgent alerts"}

✅ End of report
"""
    print(report)

    # --- Send Email ---
    send_emergency_email("Daily Road Report", report, image_path=None)  # reuse Gmail function

# --- Schedule job at 10 PM every day ---
schedule.every().day.at("22:00").do(generate_daily_report)
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)  # check every minute

def get_weather_forecast():
    # API parameters
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 23.341468,
        "longitude": 86.372060,
        "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum,precipitation_probability_mean",
        "timezone": "auto",
        "forecast_days": 7
    }

    # Fetch the data
    response = requests.get(url, params=params)
    output = []
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract daily forecast
        if "daily" in data:
            daily = data["daily"]
            output.append("7-Day Weather Forecast:")
            output.append("=" * 40)
            for i in range(len(daily["time"])):
                date = daily["time"][i]
                min_temp = daily["temperature_2m_min"][i]
                max_temp = daily["temperature_2m_max"][i]
                precip_sum = daily["precipitation_sum"][i]
                precip_prob = daily["precipitation_probability_mean"][i]
                
                output.append(f"Date: {date}")
                output.append(f"Temperature Range: {min_temp}°C to {max_temp}°C")
                output.append(f"Expected Rainfall: {precip_sum} mm")
                output.append(f"Rain Probability: {precip_prob}%")
                output.append("-" * 20)
            
            # Trend analysis
            temp_trend = "warming" if max_temp > daily["temperature_2m_max"][0] else "cooling/stable"
            precip_trend = "increasing rain risk" if precip_prob > daily["precipitation_probability_mean"][0] else "decreasing/stable"
            output.append(f"\nOverall Trends: {temp_trend} trend; {precip_trend}.")
        else:
            output.append("Error: No daily data available.")
    else:
        output.append(f"Error fetching data: {response.status_code}")
    
    return "\n".join(output)

def auto_analyze_continuous():
    print("🚦 Continuous Auto-Analyzer started...")
    last_timestamp = None

    while True:
        try:
            # Read the latest vehicle entry
            with open("vehicle_log.json", "r") as f:
                lines = f.readlines()
                if not lines:
                    time.sleep(5)
                    continue

                last_entry = json.loads(lines[-1])
                timestamp = last_entry.get("timestamp")

                # Avoid re-analyzing the same entry
                if timestamp == last_timestamp:
                    time.sleep(5)
                    continue
                last_timestamp = timestamp

                # --- inside auto_analyze_continuous() before analysis ---
                vehicle_count = last_entry.get("vehicles", 0)
                accidents = last_entry.get("accidents", [])
                speeds = [obj.get("speed_kmph", 0) for obj in last_entry.get("objects", [])]
                avg_speed_kmph = sum(speeds) / len(speeds) if speeds else 0
                avg_speed_kmps = avg_speed_kmph / 3600.0

                if not accidents and vehicle_count <= 0:
                   print("⏭️ Skipping (no YOLO alert)")
                   time.sleep(5)
                   continue

                 # --- Get Weather Data ---
                weather_data = get_weather_forecast()

                # --- Try to attach last frame ---
                image_data = None
                if os.path.exists("last_frame.jpg"):
                    with open("last_frame.jpg", "rb") as f:
                        img_bytes = f.read()
                        b64 = base64.b64encode(img_bytes).decode("utf-8")
                        image_data = f"data:image/jpeg;base64,{b64}"

                # --- Build Prompt ---
                user_prompt = f"""
You are RoadXpert AI, a professional-grade AI system specialized in:
- Pavement engineering
- Traffic flow analysis
- Accident detection
- Infrastructure risk forecasting

You operate in a critical safety environment. Accuracy and caution are mandatory.
The lifespan of a newly built road or good road is 20 to 40 years depending heavily on material (asphalt vs. concrete), traffic, climate, and upkeep.

You MUST:
- Always generate VALID, STRICT JSON.
- Never include explanations, markdown, comments, or natural language outside JSON.
- Never hallucinate data.
- If a value cannot be reliably determined from the image or inputs, mark it as "unknown".
- Use the provided traffic logs and weather data as the PRIMARY truth.
- Use the image strictly for visual confirmation (cracks, potholes, congestion, accidents).

INPUT DATA:
- Weather: {weather_data}
- Latest vehicle log: vehicles, speed, congestion
- Road image: Provided below (visual confirmation only)

--------------------------------------
CORE ENGINEERING MODEL (MANDATORY)
--------------------------------------

Use the following deterioration model ONLY if ALL required parameters are available:

damage_rate =
(vehicle_count * load_factor)
+ (rainfall_mm * 0.2)
+ (potholes * 50)
+ (cracks * 30)

Where:
- load_factor = 1.0 if avg_speed_kmph < 40  
- load_factor = 0.7 if avg_speed_kmph ≥ 40  

base_life = 20000 (durability units)

failure_days = round(base_life / damage_rate)

If ANY value is missing → set "predicted_damage_in_days" = null

--------------------------------------
RQI (Road Quality Index)
--------------------------------------

RQI = (pothole_density_score + crack_score + traffic_stress_score + weather_stress_score) / 4

Each sub-score must be between 0 to 100.

--------------------------------------
STRICT OUTPUT FORMAT (JSON ONLY)
--------------------------------------
{{
  "road_damage": {{
    "potholes_detected": true/false,
    "cracks_detected": true/false,
    "damage_severity": "low/medium/high",
    "predicted_damage_in_days": number(predict very accurately by analyzing road condition, road material, weather conditions)
    "RQI_Index": Custom RQI Index number=(pothole density + rut depth + traffic stress + weather score)/4 out of __
  }},
  "traffic": {{
    "vehicle_count": vehicle_count from image,
    "average_speed_kmph": avg_speed_kmph:.2f predict from image,
    "congestion_level": "low/medium/high"
  }},
  "weather": {{
    "raw_data": "{weather_data}"
  }},
  "accident": {{
    "detected": true/false,
    "details": "short description if detected"
  }},
  "recommendations": {{
    "needs_repair": true/false,
    "repair_priority": "low/medium/high",
    "speed_bump_suggestion": "install/remove/none",
    "safety_alert": "short urgent message for authorities"
  }},
  "emergency": true/false
}}

--------------------------------------
EMERGENCY LOGIC (MANDATORY)
--------------------------------------

Set "emergency" = true ONLY if at least ONE of the following is true:
- Accident detected
- Damage severity = "high" AND congestion = "high"
- Heavy rainfall AND cracks/potholes detected
- Predicted damage days < 7

--------------------------------------
FINAL RULES
--------------------------------------

- Never invent potholes, cracks, accidents, or speeds.
- If not clearly visible → mark as "unknown" or false.
- Be conservative in emergency decisions.
- Output ONLY the final JSON.

"""
                # --- Hugging Face API Call ---
                content_block = [{"type": "text", "text": user_prompt}]
                if image_data:
                    content_block.append({"type": "image_url", "image_url": {"url": image_data}})

                payload = {
                    "messages": [{"role": "user", "content": content_block}],
                    "model": "baidu/ERNIE-4.5-VL-424B-A47B-Base-PT"
                }

                response = query(payload)
                output = response["choices"][0]["message"]
                result_text = extract_ai_content(output)

                # --- Save log ---
                log_entry = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "analysis": json.loads(result_text) if result_text.strip().startswith("{") else result_text
                }
                with open("analysis_log.json", "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                print(" Auto-analysis logged.")

                # --- Emergency Email ---
                try:
                    result_json = json.loads(result_text)
                    if result_json.get("emergency", False):
                        frame_path = "last_frame.jpg" if os.path.exists("last_frame.jpg") else None
                        send_emergency_email(
                            "Local Road",
                            result_json.get("recommendations", {}).get("safety_alert", "Urgent road hazard"),
                            image_path=frame_path
                        )
                except Exception as e:
                    print(" Emergency parsing failed:", e)

        except Exception as e:
            print(" Auto-analyzer error:", e)

        time.sleep(30)  # check every 30 sec #you can chage this time as you want

def send_emergency_email(location, details, image_path=None):
    sender = "yyyyy@gmail.com"
    password = os.getenv("GMAIL_APP_PASSWORD")
    receiver = "xxxxxxxx@gmail.com"

    subject = f"🚨 Road Emergency Alert at {location}"
    body = f"""
    Emergency detected by RoadXpert system.
    Location: {location}
    Details: {details}

    Immediate action required.
    """

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # --- Attach image if provided ---
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                img_data = f.read()
            from email.mime.image import MIMEImage
            img = MIMEImage(img_data, name=os.path.basename(image_path))
            msg.attach(img)
            print(f"🖼️ Attached image: {image_path}")
        except Exception as e:
            print(f"⚠️ Failed to attach image: {e}")

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        print("✅ Emergency email sent to authority.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
    finally:
        # --- Delete image after sending ---
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"🗑️ Deleted temporary image: {image_path}")
            except Exception as e:
                print(f"⚠️ Could not delete image {image_path}: {e}")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
    
@app.route("/traffic", methods=["GET"])
def traffic():
    try:
        vehicle_count = 0
        last_entry = {}

        # Read last vehicle log entry
        with open("vehicle_log.json", "r") as f:
            lines = f.readlines()
            if lines:
                last_entry = json.loads(lines[-1])
                vehicle_count = last_entry.get("vehicles", 0)

        return jsonify({
            "vehicles": vehicle_count,
            "last_entry": last_entry
        })
    except FileNotFoundError:
        return jsonify({"error": "No traffic data yet"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        image_data = data.get("image")   # frontend sends data:image/jpeg;base64,...
        user_location = data.get("location", "Purulia, West Bengal")

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # --- Get Weather Data ---
        weather_data = get_weather_forecast()

        # --- Get Latest Vehicle Data ---
        vehicle_count, avg_speed_kmph, avg_speed_kmps = 0, 0, 0
        try:
            with open("vehicle_log.json", "r") as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    vehicle_count = last_entry.get("vehicles", 0)
                    speeds = [obj.get("speed_kmph", 0) for obj in last_entry.get("objects", [])]
                    if speeds:
                        avg_speed_kmph = sum(speeds) / len(speeds)
                        avg_speed_kmps = avg_speed_kmph / 3600.0
        except:
            pass

        # --- Save frame locally for emergency use ---
        try:
            # remove prefix, decode, save raw image
            if "," in image_data:
                raw_b64 = image_data.split(",")[1]
            else:
                raw_b64 = image_data
            image_bytes = base64.b64decode(raw_b64)
            with open("last_frame.jpg", "wb") as f:
                f.write(image_bytes)
        except Exception as e:
            print("⚠️ Could not save last_frame.jpg:", e)

        # --- Build Prompt --- - Average vehicle speed: {avg_speed_kmph:.2f} km/h ({avg_speed_kmps:.6f} km/s)
        user_prompt = f"""
You are RoadXpert AI, a professional-grade AI system specialized in:
- Pavement engineering
- Traffic flow analysis
- Accident detection
- Infrastructure risk forecasting

You operate in a critical safety environment. Accuracy and caution are mandatory.
The lifespan of a newly built road or good road is 20 to 40 years depending heavily on material (asphalt vs. concrete), traffic, climate, and upkeep.

You MUST:
- Always generate VALID, STRICT JSON.
- Never include explanations, markdown, comments, or natural language outside JSON.
- Never hallucinate data.
- If a value cannot be reliably determined from the image or inputs, mark it as "unknown".
- Use the provided traffic logs and weather data as the PRIMARY truth.
- Use the image strictly for visual confirmation (cracks, potholes, congestion, accidents).

INPUT DATA:
- Weather: {weather_data}
- Latest vehicle log: vehicles, speed, congestion
- Road image: Provided below (visual confirmation only)

--------------------------------------
CORE ENGINEERING MODEL (MANDATORY)
--------------------------------------

Use the following deterioration model ONLY if ALL required parameters are available:

damage_rate =
(vehicle_count * load_factor)
+ (rainfall_mm * 0.2)
+ (potholes * 50)
+ (cracks * 30)

Where:
- load_factor = 1.0 if avg_speed_kmph < 40  
- load_factor = 0.7 if avg_speed_kmph ≥ 40  

base_life = 20000 (durability units)

failure_days = round(base_life / damage_rate)

If ANY value is missing → set "predicted_damage_in_days" = null

--------------------------------------
RQI (Road Quality Index)
--------------------------------------

RQI = (pothole_density_score + crack_score + traffic_stress_score + weather_stress_score) / 4

Each sub-score must be between 0 to 100.

--------------------------------------
STRICT OUTPUT FORMAT (JSON ONLY)
--------------------------------------
{{
  "road_damage": {{
    "potholes_detected": true/false,
    "cracks_detected": true/false,
    "damage_severity": "low/medium/high",
    "predicted_damage_in_days": number(predict very accurately by analyzing road condition, road material, weather conditions)
    "RQI_Index": Custom RQI Index number=(pothole density + rut depth + traffic stress + weather score)/4 out of __
  }},
  "traffic": {{
    "vehicle_count": vehicle_count from image,
    "average_speed_kmph": avg_speed_kmph:.2f predict from image,
    "congestion_level": "low/medium/high"
  }},
  "weather": {{
    "raw_data": "{weather_data}"
  }},
  "accident": {{
    "detected": true/false,
    "details": "short description if detected"
  }},
  "recommendations": {{
    "needs_repair": true/false,
    "repair_priority": "low/medium/high",
    "speed_bump_suggestion": "install/remove/none",
    "safety_alert": "short urgent message for authorities"
  }},
  "emergency": true/false
}}

--------------------------------------
EMERGENCY LOGIC (MANDATORY)
--------------------------------------

Set "emergency" = true ONLY if at least ONE of the following is true:
- Accident detected
- Damage severity = "high" AND congestion = "high"
- Heavy rainfall AND cracks/potholes detected
- Predicted damage days < 7

--------------------------------------
FINAL RULES
--------------------------------------

- Never invent potholes, cracks, accidents, or speeds.
- If not clearly visible → mark as "unknown" or false.
- Be conservative in emergency decisions.
- Output ONLY the final JSON.
"""

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_data}}  # use full Data URI
                    ]
                }
            ],
            "model": "zai-org/GLM-4.5V:novita"
        }

        response = query(payload)
        output = response["choices"][0]["message"]
        result_text = extract_ai_content(output)

        # --- Parse & Save Logs ---
        try:
            # --- Flexible Parsing ---
            parsed_result = safe_load_json(result_text) if isinstance(result_text, str) else None

            if isinstance(parsed_result, dict):
              human_text = json_to_text(parsed_result)
              analysis_to_store = parsed_result
            else:
               human_text = json_to_text(result_text)
               analysis_to_store = result_text

            log_entry = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "location": user_location,
                "analysis": analysis_to_store,
                "human_readable": human_text
            }

            with open("analysis_log.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            with open("analysis_log.txt", "a", encoding="utf-8") as f:
                f.write("\n" + "="*40 + "\n")
                f.write(f"📅 {log_entry['timestamp']}\n")
                f.write(f"📍 Location: {user_location}\n\n")
                f.write(human_text + "\n")

        except Exception as e:
            print("⚠️ Could not save analysis log:", e)

        # --- Emergency Email ---
        try:
            result_json = parsed_result if isinstance(parsed_result, dict) else {}
            """
            send_emergency_email(
                    user_location,
                    human_text,
                    image_path="last_frame.jpg" if os.path.exists("last_frame.jpg") else None
                )"""
            text_to_send = human_text
            print("Sending data to website automatically...")
            send(text=text_to_send, image_path="last_frame.jpg")
            if emergency_detect(result_json):
                send_emergency_email(
                    user_location,
                    result_json.get("recommendations", {}).get("safety_alert", "Urgent road hazard"),
                    image_path="last_frame.jpg" if os.path.exists("last_frame.jpg") else None
                )
                text_to_send = human_text
                print("Sending data to website automatically...")
                send(text=text_to_send, image_path="last_frame.jpg")
                os.system(f'"{adb_path}" shell am start -a android.intent.action.CALL -d tel:+91{number}')

        except Exception as e:
            print("⚠️ Emergency email step failed:", e)

        return jsonify({"result": human_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        b64 = data.get("image")
        if not b64:
            return jsonify({"error": "No image"}), 400

        if "," in b64:
            b64 = b64.split(",")[1]
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # --- Run YOLO ---
        res = yolo_model.predict(frame, imgsz=416, conf=0.3, iou=0.45, verbose=False)[0]

        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for b, c, cl in zip(xyxy, confs, clss):
                if cl not in VEHICLE_CLASS_IDS and cl != PERSON_CLASS_ID:
                    continue
                x1,y1,x2,y2 = map(float, b)
                detections.append({'bbox':[x1,y1,x2,y2],'cls':int(cl),'conf':float(c)})

        # --- Track + Accidents ---
        tracks = tracker.update(detections)
        events = accident_detector.detect_collisions(tracks, time.time())

        vehicle_count = sum(1 for tr in tracks if tr.cls in VEHICLE_CLASS_IDS)
        congestion = (
            "low" if vehicle_count < 3 else
            "medium" if vehicle_count < 7 else
            "high"
        )

        result = {
            "objects_detected": len(tracks),
            "vehicles": vehicle_count,
            "congestion_level": congestion,
            "accidents": events
        }
        save_last_frame_if_needed(frame, congestion)
        # --- Log if any vehicles ---
        if vehicle_count > 0:
            log_entry = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "vehicles": vehicle_count,
                "objects": [{"id": tr.id, "cls": tr.cls, "speed_kmph": round(tr.median_speed() * 0.1, 2)} for tr in tracks],
                "congestion_level": congestion,
                "accidents": events
            }
            with open("vehicle_log.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        # --- Draw boxes & alerts on the same frame ---
        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr.bbox)
            color = (0, 255, 0) if tr.cls in VEHICLE_CLASS_IDS else (0, 0, 255)
            label = f"ID {tr.id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for e in events:
            cv2.putText(frame, "ACCIDENT!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # --- Convert frame with boxes to base64 for frontend ---
        # --- FPS calculation ---
        global _last_fps_time, _smoothed_fps
        t_now = time.time()
            # calculate time difference
        if '_last_fps_time' not in globals():
            _last_fps_time = t_now
            _smoothed_fps = 0
        dt = max(1e-6, t_now - _last_fps_time)
        _last_fps_time = t_now
        fps = 1.0 / dt
           # smooth FPS to avoid flicker
        _smoothed_fps = 0.8 * _smoothed_fps + 0.2 * fps
        overlay_text = f"FPS: {_smoothed_fps:.2f} | Congestion: {congestion.upper()} | Vehicles: {vehicle_count}"
        cv2.putText(frame, overlay_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if events:
            cv2.putText(frame, "⚠️ ACCIDENT DETECTED", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # --- Convert frame with overlays to base64 for frontend ---
        _, buffer = cv2.imencode(".jpg", frame)
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        result["image_with_boxes"] = "data:image/jpeg;base64," + frame_b64
        result["overlay_text"] = overlay_text
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Live video streaming setup ---
camera_source = 0  # use 0 for webcam; or  Use path for video files (files should be in the same directory as the script)
cap = cv2.VideoCapture(camera_source)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart if video ended
            continue

        # ---- your YOLO detection logic here ----
        res = yolo_model.predict(frame, imgsz=416, conf=0.3, iou=0.45, verbose=False)[0]

        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for b, c, cl in zip(xyxy, confs, clss):
                if cl not in VEHICLE_CLASS_IDS and cl != PERSON_CLASS_ID:
                    continue
                x1,y1,x2,y2 = map(float, b)
                detections.append({'bbox':[x1,y1,x2,y2],'cls':int(cl),'conf':float(c)})

        # --- Track + Accidents ---
        tracks = tracker.update(detections)
        events = accident_detector.detect_collisions(tracks, time.time())

        vehicle_count = sum(1 for tr in tracks if tr.cls in VEHICLE_CLASS_IDS)
        congestion = (
            "low" if vehicle_count < 3 else
            "medium" if vehicle_count < 7 else
            "high"
        )
        save_last_frame_if_needed(frame, congestion)
        result = {
            "objects_detected": len(tracks),
            "vehicles": vehicle_count,
            "congestion_level": congestion,
            "accidents": events
        }

        # --- Log if any vehicles ---
        if vehicle_count > 0:
            log_entry = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "vehicles": vehicle_count,
                "objects": [{"id": tr.id, "cls": tr.cls, "speed_kmph": round(tr.median_speed() * 0.1, 2)} for tr in tracks],
                "congestion_level": congestion,
                "accidents": events
            }
            with open("vehicle_log.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        # --- Draw boxes & alerts on the same frame ---
        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr.bbox)
            color = (0, 255, 0) if tr.cls in VEHICLE_CLASS_IDS else (0, 0, 255)
            label = f"ID {tr.id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for e in events:
            cv2.putText(frame, "ACCIDENT!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # --- Convert frame with boxes to base64 for frontend ---
        t1 = time.time()
        fps = 1.0 / max(1e-6, (t1 - time.time()))
        overlay_text = f"Congestion: {congestion.upper()} | Vehicles: {vehicle_count}"
        cv2.putText(frame, overlay_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if events:
            cv2.putText(frame, "⚠️ ACCIDENT DETECTED", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Encode frame as JPEG and yield as stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/capture_frame")
def capture_frame():
    """Capture a clean frame (without boxes/text) from the live video feed."""
    success, frame = cap.read()
    if not success:
        return jsonify({"error": "No frame captured"}), 500

    # Save clean frame for manual analysis
    _, buffer = cv2.imencode(".jpg", frame)
    frame_b64 = base64.b64encode(buffer).decode("utf-8")
    image_data = "data:image/jpeg;base64," + frame_b64
    return jsonify({"image": image_data})
  

@app.route('/video_feed')
def video_feed():
    """Continuous live feed for either camera or video."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/server_time")
def server_time():
    now = datetime.datetime.now().strftime("%H:%M")
    return jsonify({"time": now})


if __name__ == "__main__":
     # Start daily report scheduler (10 PM summary)
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    # Start continuous auto-analyzer (real-time traffic risk analysis)
    #analyzer_thread = threading.Thread(target=auto_analyze_continuous, daemon=True)
    #analyzer_thread.start()

    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=5000)
    #app.run(host="0.0.0.0", port=5000, ssl_context=("cert.pem", "key.pem"))

