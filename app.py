from flask import Flask, render_template, request, jsonify, make_response, send_file, session
import pickle
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import io
import base64
import random
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, HRFlowable, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.legends import Legend
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')
MODEL_PATH = os.getenv('MODEL_PATH', 'model_rf.pkl')
DATA_PATH = os.getenv('DATA_PATH', 'processed_data.csv')
HISTORY_FILE = 'prediction_history.json'
model_svc = pickle.load(open(MODEL_PATH, 'rb'))  # Model Loading
df = pd.read_csv(DATA_PATH)
import json
def load_history():
    """Load prediction history from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []
def save_history(history):
    """Save prediction history to file"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
def add_to_history(prediction_data):
    """Add new prediction to history"""
    history = load_history()
    prediction_data['id'] = len(history) + 1
    prediction_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append(prediction_data)
    if len(history) > 50:
        history = history[-50:]
    save_history(history)
    return prediction_data['id']
EXERCISE_TYPES = {
    0: 'Running',
    1: 'Cycling', 
    2: 'Swimming',
    3: 'Yoga',
    4: 'Weightlifting',
    5: 'HIIT'
}
def get_ai_suggestions(calories, bmi, bmi_status, steps, workout_mins, heart_rate, age, exercise_type):
    """Generate AI-powered personalized suggestions"""
    suggestions = []
    warnings = []
    achievements = []
    if bmi < 18.5:
        suggestions.append({
            'icon': 'utensils',
            'title': 'Increase Calorie Intake',
            'desc': 'Consider adding 300-500 extra calories daily with protein-rich foods like eggs, nuts, and lean meats.',
            'priority': 'high'
        })
        suggestions.append({
            'icon': 'dumbbell',
            'title': 'Focus on Strength Training',
            'desc': 'Add weightlifting 3x/week to build muscle mass and healthy weight.',
            'priority': 'medium'
        })
    elif bmi >= 25 and bmi < 30:
        suggestions.append({
            'icon': 'fire-alt',
            'title': 'Increase Cardio Activity',
            'desc': 'Add 30 mins of cardio 4-5x/week. Try brisk walking, cycling, or swimming.',
            'priority': 'high'
        })
        suggestions.append({
            'icon': 'apple-alt',
            'title': 'Reduce Processed Foods',
            'desc': 'Replace processed snacks with fruits, vegetables, and whole grains.',
            'priority': 'medium'
        })
    elif bmi >= 30:
        warnings.append({
            'icon': 'exclamation-triangle',
            'title': 'Health Alert',
            'desc': 'Your BMI indicates obesity. Consider consulting a healthcare professional for a personalized plan.',
            'priority': 'critical'
        })
        suggestions.append({
            'icon': 'walking',
            'title': 'Start with Low-Impact Exercise',
            'desc': 'Begin with 20-30 mins walking daily, gradually increasing intensity.',
            'priority': 'high'
        })
    if steps < 5000:
        suggestions.append({
            'icon': 'shoe-prints',
            'title': 'Increase Daily Steps',
            'desc': f'You took {steps} steps. Aim for 8,000-10,000 steps daily. Try taking stairs instead of elevator.',
            'priority': 'medium'
        })
    elif steps >= 10000:
        achievements.append({
            'icon': 'medal',
            'title': 'Step Champion! 🏆',
            'desc': f'Amazing! You hit {steps} steps today. Keep maintaining this active lifestyle!',
            'type': 'gold'
        })
    elif steps >= 7500:
        achievements.append({
            'icon': 'star',
            'title': 'Great Progress! ⭐',
            'desc': f'{steps} steps is excellent! Just a bit more to reach the 10K goal.',
            'type': 'silver'
        })
    if workout_mins < 30:
        suggestions.append({
            'icon': 'clock',
            'title': 'Extend Workout Duration',
            'desc': f'{workout_mins} mins is a good start. Try to reach 45-60 mins for optimal results.',
            'priority': 'medium'
        })
    elif workout_mins >= 60:
        achievements.append({
            'icon': 'fire',
            'title': 'Workout Warrior! 💪',
            'desc': f'{workout_mins} minutes of workout! Outstanding dedication to fitness.',
            'type': 'gold'
        })
    max_hr = 220 - age
    if heart_rate > max_hr * 0.9:
        warnings.append({
            'icon': 'heartbeat',
            'title': 'High Heart Rate Alert',
            'desc': f'Your heart rate ({heart_rate} BPM) is very high. Consider reducing intensity and consult a doctor if this persists.',
            'priority': 'high'
        })
    elif heart_rate < max_hr * 0.5 and workout_mins > 20:
        suggestions.append({
            'icon': 'heart',
            'title': 'Increase Workout Intensity',
            'desc': f'Your heart rate ({heart_rate} BPM) is low for your workout. Try increasing intensity for better results.',
            'priority': 'low'
        })
    if calories >= 3000:
        achievements.append({
            'icon': 'trophy',
            'title': 'Elite Burner! 🔥',
            'desc': f'Incredible! {calories} calories burned puts you in the elite category.',
            'type': 'platinum'
        })
    elif calories >= 2500:
        achievements.append({
            'icon': 'award',
            'title': 'Super Active! 🌟',
            'desc': f'{calories} calories burned! You\'re performing at an athlete level.',
            'type': 'gold'
        })
    exercise_tips = {
        0: {'tip': 'For running, try interval training - alternate between sprinting and jogging for maximum calorie burn.', 'icon': 'running'},
        1: {'tip': 'Cycling tip: Maintain 80-100 RPM cadence for optimal fat burning and endurance.', 'icon': 'biking'},
        2: {'tip': 'Swimming works all muscle groups! Try different strokes to target various muscles.', 'icon': 'swimmer'},
        3: {'tip': 'Yoga improves flexibility and mental health. Try holding poses for 30-60 seconds for strength.', 'icon': 'spa'},
        4: {'tip': 'For weightlifting, focus on compound movements like squats and deadlifts for maximum calorie burn.', 'icon': 'dumbbell'},
        5: {'tip': 'HIIT is highly effective! Ensure proper rest between sessions (48 hours) for recovery.', 'icon': 'bolt'}
    }
    if exercise_type in exercise_tips:
        suggestions.append({
            'icon': exercise_tips[exercise_type]['icon'],
            'title': f'{EXERCISE_TYPES[exercise_type]} Pro Tip',
            'desc': exercise_tips[exercise_type]['tip'],
            'priority': 'info'
        })
    water_needed = round((calories / 1000) * 1.5, 1)
    suggestions.append({
        'icon': 'tint',
        'title': 'Stay Hydrated',
        'desc': f'Based on your activity, drink at least {water_needed}L of water today to stay properly hydrated.',
        'priority': 'info'
    })
    if workout_mins >= 45 or calories >= 2000:
        suggestions.append({
            'icon': 'bed',
            'title': 'Recovery is Key',
            'desc': 'After intense workouts, ensure 7-8 hours of sleep for muscle recovery and growth.',
            'priority': 'info'
        })
    return {
        'suggestions': suggestions,
        'warnings': warnings,
        'achievements': achievements,
        'water_needed': water_needed
    }
def get_weekly_plan(exercise_type, fitness_level, bmi_status):
    """Generate a personalized weekly workout plan"""
    if bmi_status == "Obese":
        plan = {
            'monday': {'activity': 'Walking', 'duration': '30 mins', 'intensity': 'Low'},
            'tuesday': {'activity': 'Rest / Light Stretching', 'duration': '15 mins', 'intensity': 'Very Low'},
            'wednesday': {'activity': 'Swimming / Water Aerobics', 'duration': '30 mins', 'intensity': 'Low'},
            'thursday': {'activity': 'Walking', 'duration': '35 mins', 'intensity': 'Low'},
            'friday': {'activity': 'Light Cycling', 'duration': '25 mins', 'intensity': 'Low'},
            'saturday': {'activity': 'Yoga', 'duration': '30 mins', 'intensity': 'Low'},
            'sunday': {'activity': 'Rest Day', 'duration': '-', 'intensity': '-'}
        }
    elif bmi_status == "Overweight":
        plan = {
            'monday': {'activity': 'Brisk Walking / Jogging', 'duration': '40 mins', 'intensity': 'Moderate'},
            'tuesday': {'activity': 'Strength Training', 'duration': '30 mins', 'intensity': 'Moderate'},
            'wednesday': {'activity': 'Cycling', 'duration': '45 mins', 'intensity': 'Moderate'},
            'thursday': {'activity': 'HIIT', 'duration': '25 mins', 'intensity': 'High'},
            'friday': {'activity': 'Swimming', 'duration': '40 mins', 'intensity': 'Moderate'},
            'saturday': {'activity': 'Yoga / Flexibility', 'duration': '45 mins', 'intensity': 'Low'},
            'sunday': {'activity': 'Active Recovery / Walk', 'duration': '30 mins', 'intensity': 'Low'}
        }
    elif bmi_status == "Underweight":
        plan = {
            'monday': {'activity': 'Weightlifting (Upper)', 'duration': '45 mins', 'intensity': 'Moderate'},
            'tuesday': {'activity': 'Light Cardio', 'duration': '20 mins', 'intensity': 'Low'},
            'wednesday': {'activity': 'Weightlifting (Lower)', 'duration': '45 mins', 'intensity': 'Moderate'},
            'thursday': {'activity': 'Rest / Yoga', 'duration': '30 mins', 'intensity': 'Low'},
            'friday': {'activity': 'Weightlifting (Full Body)', 'duration': '50 mins', 'intensity': 'Moderate'},
            'saturday': {'activity': 'Swimming', 'duration': '30 mins', 'intensity': 'Low'},
            'sunday': {'activity': 'Rest Day', 'duration': '-', 'intensity': '-'}
        }
    else:  # Normal BMI
        plan = {
            'monday': {'activity': 'Running / Jogging', 'duration': '45 mins', 'intensity': 'Moderate-High'},
            'tuesday': {'activity': 'Strength Training', 'duration': '50 mins', 'intensity': 'High'},
            'wednesday': {'activity': 'HIIT / Cycling', 'duration': '40 mins', 'intensity': 'High'},
            'thursday': {'activity': 'Active Recovery / Yoga', 'duration': '30 mins', 'intensity': 'Low'},
            'friday': {'activity': 'Swimming / Sports', 'duration': '60 mins', 'intensity': 'Moderate'},
            'saturday': {'activity': 'Long Run / Hike', 'duration': '60-90 mins', 'intensity': 'Moderate'},
            'sunday': {'activity': 'Rest / Light Stretching', 'duration': '20 mins', 'intensity': 'Very Low'}
        }
    return plan
def get_nutrition_plan(calories, bmi_status, body_mass):
    """Generate nutrition recommendations"""
    if bmi_status == "Underweight":
        protein = round(body_mass * 2.0)  # Higher protein for muscle gain
        daily_calories = calories + 500  # Surplus for weight gain
        carbs = round((daily_calories * 0.5) / 4)
        fats = round((daily_calories * 0.25) / 9)
    elif bmi_status in ["Overweight", "Obese"]:
        protein = round(body_mass * 1.8)
        daily_calories = calories - 300  # Deficit for weight loss
        carbs = round((daily_calories * 0.35) / 4)
        fats = round((daily_calories * 0.25) / 9)
    else:
        protein = round(body_mass * 1.6)
        daily_calories = calories
        carbs = round((daily_calories * 0.45) / 4)
        fats = round((daily_calories * 0.25) / 9)
    return {
        'daily_calories': daily_calories,
        'protein': protein,
        'carbs': carbs,
        'fats': fats,
        'meals': [
            {'time': '7:00 AM', 'meal': 'Breakfast', 'suggestion': 'Oatmeal with fruits, eggs, and green tea'},
            {'time': '10:00 AM', 'meal': 'Snack', 'suggestion': 'Greek yogurt with nuts or protein shake'},
            {'time': '1:00 PM', 'meal': 'Lunch', 'suggestion': 'Grilled chicken, brown rice, and vegetables'},
            {'time': '4:00 PM', 'meal': 'Pre-Workout', 'suggestion': 'Banana with peanut butter'},
            {'time': '7:00 PM', 'meal': 'Dinner', 'suggestion': 'Fish/tofu, quinoa, and salad'},
            {'time': '9:00 PM', 'meal': 'Optional', 'suggestion': 'Casein protein or cottage cheese'}
        ]
    }
@app.route('/')
def home():
    stats = {
        'total_users': len(df),
        'avg_calories': int(df['Calories'].mean()),
        'max_calories': int(df['Calories'].max()),
        'avg_steps': int(df['Steps'].mean())
    }
    history = load_history()
    return render_template('index.html', stats=stats, history=history[-10:][::-1])  # Last 10, newest first
@app.route('/history')
def get_history():
    """Get prediction history as JSON"""
    history = load_history()
    return jsonify(history[::-1])  # Newest first
@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear all prediction history"""
    save_history([])
    return jsonify({'success': True, 'message': 'History cleared'})
@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_id = int(request.form.get('user_id', 1))
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        body_mass = int(request.form['body_mass'])
        height = int(request.form['height'])
        steps = int(request.form['steps'])
        workout_minutes = int(request.form['workout_minutes'])
        exercise_type = int(request.form['exercise_type'])
        avg_heart_rate = int(request.form['avg_heart_rate'])
        intensity_score = int(request.form['intensity_score'])
        features = np.array([[user_id, age, gender, body_mass, height, steps, 
                             workout_minutes, exercise_type, avg_heart_rate, intensity_score]])
        prediction = model_svc.predict(features)[0]
        calories = int(prediction)
        bmi = round(body_mass / ((height/100) ** 2), 1)
        if bmi < 18.5:
            bmi_status = "Underweight"
            bmi_color = "#3498db"
        elif bmi < 25:
            bmi_status = "Normal"
            bmi_color = "#10b981"
        elif bmi < 30:
            bmi_status = "Overweight"
            bmi_color = "#f59e0b"
        else:
            bmi_status = "Obese"
            bmi_color = "#ef4444"
        if calories > 2500:
            fitness_level = "Elite Athlete"
            fitness_icon = "trophy"
            fitness_score = 95
        elif calories > 2000:
            fitness_level = "Very Active"
            fitness_icon = "bolt"
            fitness_score = 80
        elif calories > 1500:
            fitness_level = "Active"
            fitness_icon = "running"
            fitness_score = 65
        else:
            fitness_level = "Moderate"
            fitness_icon = "walking"
            fitness_score = 50
        ai_data = get_ai_suggestions(calories, bmi, bmi_status, steps, workout_minutes, 
                                      avg_heart_rate, age, exercise_type)
        weekly_plan = get_weekly_plan(exercise_type, fitness_level, bmi_status)
        nutrition = get_nutrition_plan(calories, bmi_status, body_mass)
        health_score = min(100, int(
            (min(steps, 10000) / 10000 * 25) +
            (min(workout_minutes, 60) / 60 * 25) +
            (25 if 18.5 <= bmi <= 25 else 10) +
            (min(calories, 2500) / 2500 * 25)
        ))
        stats = {
            'total_users': len(df),
            'avg_calories': int(df['Calories'].mean()),
            'max_calories': int(df['Calories'].max()),
            'avg_steps': int(df['Steps'].mean())
        }
        result = {
            'calories': calories,
            'bmi': bmi,
            'bmi_status': bmi_status,
            'bmi_color': bmi_color,
            'fitness_level': fitness_level,
            'fitness_icon': fitness_icon,
            'fitness_score': fitness_score,
            'workout_mins': workout_minutes,
            'steps': steps,
            'heart_rate': avg_heart_rate,
            'age': age,
            'gender': 'Male' if gender == 0 else 'Female',
            'body_mass': body_mass,
            'height': height,
            'exercise_type': EXERCISE_TYPES.get(exercise_type, 'Unknown'),
            'intensity_score': intensity_score,
            'health_score': health_score,
            'date': datetime.now().strftime("%B %d, %Y"),
            'time': datetime.now().strftime("%I:%M %p")
        }
        history_entry = {
            'age': age,
            'gender': 'Male' if gender == 0 else 'Female',
            'body_mass': body_mass,
            'height': height,
            'steps': steps,
            'workout_mins': workout_minutes,
            'exercise_type': EXERCISE_TYPES.get(exercise_type, 'Unknown'),
            'heart_rate': avg_heart_rate,
            'intensity': intensity_score,
            'calories': calories,
            'bmi': bmi,
            'bmi_status': bmi_status,
            'health_score': health_score,
            'fitness_level': fitness_level
        }
        add_to_history(history_entry)
        history = load_history()
        return render_template('index.html', 
                             result=result,
                             stats=stats,
                             ai_data=ai_data,
                             weekly_plan=weekly_plan,
                             nutrition=nutrition,
                             history=history[-10:][::-1],
                             show_result=True)
    except Exception as e:
        stats = {
            'total_users': len(df),
            'avg_calories': int(df['Calories'].mean()),
            'max_calories': int(df['Calories'].max()),
            'avg_steps': int(df['Steps'].mean())
        }
        return render_template('index.html', 
                             error=str(e),
                             stats=stats,
                             show_result=True)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for predictions"""
    try:
        data = request.get_json()
        features = np.array([[
            data.get('user_id', 1),
            data['age'],
            data['gender'],
            data['body_mass'],
            data['height'],
            data['steps'],
            data['workout_minutes'],
            data['exercise_type'],
            data['avg_heart_rate'],
            data['intensity_score']
        ]])
        prediction = model_svc.predict(features)[0]
        calories = int(prediction)
        bmi = round(data['body_mass'] / ((data['height']/100) ** 2), 1)
        return jsonify({
            'success': True,
            'prediction': {
                'calories': calories,
                'bmi': bmi,
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400
@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    """Generate comprehensive PDF report with analysis"""
    try:
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        body_mass = int(request.form['body_mass'])
        height = int(request.form['height'])
        steps = int(request.form['steps'])
        workout_minutes = int(request.form['workout_minutes'])
        exercise_type = int(request.form['exercise_type'])
        avg_heart_rate = int(request.form['avg_heart_rate'])
        intensity_score = int(request.form['intensity_score'])
        features = np.array([[1, age, gender, body_mass, height, steps, 
                             workout_minutes, exercise_type, avg_heart_rate, intensity_score]])
        prediction = model_svc.predict(features)[0]
        calories = int(prediction)
        bmi = round(body_mass / ((height/100) ** 2), 1)
        if bmi < 18.5:
            bmi_status = "Underweight"
        elif bmi < 25:
            bmi_status = "Normal"
        elif bmi < 30:
            bmi_status = "Overweight"
        else:
            bmi_status = "Obese"
        if calories > 2500:
            fitness_level = "Elite Athlete"
            fitness_score = 95
        elif calories > 2000:
            fitness_level = "Very Active"
            fitness_score = 80
        elif calories > 1500:
            fitness_level = "Active"
            fitness_score = 65
        else:
            fitness_level = "Moderate"
            fitness_score = 50
        health_score = min(100, int(
            (min(steps, 10000) / 10000 * 25) +
            (min(workout_minutes, 60) / 60 * 25) +
            (25 if 18.5 <= bmi <= 25 else 10) +
            (min(calories, 2500) / 2500 * 25)
        ))
        fat_percentage = max(5, min(40, 10 + (bmi - 18.5) * 2))  # Estimated body fat
        whr = round(0.7 + (bmi - 20) * 0.01, 2) if bmi > 20 else 0.7  # Waist-to-hip ratio estimate
        if fitness_score >= 90:
            grade = "A+"
        elif fitness_score >= 80:
            grade = "A"
        elif fitness_score >= 70:
            grade = "B"
        elif fitness_score >= 60:
            grade = "C"
        else:
            grade = "D"
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
        elements = []
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle', 
            parent=styles['Heading1'], 
            fontSize=28, 
            textColor=colors.HexColor('#e74c3c'),
            alignment=TA_CENTER,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#34495e'),
            alignment=TA_CENTER,
            spaceAfter=6,
            fontName='Helvetica-Oblique'
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceBefore=20,
            spaceAfter=12,
            fontName='Helvetica-Bold',
            backColor=colors.HexColor('#ecf0f1'),
            leftIndent=10,
            borderPadding=8
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=8,
            fontName='Helvetica'
        )
        elements.append(Spacer(1, 80))
        drawing = Drawing(500, 60)
        drawing.add(Circle(250, 30, 25, fillColor=colors.HexColor('#e74c3c'), strokeColor=colors.HexColor('#c0392b'), strokeWidth=3))
        elements.append(drawing)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("🔥 FitBurn AI", title_style))
        elements.append(Paragraph("Personalized Fitness Analysis Report", subtitle_style))
        elements.append(Spacer(1, 40))
        report_info = [
            ['Report ID:', f'FB-{datetime.now().strftime("%Y%m%d-%H%M")}'],
            ['Generated:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Model Accuracy:', '99.26%']
        ]
        info_table = Table(report_info, colWidths=[120, 250])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#3498db')),
            ('LINEABOVE', (0, 1), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 60))
        quote_text = Paragraph(
            '<i>"The only bad workout is the one that didn\'t happen."</i>',
            ParagraphStyle('quote', parent=styles['Normal'], fontSize=12, 
                         textColor=colors.HexColor('#7f8c8d'), alignment=TA_CENTER)
        )
        elements.append(quote_text)
        elements.append(PageBreak())
        elements.append(Paragraph("📋 Personal Information", heading_style))
        elements.append(Spacer(1, 10))
        personal_data = [
            ['<b>Parameter</b>', '<b>Value</b>', '<b>Parameter</b>', '<b>Value</b>'],
            ['Age', f'{age} years', 'Gender', 'Male 👨' if gender == 0 else 'Female 👩'],
            ['Weight', f'{body_mass} kg', 'Height', f'{height} cm'],
            ['Exercise', EXERCISE_TYPES.get(exercise_type, 'Unknown'), 'Intensity', f'{intensity_score}/10 ⭐'],
            ['Heart Rate', f'{avg_heart_rate} BPM ❤️', 'Duration', f'{workout_minutes} mins ⏱️']
        ]
        personal_table = Table(personal_data, colWidths=[100, 120, 100, 120])
        personal_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 1), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')])
        ]))
        elements.append(personal_table)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("🎯 Prediction Results", heading_style))
        elements.append(Spacer(1, 15))
        cal_data = [[Paragraph(f'<font size=36 color="#e74c3c"><b>{calories:.0f}</b></font><br/><font size=12 color="#7f8c8d">Calories Burned</font>', 
                              ParagraphStyle('cal', alignment=TA_CENTER))]]
        cal_table = Table(cal_data, colWidths=[440])
        cal_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffe5e5')),
            ('BOX', (0, 0), (-1, -1), 3, colors.HexColor('#e74c3c')),
            ('TOPPADDING', (0, 0), (-1, -1), 25),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 25),
        ]))
        elements.append(cal_table)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("📊 Health Metrics", heading_style))
        elements.append(Spacer(1, 10))
        metrics_data = [
            ['<b>Metric</b>', '<b>Value</b>', '<b>Metric</b>', '<b>Value</b>'],
            ['BMI', f'{bmi:.1f}', 'Status', f'{bmi_status} ✓'],
            ['Fat %', f'{fat_percentage:.1f}%', 'WHR', f'{whr:.2f}'],
            ['Fitness Score', f'{fitness_score}/100 🎯', 'Grade', f'{grade} ⭐']
        ]
        metrics_table = Table(metrics_data, colWidths=[110, 110, 110, 110])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 1), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f8f5')])
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("💡 Recommendations", heading_style))
        elements.append(Spacer(1, 10))
        recommendations = []
        if bmi < 18.5:
            recommendations.append("• Increase calorie intake by 300-500 calories daily")
            recommendations.append("• Focus on protein-rich foods")
            recommendations.append("• Add strength training 3-4 times per week")
        elif bmi >= 25:
            recommendations.append("• Increase cardio to 30-45 minutes, 4-5 times per week")
            recommendations.append("• Reduce processed foods, increase vegetables and fruits")
            recommendations.append("• Consider HIIT training")
        else:
            recommendations.append("• Continue your balanced routine")
            recommendations.append("• Focus on consistency over intensity")
        if steps < 8000:
            recommendations.append(f"• Increase daily steps to 10,000")
        if workout_minutes < 30:
            recommendations.append("• Extend workout sessions to 45-60 minutes")
        recommendations.append(f"• Drink at least {round((calories/1000)*1.5, 1)}L water daily")
        recommendations.append("• Ensure 7-8 hours of sleep for recovery")
        for rec in recommendations:
            elements.append(Paragraph(rec, normal_style))
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Weekly Workout Plan", heading_style))
        elements.append(Spacer(1, 10))
        weekly_data = [
            ['Day', 'Activity', 'Duration'],
            ['Monday', 'Cardio', '30-45 mins'],
            ['Tuesday', 'Strength Training', '30-45 mins'],
            ['Wednesday', 'Active Recovery', '20-30 mins'],
            ['Thursday', 'Cardio', '30-45 mins'],
            ['Friday', 'Strength Training', '30-45 mins'],
            ['Saturday', 'Light Activity', '30 mins'],
            ['Sunday', 'Rest', '-']
        ]
        weekly_table = Table(weekly_data, colWidths=[100, 200, 140])
        weekly_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(weekly_table)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Nutrition Plan", heading_style))
        elements.append(Spacer(1, 10))
        if bmi_status == "Underweight":
            protein = round(body_mass * 2.0)
            daily_cal = calories + 500
        elif bmi_status in ["Overweight", "Obese"]:
            protein = round(body_mass * 1.8)
            daily_cal = calories - 300
        else:
            protein = round(body_mass * 1.6)
            daily_cal = calories
        carbs = round((daily_cal * 0.45) / 4)
        fats = round((daily_cal * 0.25) / 9)
        nutrition_data = [
            ['Calories', 'Protein', 'Carbs', 'Fats'],
            [f'{daily_cal} kcal', f'{protein}g', f'{carbs}g', f'{fats}g']
        ]
        nutrition_table = Table(nutrition_data, colWidths=[110, 110, 110, 110])
        nutrition_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        elements.append(nutrition_table)
        elements.append(Spacer(1, 20))
        elements.append(Spacer(1, 30))
        footer_text = Paragraph(
            '<i>This report is generated by FitBurn AI. Consult a healthcare professional before making significant changes to your fitness routine.</i>',
            ParagraphStyle('footer', parent=normal_style, fontSize=8, 
                         textColor=colors.grey, alignment=TA_CENTER)
        )
        elements.append(footer_text)
        doc.build(elements)
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'FitBurn_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True, port=5000)
