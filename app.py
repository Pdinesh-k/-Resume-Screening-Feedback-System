import streamlit as st
import os
import requests
import json
import spacy
import PyPDF2
import docx
import pandas as pd
from dotenv import load_dotenv
import re
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from queue import Queue
from threading import Lock
import asyncio

# Load environment variables
load_dotenv()

# Initialize Groq API settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please check your .env file.")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"

# API Rate Limiting
MAX_RETRIES = 3
INITIAL_WAIT = 5  # seconds
MAX_WAIT = 20  # seconds
MIN_TIME_BETWEEN_REQUESTS = 2  # seconds

# Global state for rate limiting
last_request_time = 0
request_lock = Lock()

def wait_for_rate_limit():
    """Ensure minimum time between API requests"""
    global last_request_time
    with request_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < MIN_TIME_BETWEEN_REQUESTS:
            time.sleep(MIN_TIME_BETWEEN_REQUESTS - time_since_last)
        last_request_time = time.time()

@retry(stop=stop_after_attempt(MAX_RETRIES), 
       wait=wait_exponential(multiplier=INITIAL_WAIT, max=MAX_WAIT))
def call_groq_api(data, headers):
    """Make API call with rate limiting and retry logic"""
    wait_for_rate_limit()
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        
        if response.status_code == 429:  # Too Many Requests
            retry_after = int(response.headers.get('Retry-After', INITIAL_WAIT))
            st.warning(f"Rate limit reached. Waiting {retry_after} seconds before retry...")
            time.sleep(retry_after)
            raise Exception("Rate limit reached")
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        raise

def groq_completion(prompt, task_name="Analysis"):
    """Get completion from Groq API with improved rate limiting"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert resume analyzer. Return ONLY valid JSON without any additional text."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
        "response_format": { "type": "json_object" }
    }
    
    try:
        with st.spinner(f'{task_name} in progress... (This may take a few moments)'):
            response_data = call_groq_api(data, headers)
            content = response_data['choices'][0]['message']['content']
            
            # Ensure we have valid JSON
            if isinstance(content, str):
                content = content.strip()
                # Remove any non-JSON text before the first {
                start_idx = content.find('{')
                if start_idx != -1:
                    content = content[start_idx:]
                # Remove any non-JSON text after the last }
                end_idx = content.rfind('}')
                if end_idx != -1:
                    content = content[:end_idx + 1]
            
            # Parse JSON to validate it
            return json.loads(content)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse API response as JSON: {str(e)}")
        return None
    except Exception as e:
        st.error(f"API error: {str(e)}")
        if "rate limit" in str(e).lower():
            st.info("The system is currently busy. Please try again in a few moments.")
        return None

def analyze_resume(text):
    """Analyze resume using LLM"""
    return groq_completion(
        prompt=f"""Analyze this resume text and extract key information:

Resume Text:
{text}

Expected JSON format:
{{
    "skills": {{
        "programming_languages": [],
        "frameworks_libraries": [],
        "databases": [],
        "cloud_platforms": [],
        "tools_technologies": [],
        "soft_skills": []
    }},
    "experience": {{
        "internships": [
            {{
                "company": "",
                "role": "",
                "duration": "",
                "technologies_used": [],
                "key_achievements": []
            }}
        ],
        "projects": [
            {{
                "name": "",
                "description": "",
                "technologies": [],
                "key_features": [],
                "impact": ""
            }}
        ]
    }},
    "education": [
        {{
            "degree": "",
            "institution": "",
            "duration": "",
            "major": "",
            "achievements": []
        }}
    ]
}}""",
        task_name="Resume Analysis"
    )

def analyze_resume_with_job(resume_text, job_description):
    """Analyze resume against job description"""
    return groq_completion(
        prompt=f"""You are an expert resume analyzer. Carefully analyze how well the given resume matches the job description. Consider:
1. Required skills vs candidate's skills
2. Years of experience required vs actual experience
3. Education requirements vs candidate's education
4. Required certifications vs candidate's certifications
5. Soft skills and cultural fit

Resume:
{resume_text}

Job Description:
{job_description}

Analyze thoroughly and return a JSON response with:
1. An accurate match percentage based on requirements met
2. Actually found matching skills/experience from the resume
3. Real missing requirements from the job description
4. True experience comparison
5. Specific, actionable recommendations

Return this exact JSON format:
{{
    "match_percentage": <calculate real percentage based on matched requirements>,
    "key_matches": [
        {{
            "skill": "<skill/requirement found in BOTH job description AND resume>",
            "context": "<explain where and how it was found in the resume>"
        }}
    ],
    "missing_skills": [
        {{
            "skill": "<skill/requirement in job description BUT NOT in resume>",
            "importance": "<Required/Preferred based on job description>",
            "suggestion": "<specific actionable suggestion to acquire this skill>"
        }}
    ],
    "experience_match": {{
        "years_required": "<extract from job description>",
        "years_found": "<calculate from resume>",
        "match_level": "<Excellent/Good/Fair/Insufficient based on comparison>"
    }},
    "recommendations": [
        {{
            "area": "<specific area from resume that needs improvement>",
            "suggestion": "<detailed, actionable suggestion>",
            "priority": "<High/Medium/Low based on job requirements>"
        }}
    ]
}}""",
        task_name="Job Match Analysis"
    )

def generate_cover_letter(resume_text, job_description, analysis):
    """Generate a tailored cover letter"""
    return groq_completion(
        prompt=f"""Generate a compelling cover letter based on this match:

Resume:
{resume_text}

Job Description:
{job_description}

Analysis:
{json.dumps(analysis)}

Return this exact JSON format:
{{
    "salutation": "Dear Hiring Manager",
    "introduction": "First paragraph introducing yourself and expressing interest",
    "body": [
        "Paragraph about relevant skills and experience",
        "Paragraph about specific achievements and their relevance",
        "Paragraph about company fit and enthusiasm"
    ],
    "closing": "Thank you paragraph and call to action",
    "signature": "Best regards,\\n[Candidate Name]"
}}""",
        task_name="Cover Letter Generation"
    )

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
    return text

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    text = ""
    try:
        doc = docx.Document(docx_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        st.error(f"DOCX extraction error: {str(e)}")
    return text

def display_analysis(analysis):
    """Display the resume analysis in a structured format"""
    if not analysis:
        return

    # Skills Section
    st.header("🎯 Skills")
    skills = analysis['skills']
    cols = st.columns(2)
    
    with cols[0]:
        st.subheader("Technical Skills")
        if skills['programming_languages']:
            st.write("**Programming Languages:**")
            st.write(", ".join(skills['programming_languages']))
        if skills['frameworks_libraries']:
            st.write("**Frameworks & Libraries:**")
            st.write(", ".join(skills['frameworks_libraries']))
        if skills['databases']:
            st.write("**Databases:**")
            st.write(", ".join(skills['databases']))
        if skills['cloud_platforms']:
            st.write("**Cloud Platforms:**")
            st.write(", ".join(skills['cloud_platforms']))
        if skills['tools_technologies']:
            st.write("**Tools & Technologies:**")
            st.write(", ".join(skills['tools_technologies']))
    
    with cols[1]:
        if skills['soft_skills']:
            st.subheader("Soft Skills")
            st.write(", ".join(skills['soft_skills']))

    # Experience Section
    st.header("💼 Experience")
    
    # Internships
    if analysis['experience']['internships']:
        st.subheader("Internships")
        for internship in analysis['experience']['internships']:
            with st.expander(f"{internship['role']} at {internship['company']}"):
                st.write(f"**Duration:** {internship['duration']}")
                st.write("**Technologies Used:**")
                st.write(", ".join(internship['technologies_used']))
                st.write("**Key Achievements:**")
                for achievement in internship['key_achievements']:
                    st.write(f"• {achievement}")

    # Projects
    if analysis['experience']['projects']:
        st.subheader("Projects")
        for project in analysis['experience']['projects']:
            with st.expander(project['name']):
                st.write(f"**Description:** {project['description']}")
                st.write("**Technologies:**")
                st.write(", ".join(project['technologies']))
                st.write("**Key Features:**")
                for feature in project['key_features']:
                    st.write(f"• {feature}")
                if project['impact']:
                    st.write(f"**Impact:** {project['impact']}")

    # Education Section
    st.header("🎓 Education")
    for edu in analysis['education']:
        st.write(f"**{edu['degree']}** - {edu['institution']}")
        st.write(f"Duration: {edu['duration']}")
        if edu['major']:
            st.write(f"Major: {edu['major']}")
        if edu['achievements']:
            st.write("Achievements:")
            for achievement in edu['achievements']:
                st.write(f"• {achievement}")

def display_job_match(analysis):
    """Display job match analysis in a structured format"""
    if not analysis:
        return

    # Match Overview
    st.header("🎯 Job Match Analysis")
    st.write(f"**Overall Match:** {analysis['match_percentage']}%")
    
    # Key Matches
    if analysis['key_matches']:
        st.subheader("✅ Strong Matches")
        for match in analysis['key_matches']:
            st.write(f"**{match['skill']}:** {match['context']}")
    
    # Missing Skills
    if analysis['missing_skills']:
        st.subheader("🔍 Areas for Development")
        for skill in analysis['missing_skills']:
            st.write(f"**{skill['skill']}** ({skill['importance']})")
            st.write(f"💡 *Suggestion:* {skill['suggestion']}")
    
    # Experience Match
    st.subheader("⏳ Experience Match")
    exp = analysis['experience_match']
    st.write(f"Required: {exp['years_required']}")
    st.write(f"Found: {exp['years_found']}")
    st.write(f"Match Level: {exp['match_level']}")
    
    # Recommendations
    if analysis['recommendations']:
        st.subheader("📝 Recommendations")
        for rec in analysis['recommendations']:
            with st.expander(f"{rec['area']} (Priority: {rec['priority']})"):
                st.write(rec['suggestion'])

def display_cover_letter(cover_letter):
    """Display the generated cover letter"""
    if not cover_letter:
        return

    st.header("📨 Generated Cover Letter")
    
    # Display the cover letter in a professional format
    st.write(cover_letter['salutation'])
    st.write("")
    st.write(cover_letter['introduction'])
    st.write("")
    for paragraph in cover_letter['body']:
        st.write(paragraph)
        st.write("")
    st.write(cover_letter['closing'])
    st.write("")
    st.write(cover_letter['signature'])
    
    # Add download button
    letter_text = f"{cover_letter['salutation']}\n\n"
    letter_text += f"{cover_letter['introduction']}\n\n"
    letter_text += "\n\n".join(cover_letter['body'])
    letter_text += f"\n\n{cover_letter['closing']}\n\n"
    letter_text += cover_letter['signature']
    
    st.download_button(
        label="Download Cover Letter",
        data=letter_text,
        file_name="cover_letter.txt",
        mime="text/plain"
    )

def main():
    st.title("AI Resume Analyzer")
    st.write("Upload your resume and enter job description for detailed analysis")

    uploaded_file = st.file_uploader("Choose a resume file", type=['pdf', 'docx'])
    job_description = st.text_area("Enter the job description", height=200)
    
    if uploaded_file and job_description:
        with st.spinner('Reading file...'):
            # Extract text based on file type
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            else:
                text = extract_text_from_docx(uploaded_file)
        
        if text:
            # First analyze the resume
            resume_analysis = analyze_resume(text)
            
            if resume_analysis:
                # Show resume analysis
                display_analysis(resume_analysis)
                
                # Analyze job match
                match_analysis = analyze_resume_with_job(text, job_description)
                if match_analysis:
                    display_job_match(match_analysis)
                    
                    # Generate cover letter
                    if st.button("Generate Cover Letter"):
                        cover_letter = generate_cover_letter(text, job_description, match_analysis)
                        if cover_letter:
                            display_cover_letter(cover_letter)
            else:
                st.error("Failed to analyze resume. Please try again.")

if __name__ == "__main__":
    main()
